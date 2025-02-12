# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    print(f"1 vllm_rollout.py, _pre_process_inputs, len(token_ids): {len(token_ids)} and pad_token_id: {pad_token_id}")
    return token_ids
    """
    这个 `_pre_process_inputs` 函数的作用是对输入的 `prompt_token_ids` 进行预处理，具体来说，是**去除左侧的填充（padding）部分**，并返回一个不包含填充的 token ID 列表。以下是对代码的详细解释：

    ---

    ### **函数功能**
    1. **输入**：
    - `pad_token_id`：填充 token 的 ID（通常是 `tokenizer.pad_token_id`）。
    - `prompt_token_ids`：一个包含 token ID 的 PyTorch 张量（`torch.Tensor`），通常是经过填充的 token ID 序列。

    2. **输出**：
    - 返回一个不包含左侧填充的 token ID 列表（`List[int]`）。

    ---

    ### **代码逐行解释**

    #### 1. **`non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]`**
    - **作用**：
    - 找到第一个非填充 token 的索引。
    - **实现细节**：
    - `prompt_token_ids != pad_token_id`：生成一个布尔张量，标记哪些位置的 token 不是填充 token。
    - `torch.nonzero(...)`：返回所有非零元素（即非填充 token）的索引。
    - `[0][0]`：取第一个非填充 token 的索引。

    #### 2. **`token_ids = prompt_token_ids[non_pad_index:].tolist()`**
    - **作用**：
    - 从第一个非填充 token 开始，截取 `prompt_token_ids` 的剩余部分，并将其转换为 Python 列表。
    - **实现细节**：
    - `prompt_token_ids[non_pad_index:]`：从 `non_pad_index` 开始截取张量，去除左侧的填充部分。
    - `.tolist()`：将 PyTorch 张量转换为 Python 列表。

    #### 3. **`return token_ids`**
    - **作用**：
    - 返回不包含左侧填充的 token ID 列表。

    ---

    ### **示例**
    假设：
    - `pad_token_id = 0`
    - `prompt_token_ids = torch.tensor([0, 0, 0, 1, 2, 3, 0, 0])`

    #### 执行过程：
    1. **找到第一个非填充 token 的索引**：
    - `prompt_token_ids != pad_token_id` 的结果是 `[False, False, False, True, True, True, False, False]`。
    - `torch.nonzero(...)` 的结果是 `[[3], [4], [5]]`。
    - `[0][0]` 的结果是 `3`（第一个非填充 token 的索引）。

    2. **截取并转换**：
    - `prompt_token_ids[3:]` 的结果是 `[1, 2, 3, 0, 0]`。
    - `.tolist()` 的结果是 `[1, 2, 3, 0, 0]`。

    3. **返回值**：
    - 返回 `[1, 2, 3, 0, 0]`。

    ---

    ### **优化建议**
    根据代码中的注释（`NOTE(sgm)`），当前的实现是为了处理数据加载器（`dataloader`）返回的填充后的 token ID 序列。如果数据加载器可以直接返回未填充的 token ID 列表（`List[int]`），则可以避免这个预处理步骤，从而提高效率。

    ---

    ### **总结**
    `_pre_process_inputs` 函数的作用是去除输入 token ID 序列中的左侧填充部分，并返回一个不包含填充的 token ID 列表。它的主要应用场景是处理经过填充的 token ID 序列，以便后续的模型处理可以忽略填充部分。

    """


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        print(f"1 vllm_rollout.py, vLLMRollout::init, self.sampling_params: {self.sampling_params}")

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        print(f"0 vllm_rollout.py, generate_sequences, type(idx): {type(idx)}")
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        print(f"1 vllm_rollout.py, generate_sequences, type(attention_mask): {type(attention_mask)}")
        position_ids = prompts.batch['position_ids']
        print(f"2 vllm_rollout.py, generate_sequences, type(position_ids): {type(position_ids)}")
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            if i<=2:
                print(f"3 vllm_rollout.py, generate_sequences, idx[i]: {idx[i]}")
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
