**1 verl/trainer/main_ppo.py**
1) verl/trainer/main_ppo.py
![image](https://github.com/user-attachments/assets/81b7a6c0-767c-4f1d-8916-6362f60e3ffc)

这里初始化role_worker_mapping, 对应的worker是哪个

### 问题1：ray.remote的作用是什么，ActorRollRefWorker都是class

`ray.remote(A)` 是 **Ray** 框架中的一个重要操作，它的作用是将类 `A` 标记为 **远程类**，使得 `A` 的实例可以在 Ray 的分布式环境中运行。具体来说，`ray.remote(A)` 会返回一个 **远程类** 的引用，你可以通过这个引用在 Ray 集群中创建 `A` 的实例，并调用其方法。

2) verl/trainer/main_ppo.py
 ![image](https://github.com/user-attachments/assets/efbe7984-332b-4d42-9b7e-37d1d3ee92a5)
  I) class ResourcePoolManager(verl/trainer/ppo/ray_trainer.py)
    class ResourcePoolManager(verl/trainer/ppo/ray_trainer.py)
3)RayPROTrainer init( verl/trainer/main_ppo.py)
![image](https://github.com/user-attachments/assets/b249cfe2-ee32-45f4-993b-628c19c1fe9f)

4) verl/trainer/main_ppo.py   trainer.init_workers() → verl/trainer/ppo/ray_trainer.py
  I)初始化Actor, Critic, Ref, RM worker
   ![image](https://github.com/user-attachments/assets/10e24206-83a6-4b20-a364-3aaa221eccfb)
   ![image](https://github.com/user-attachments/assets/91ec0fa5-99cf-40e0-b7ea-258b1fb2e011)
  line 672的self.respource_pool_to_cls是RayWorkerGroup(from verl.single_controller.ray import RayWorkerGroup， config.actor_rollout_ref.actor.strategy == 'fsdp')

  II)ActorRolloutRefWorker, CriticWorker,  RewardModelWorker的init_model
    a)先了解fsdp 
    
    ```
    1 FSDP 的核心思想
    FSDP 的核心思想是 完全分片（Fully Sharded），即：
    - 模型参数分片：将模型的每一层参数分片到多个 GPU 上，而不是在每个 GPU 上保存完整的模型副本。
    - 梯度分片：在反向传播时，每个 GPU 只计算自己分片部分的梯度。
    - 优化器状态分片：优化器的状态（如动量、Adam 的二阶矩估计等）也分片到多个 GPU 上。
    
    通过这种方式，FSDP 可以显著减少单个 GPU 的显存占用，从而支持更大规模的模型训练。
    2 FSDP 的使用方法
 
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.nn import Linear
    
    #初始化分布式环境
    dist.init_process_group(backend="nccl")
    
    #定义一个简单的模型
    class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(1024, 1024)
        self.layer2 = Linear(1024, 1024)
    
    def forward(self, x):
        return self.layer2(self.layer1(x))
    
    #创建模型
    model = MyModel().cuda()
    
    #使用 FSDP 包装模型
    sharding_strategy = ShardingStrategy.FULL_SHARD  #完全分片策略
    model = FSDP(model, sharding_strategy=sharding_strategy)
    
    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    #训练步骤
    inputs = torch.randn(32, 1024).cuda()
    outputs = model(inputs)
    loss = outputs.sum()
    loss.backward()
    optimizer.step()

    ```
    III) ActorRolloutRefWorker::init
    ![image](https://github.com/user-attachments/assets/37827aef-26d5-4943-ad2a-f517e723137b)

2 RayRPOTrainer::fit(verl/trainer/ppo/ray_trainer.py)
 
 1) line 828 batch: DataProto = DataProto.from_single_dict(batch_dict)
   
   line 831 gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
  ![image](https://github.com/user-attachments/assets/284eaa51-a21e-4d56-815f-b1cba6b46fa5)




