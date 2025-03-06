# NCCL

[Link](https://www.bilibili.com/video/BV1GrF3eyE4T?spm_id_from=333.788.videopod.sections&vd_source=a88e8ba2459d4c8f2a1c52027ba8f422)

## RDMA

所有的通过RDMA来通讯

- Broadcast
  - root节点把输入广播到所有节点
- AllGather
  - 在通信组中的 rank id 来确定它在环形拓扑中的位置。具体来说，每个 GPU 的 rank id 表示它在整个集群中的顺序，然后在 Allgather 的过程中，每个 GPU 会按照这个顺序，将自己的数据发送给下一个 GPU（例如，rank i 发送给 rank (i+1) mod N），并同时从前一个 GPU（rank (i-1) mod N）接收数据
- Scatter
  - root节点的数据均分并散布至其他rank
- Reduce
  - 在 Reduce 操作中，每个 GPU（或进程）都会发送自己的数据，然后根据指定的规约算子（例如求和、求最大值、求最小值等）将所有数据进行合并，最终将结果发送到指定的根（root） GPU 上
- AllReduce
  - 在 Reduce 操作中，每个 GPU（或进程）都会发送自己的数据，然后根据指定的规约算子（例如求和、求最大值、求最小值等）将所有数据进行合并，最终将结果发送到指定的根（root） GPU 上
- ReduceScatter
  - ReduceScatter 则是在执行规约操作后，将聚合结果分散（scatter）到各个进程，每个进程仅获得结果的一部分。

# 分布式训练 

## 数据并行 Data Parallel

不同样本在多个设备进行计算

通信模式：AllReduce同步提督

通信量与模型规模正相关，单卡达10GB+

一个step一次通信

## 流水线并行 Pipeline Parallel

多个子模型在多个设备的技术

通信模式：点到点，正向传激活，反向传梯度

通信量与层间交互相关，一般在MB级别

一个step要几十次的通信

## 张量并行 Tensor Parallel

多个模型在多个设备的技术

通信模式： AllReduce同步矩阵计算结果

通信量和batchsize有关，矩阵可达GB级别

一个step几十次通信

## 专家并行 Expert Parallel

不同Token在多个设备上的计算