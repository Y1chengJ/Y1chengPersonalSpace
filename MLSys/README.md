# 计算图

## 静态图动态图生成
### 静态图
使用前端语言定义模型形成完整的程序表达后，机器学习框架首先对神经网络模型进行分析，获取网络层之间的连接拓扑关系以及参数变量设置、损失函数等信息。然后机器学习框架会将完整的模型描述编译为可被后端计算硬件调用执行的固定代码文本，这种固定代码文本通常被称为静态计算图。当使用静态计算图进行模型训练或者推理过程时，无需编译前端语言模型。静态计算图直接接收数据并通过相应硬件调度执行图中的算子来完成任务。静态计算图可以通过优化策略转换成等价的更加高效的结构，提高后端硬件的计算效率。

### 动态图
动态图原采用解析式的执行方式，其核心特点是编译与执行同时发生。动态图采用前端语言自身的解释器对代码进行解析，利用机器学习框架本身的算子分发功能，算子会即刻执行并输出结果。动态图模式采用用户友好的命令式编程范式，使用前端语言构建神经网络模型更加简洁，深受广大深度学习研究者青睐。

### 静态图与动态图转换
基于追踪转换：以动态图模式执行并记录调度的算子，构建和保存为静态图模型。

基于源码转换：分析前端代码来将动态图代码自动转写为静态图代码，并在底层自动帮用户使用静态图执行器运行。
```python
torch.jit.script() # 基于源码转换，支持控制流（if-else、for、while）
torch.jit.trace() # 基于追踪转换，不支持控制流
```

## 计算图算子调度

### 算子调度执行
算子的执行调度包含两个步骤，第一步，根据拓扑排序算法，将计算图进行拓扑排序得到线性的算子调度序列；第二步，将序列中的算子分配到指令流进行运算，尽可能将序列中的算子并行执行，提高计算资源的利用率。

生成调度序列之后，需要将序列中的算子与数据分发到指定的GPU/NPU上执行运算。根据算子依赖关系和计算设备数量，可以将无相互依赖关系的算子分发到不同的计算设备，同时执行运算，这一过程称之为并行计算，与之相对应的按照序贯顺序在同一设备执行运算被称为串行计算。在深度学习中，当数据集和参数量的规模越来越大在分发数据与算子时通信消耗会随之而增加，计算设备会在数据传输的过程中处于闲置状态。此时采用同步与异步的任务调度机制可以更好的协调通信与训练任务，提高通信模块与计算设备的使用率，在后续的小节中将详细介绍串行与并行、同步与异步的概念。

### 串行并行计算
串行：队列中的任务必须按照顺序进行调度执行直至队列结束；
并行：队列中的任务可以同时进行调度执行，加快执行效率。

### 同步异步计算
一次完整计算图的训练执行过程包含：数据载入、数据预处理、网络训练三个环节。三个环节之间的任务调度是以串行方式进行，每一个环节都有赖于前一个环节的输出。但计算图的训练是多轮迭代的过程，多轮训练之间的三个环节可以用同步与异步两种机制来进行调度执行。
- 同步：顺序执行任务，当前任务执行完后会等待后续任务执行情况，任务之间需要等待、协调运行；
- 异步：当前任务完成后，不需要等待后续任务的执行情况，可继续执行当前任务下一轮迭代。



# Reference
[计算图](https://openmlsys.github.io/chapter_computational_graph/index.html)