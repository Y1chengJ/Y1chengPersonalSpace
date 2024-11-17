# Concepts

## Batch Normalization and Layer Normalization

![img](https://pic1.zhimg.com/v2-b9a2091944d2a58abb5c8bc3028e098e_1440w.jpg)

Normalization 的作用很明显，把数据拉回标准正态分布，因为神经网络的Block大部分都是矩阵运算，一个向量经过矩阵运算后值会越来越大，为了网络的稳定性，我们需要及时把值拉回正态分布。

Normalization根据标准化操作的维度不同可以分为batch Normalization和Layer Normalization，不管在哪个维度上做noramlization，本质都是为了让数据在这个维度上归一化，因为在训练过程中，上一层传递下去的值千奇百怪，什么样子的分布都有。BatchNorm就是通过对batch size这个维度归一化来让分布稳定下来。LayerNorm则是通过对Hidden size这个维度归一化来让某层的分布稳定。

可以这样理解，深度网络每一层网络是相对独立的，也就是说每一层网络可以单独看成一个Classifier.不停对上一层的输出数据进行分类，每一层输出的数据分布又不一样，这就会出现Internal Covariate Shift（内部协变量偏移）. 随着网络的层数不断增大，这种误差就会不断积累，最终导致效果欠佳。显然对数据预处理只能解决第一层的问题，之后需要Normalization等方法来解决。

## Layer Normalization Example

**For a tensor with shape (2, 3, 4), how many times will the mean and variance be calculated in the layer normalization process?**

 - 2 * 3 = 6 times

Batch size = 2 (number of sequences in a batch)
Sequence length = 3 (number of tokens in each sequence)
Hidden size = 4 (dimensionality of each token’s embedding)
Layer Normalization in NLP with (2, 3, 4) Shape
In this case, LayerNorm operates independently on the hidden size for each token in the sequence. This means that for each token (each (4) vector), LayerNorm will calculate a mean and variance across the hidden size dimension.

Given the shape (2, 3, 4):

Number of Mean and Variance Calculations:

We have a total of 6 tokens (2 sequences × 3 tokens per sequence).
LayerNorm calculates the mean and variance independently for each token across its hidden size (4).
Therefore, the mean and variance are each calculated 6 times — once for each token.
Steps for Each Token:

For each token (e.g., a (4,) vector), LayerNorm:
Calculates the mean of its 4 features.
Calculates the variance of its 4 features.
Normalizes each feature based on this mean and variance.

**What is the shape of scale and shift?**

- 4

Explanation

For a tensor of shape `(batch_size, seq_length, hidden_size)`, the scale and bias parameters for LayerNorm:

- **Scale (gamma)**: A learnable parameter that has the shape `(hidden_size,)`, which is `(4,)` in this example. It is broadcasted across the batch and sequence dimensions.
- **Bias (beta)**: Another learnable parameter with the same shape as the scale, `(4,)`, which is also broadcasted across the batch and sequence dimensions.

## Batch Normalization Example

**For a tensor with shape (2, 3, 4, 4), how many times will the mean and variance be calculated in the batch normalization process?**

- 2 $\times$ 3 = 6

In **computer vision (CV)**, **Batch Normalization** (BatchNorm) typically normalizes each **feature channel** independently across the batch and spatial dimensions (height and width). Given a tensor shape of `(2, 3, 4, 4)` — a typical 4D tensor format for CV models — let’s break down the BatchNorm calculation.

Tensor Shape: `(2, 3, 4, 4)`

Here, the dimensions represent:

- **Batch size**: 2 (number of images or data samples in the batch).
- **Channels**: 3 (number of feature channels or filters).
- **Height**: 4 (height of each feature map).
- **Width**: 4 (width of each feature map).

In Batch Normalization for CV, normalization is performed across the batch, height, and width dimensions, but **independently for each channel**.

Number of Times Mean and Variance Are Calculated in Batch Normalization

1. **Mean and Variance Calculations Per Channel**:
   - BatchNorm computes the mean and variance for each feature channel independently, aggregating over the **batch, height, and width dimensions**.
   - For each channel, the mean and variance are calculated over all values in a shape of `(2, 4, 4)`, totaling `2 * 4 * 4 = 32` elements per channel.
2. **Total Number of Channels**:
   - Since there are 3 channels, the mean and variance are each calculated **3 times** — once for each channel.

**What is the shape of scale and shift?**

- 3

Given a tensor with shape `(2, 3, 4, 4)`, where:

- `2` is the **batch size**,
- `3` is the **number of channels**,
- `4` and `4` are the **height** and **width**,

the **scale** and **shift** parameters will have a shape of `(3,)`, matching the number of channels. This allows each channel to have its own scaling and shifting factor.

Explanation

1. **Scale (Gamma) and Shift (Beta) Shape**:
   - Since BatchNorm normalizes each channel independently, it requires one scaling factor and one shifting factor per channel.
   - Therefore, `gamma` (scale) and `beta` (shift) each have a shape of `(3,)` to match the `3` channels in the input tensor.
   - These parameters are broadcasted across the batch, height, and width dimensions when applied to the normalized output.

### 
