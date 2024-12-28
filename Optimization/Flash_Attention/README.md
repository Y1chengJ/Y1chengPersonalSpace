# Flash Attention Reproduction

## Attention Steps
$Q, K, V \in R^{N*d}$ Store in HBM
1. Load Q, K, V from High Bandwidth Memory (HBM) to SRAM
2. Compute S = QK^T
3. Write S to HBM
4. Load S to SRAM
5. Compute P = softmax(S)
6. Write P to HBM
7. Load P and V to SRAM
8. Compute O = PV
9. Write O to HBM
10. Return O


## Compute Bound
Bigger matrix multiplication and Multi-channel Convolution

## Memory Bound
bit-wise operations: ReLU, Dropout, etc.
sum, sofmax, etc.
### Optimization
1. Fusion: Don't store intermediate results, reduce HBM access
   1. It can accelerate and increase efficiency, but during training, it needs to store intermediate results for backpropagation.


## Flash Attention 
Goal: Reduce HBM access
1. Through block computation, fuse multiple operations into one operation, reduce HBM access
2. Recompute the intermediate results during backpropagation


# Reference
[Flash Attention v1](https://github.com/tspeterkim/flash-attention-minimal/blob/main/flash.cu)


