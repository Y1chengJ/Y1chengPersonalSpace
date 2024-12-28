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

