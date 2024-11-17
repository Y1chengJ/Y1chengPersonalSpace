import torch

# Parameters
batch_size = 4
num_channels = 3
height = 5
width = 5

# Input tensor (e.g., feature map after a convolutional layer)
x = torch.randn(batch_size, num_channels, height, width)
print("Input shape:", x.shape)

# Hard-coded Batch Normalization
def batch_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    # Compute mean and variance along the batch and spatial dimensions
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    print("Batch Norm - Mean shape:", mean.shape)
    print("Batch Norm - Variance shape:", var.shape)
    
    # Normalize
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    print("Batch Norm - Normalized shape:", x_normalized.shape)
    
    # Scale and shift
    x_out = gamma * x_normalized + beta
    print("Batch Norm - Output shape:", x_out.shape)
    return x_out

# Apply hard-coded Batch Normalization
print("\nApplying hard-coded Batch Normalization:")
x_batch_norm = batch_norm(x)


# Hard-coded Layer Normalization
def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    # Compute mean and variance along the channel and spatial dimensions for each example in the batch
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
    print("Layer Norm - Mean shape:", mean.shape)
    print("Layer Norm - Variance shape:", var.shape)
    
    # Normalize
    x_normalized = (x - mean) / torch.sqrt(var + eps)
    print("Layer Norm - Normalized shape:", x_normalized.shape)
    
    # Scale and shift
    x_out = gamma * x_normalized + beta
    print("Layer Norm - Output shape:", x_out.shape)
    return x_out

# Apply hard-coded Layer Normalization
print("\nApplying hard-coded Layer Normalization:")
x_layer_norm = layer_norm(x)
