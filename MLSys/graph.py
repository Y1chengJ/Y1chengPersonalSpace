import torch
import torch.nn as nn

# 定义具有动态控制流的模型
class DynamicModel(nn.Module):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        # 动态控制流示例
        if x.sum() > 0:
            x = self.fc1(x)
            for _ in range(3):  # 循环操作
                x = torch.relu(x)
        else:
            x = self.fc2(x)
        return x

# 创建模型实例
model = DynamicModel()

# 输入张量
input_tensor = torch.randn(5, 10)

# 1. 使用 torch.jit.script()（基于源码转换）
scripted_model = torch.jit.script(model)
scripted_output = scripted_model(input_tensor)
print("TorchScript (script) Output:", scripted_output)

# 2. 使用 torch.jit.trace()（基于追踪转换）
traced_model = torch.jit.trace(model, input_tensor)
traced_output = traced_model(input_tensor)
print("TorchScript (trace) Output:", traced_output)
