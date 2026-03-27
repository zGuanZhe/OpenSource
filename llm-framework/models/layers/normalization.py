import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """均方根归一化 (Root Mean Square Normalization)
    现代大模型(如LLaMA)通常使用RMSNorm替代LayerNorm，因为它省略了均值计算，
    计算效率更高，同时保持了相似的性能。
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        # 可学习的缩放权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # 计算方差前转换到fp32，防止数值溢出
        hidden_states = hidden_states.to(torch.float32)
        # 计算均方根
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 乘以缩放权重并转换回原始数据类型
        return self.weight * hidden_states.to(input_dtype)
