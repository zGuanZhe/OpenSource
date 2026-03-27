import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    """基于 SwiGLU 的前馈神经网络 (LLaMA 风格)
    传统的 Transformer 使用 ReLU 或 GELU，而现代模型多采用 SwiGLU。
    SwiGLU 包含一个门控机制 (Gate)，表现出更好的模型收敛性和性能。
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # 门控投影 (Gate Projection) 和 上采样投影 (Up Projection)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 下采样投影 (Down Projection)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        # SwiGLU 激活公式: SiLU(gate(x)) * up(x)
        gate = F.silu(self.gate_proj(x)) # SiLU 也就是 Swish(beta=1)
        up = self.up_proj(x)
        x = gate * up
        # 下采样回原来的维度
        x = self.down_proj(x)
        x = self.dropout(x)
        return x
