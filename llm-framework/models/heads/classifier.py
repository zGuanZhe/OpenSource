import torch
import torch.nn as nn

class SequenceClassifierHead(nn.Module):
    """序列分类头 (Sequence Classification Head)
    可用于微调模型进行分类任务（如情感分析）。
    """
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        
    def forward(self, hidden_states):
        # 通常取序列的最后一个Token或池化输出作为分类特征
        x = hidden_states[:, -1, :] 
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x) # 激活函数
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
