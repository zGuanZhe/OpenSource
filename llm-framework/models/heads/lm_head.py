import torch.nn as nn

class LMHead(nn.Module):
    """语言建模头 (Language Modeling Head)
    用于将Transformer的隐藏状态映射回词表维度，输出每个词的概率分布(Logits)。
    """
    def __init__(self, config):
        super().__init__()
        # 线性映射，不使用偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, hidden_states):
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, vocab_size]
        return self.decoder(hidden_states)
