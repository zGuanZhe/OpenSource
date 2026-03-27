import torch.nn as nn

class CrossEntropyLossWithLM(nn.Module):
    """带移位操作的交叉熵损失函数 (Cross Entropy Loss for LM)
    自回归语言模型中，需要用前 n-1 个 token 预测第 n 个 token，
    因此在计算损失时需要对 logits 和 labels 进行移位 (Shift)。
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss()
        self.vocab_size = vocab_size

    def forward(self, logits, labels):
        # 将预测的 logits 丢弃最后一个时间步 (因为它没有对应的真实下一个词)
        shift_logits = logits[..., :-1, :].contiguous()
        # 将标签 labels 丢弃第一个时间步 (因为第一个词没有前置词来预测它)
        shift_labels = labels[..., 1:].contiguous()
        
        # 展平 tensor 进行损失计算
        loss = self.loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        return loss
