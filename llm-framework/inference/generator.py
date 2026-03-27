import torch
from .sampler import top_k_top_p_filtering
import torch.nn.functional as F

class TextGenerator:
    """文本生成器 (Text Generator)
    实现自回归解码过程，支持 Top-K 和 Top-p (Nucleus) 采样。
    """
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_length=50, temperature=1.0, top_k=50, top_p=0.9):
        # 编码提示词
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        
        # 简单的自回归生成循环
        for _ in range(max_length):
            # 获取模型输出
            logits, _ = self.model(input_ids)
            # 取最后一个 token 的 logits，并应用温度缩放
            next_token_logits = logits[:, -1, :] / temperature
            # 应用 Top-K / Top-p 过滤
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # 计算概率并采样下一个 token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 拼接到输入序列中
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 如果生成了结束符 (EOS)，则停止生成
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # 解码并返回生成的文本
        return self.tokenizer.decode(input_ids[0].cpu().tolist())
