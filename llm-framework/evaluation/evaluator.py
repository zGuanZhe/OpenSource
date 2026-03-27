import torch
from .metrics import calculate_perplexity

class Evaluator:
    """模型评估器 (Evaluator)
    用于在验证集上评估模型的 Loss 和 Perplexity 等指标。
    """
    def __init__(self, model, eval_dataloader, loss_fct, device="cpu"):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.loss_fct = loss_fct
        self.device = device

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits, _ = self.model(input_ids, attention_mask=attention_mask)
            loss = self.loss_fct(logits, labels)
            total_loss += loss.item()
            
        avg_loss = total_loss / len(self.eval_dataloader)
        perplexity = calculate_perplexity(avg_loss)
        
        print(f"评估损失 (Eval Loss): {avg_loss:.4f} | 困惑度 (Perplexity): {perplexity:.4f}")
        return {"loss": avg_loss, "perplexity": perplexity}
