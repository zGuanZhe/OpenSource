import torch

class DataCollatorForLanguageModeling:
    """语言模型的数据整理器 (Data Collator)
    负责将 Dataset 输出的多个样本打包成一个 Batch，并生成对应的 Labels。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        batch = {}
        # 将样本堆叠为 Batch
        batch["input_ids"] = torch.stack([e["input_ids"] for e in examples])
        batch["attention_mask"] = torch.stack([e["attention_mask"] for e in examples])
        
        # 对于因果语言模型 (Causal LM)，标签(labels)与输入(input_ids)相同
        # 移位(Shift)操作通常在损失函数或模型内部处理
        batch["labels"] = batch["input_ids"].clone()
        
        return batch
