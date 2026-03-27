import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """简单的文本数据集类 (Text Dataset)
    用于将纯文本列表转换为模型可接受的 Tensor 格式。
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for text in texts:
            # 使用分词器对文本进行编码
            encoded = self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True)
            self.examples.append(encoded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long)
        }
