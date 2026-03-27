import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import ModelConfig
from models.llm import LLM
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset
from data.collator import DataCollatorForLanguageModeling
from training.loss import CrossEntropyLossWithLM
from evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader

def main():
    print("正在初始化配置...")
    model_config = ModelConfig(
        vocab_size=1000, 
        hidden_size=128, 
        num_hidden_layers=2, 
        num_attention_heads=4,
        intermediate_size=512
    )
    tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
    model = LLM(model_config)
    
    # 模拟验证数据
    texts = ["验证集文本一。", "验证集文本二。"] * 5
    dataset = TextDataset(texts, tokenizer, max_length=16)
    collator = DataCollatorForLanguageModeling(tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator)
    
    loss_fct = CrossEntropyLossWithLM(model_config.vocab_size)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    model.to(device)
    evaluator = Evaluator(model, dataloader, loss_fct, device=device)
    
    print("正在运行模型评估...")
    evaluator.evaluate()

if __name__ == "__main__":
    main()
