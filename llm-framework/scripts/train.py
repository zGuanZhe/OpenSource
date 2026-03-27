import sys
import os
import torch

# 将项目根目录添加到系统路径中，以便能够导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import ModelConfig
from config.train_config import TrainConfig
from models.llm import LLM
from data.tokenizer import SimpleTokenizer
from data.dataset import TextDataset
from data.collator import DataCollatorForLanguageModeling
from training.trainer import Trainer

def main():
    print("正在初始化配置...")
    # 使用较小的参数用于测试运行
    model_config = ModelConfig(
        vocab_size=1000, 
        hidden_size=128, 
        num_hidden_layers=2, 
        num_attention_heads=4,
        intermediate_size=512
    )
    train_config = TrainConfig(num_train_epochs=2, logging_steps=2)

    print("正在加载分词器和数据集...")
    tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
    
    # 模拟数据
    texts = [
        "你好世界！这是一个简单的测试。", 
        "大语言模型的架构非常有趣。", 
        "从零开始训练一个模型需要很多数据。"
    ] * 10
    
    dataset = TextDataset(texts, tokenizer, max_length=16)
    collator = DataCollatorForLanguageModeling(tokenizer)

    print("正在初始化 LLaMA 风格模型...")
    model = LLM(model_config)

    # 自动选择计算设备
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"开始在 {device} 上进行训练...")
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        config=train_config,
        collator=collator,
        device=device
    )

    trainer.train()
    print("训练完成！")

if __name__ == "__main__":
    main()
