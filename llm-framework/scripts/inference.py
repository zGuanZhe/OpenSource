import sys
import os
import torch

# 将项目根目录添加到系统路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import ModelConfig
from models.llm import LLM
from data.tokenizer import SimpleTokenizer
from inference.generator import TextGenerator

def main():
    # 保持与训练脚本一致的小模型参数用于测试
    model_config = ModelConfig(
        vocab_size=1000, 
        hidden_size=128, 
        num_hidden_layers=2, 
        num_attention_heads=4,
        intermediate_size=512
    )
    tokenizer = SimpleTokenizer(vocab_size=model_config.vocab_size)
    
    print("正在加载模型...")
    model = LLM(model_config)
    
    # 真实场景中应在此处加载预训练权重:
    # model.load_state_dict(torch.load("path_to_checkpoint.pt"))
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    generator = TextGenerator(model, tokenizer, device=device)

    prompt = "你好"
    print(f"输入提示词 (Prompt): {prompt}")
    
    print("正在生成文本...")
    output = generator.generate(prompt, max_length=20, temperature=0.8)
    print(f"生成结果: {output}")

if __name__ == "__main__":
    main()
