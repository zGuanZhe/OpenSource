import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import ModelConfig
from models.llm import LLM

def main():
    model_config = ModelConfig()
    model = LLM(model_config)
    
    export_path = "model_exported.pt"
    # Example export (just state dict for simplicity, could be ONNX/TorchScript)
    torch.save(model.state_dict(), export_path)
    print(f"Model exported to {export_path}")

if __name__ == "__main__":
    main()
