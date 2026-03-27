import unittest
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.model_config import ModelConfig
from models.llm import LLM

class TestLLM(unittest.TestCase):
    def test_model_forward(self):
        config = ModelConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
        model = LLM(config)
        
        input_ids = torch.randint(0, 100, (2, 10))
        logits, _ = model(input_ids)
        
        self.assertEqual(logits.shape, (2, 10, 100))

if __name__ == "__main__":
    unittest.main()
