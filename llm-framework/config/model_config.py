from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """大语言模型超参数配置 (基于现代 LLaMA 架构)"""
    vocab_size: int = 32000     # 词表大小
    hidden_size: int = 768      # 隐藏层维度 (词向量维度)
    num_hidden_layers: int = 12 # Transformer 层数
    num_attention_heads: int = 12 # 注意力头数
    num_key_value_heads: Optional[int] = None # 键值头数 (用于GQA)，若为None则退化为MHA(多头注意力)
    intermediate_size: int = 3072 # 前馈网络中间层维度 (用于SwiGLU)
    hidden_dropout_prob: float = 0.1 # 隐藏层 Dropout 概率
    attention_probs_dropout_prob: float = 0.1 # 注意力矩阵 Dropout 概率
    max_position_embeddings: int = 2048 # 最大序列长度
    rms_norm_eps: float = 1e-6  # RMSNorm 的微小常数，防止除以零
    initializer_range: float = 0.02 # 权重初始化标准差
    use_cache: bool = True      # 推理时是否使用 KV Cache 加速
    rope_theta: float = 10000.0 # RoPE (旋转位置编码) 的 base theta 参数
