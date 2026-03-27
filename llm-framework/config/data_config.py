from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    """数据处理配置参数"""
    dataset_name: str = "wikitext" # 数据集名称
    dataset_config_name: str = "wikitext-103-v1" # 数据集版本
    train_file: Optional[str] = None         # 自定义训练集文件路径
    validation_file: Optional[str] = None    # 自定义验证集文件路径
    max_seq_length: int = 512      # 输入模型的最大序列长度
    overwrite_cache: bool = False  # 是否覆盖缓存的数据集
    preprocessing_num_workers: int = 4 # 数据预处理的线程数
    mlm_probability: float = 0.15  # 掩码语言模型(MLM)的掩码概率(若有需要)
