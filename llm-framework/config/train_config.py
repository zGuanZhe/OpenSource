from dataclasses import dataclass

@dataclass
class TrainConfig:
    """训练超参数配置"""
    output_dir: str = "./outputs"    # 模型输出路径
    learning_rate: float = 5e-5      # 峰值学习率
    weight_decay: float = 0.01       # 权重衰减 (L2正则化)
    adam_beta1: float = 0.9          # AdamW 优化器的 beta1
    adam_beta2: float = 0.999        # AdamW 优化器的 beta2
    adam_epsilon: float = 1e-8       # AdamW 优化器的 epsilon
    max_grad_norm: float = 1.0       # 梯度裁剪的最大范数
    num_train_epochs: int = 3        # 训练的总轮数 (Epochs)
    per_device_train_batch_size: int = 8 # 每个设备的训练批次大小
    per_device_eval_batch_size: int = 8  # 每个设备的评估批次大小
    gradient_accumulation_steps: int = 1 # 梯度累加步数
    warmup_steps: int = 0            # 学习率预热步数
    logging_steps: int = 10          # 日志打印的步数间隔
    save_steps: int = 500            # 模型保存的步数间隔
    eval_steps: int = 500            # 模型评估的步数间隔
    seed: int = 42                   # 随机种子，确保结果可复现
    fp16: bool = False               # 是否使用混合精度训练
