import torch
from torch.utils.data import DataLoader
from .loss import CrossEntropyLossWithLM
from .optimizer import create_optimizer
from .scheduler import get_cosine_schedule_with_warmup

class Trainer:
    """训练器主类 (Trainer)
    封装了模型训练的核心循环、优化器初始化、学习率调度和数据加载。
    """
    def __init__(self, model, train_dataset, config, collator=None, device="cpu"):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config
        self.device = device
        self.collator = collator

        # 构建数据加载器
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collator
        )

        # 初始化优化器和调度器
        self.optimizer = create_optimizer(self.model, self.config)
        
        num_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps
        )
        # 初始化损失函数
        self.loss_fct = CrossEntropyLossWithLM(self.model.config.vocab_size)

    def train(self):
        """开始训练循环"""
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(self.config.num_train_epochs):
            total_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                # 将数据移动到对应设备 (CPU/GPU)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 前向传播
                logits, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 计算损失
                loss = self.loss_fct(logits, labels)
                
                # 反向传播
                loss.backward()
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 更新权重
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                # 打印日志
                if step % self.config.logging_steps == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
                    
            avg_loss = total_loss / len(self.train_dataloader)
            print(f"--- Epoch {epoch} 完成 | 平均损失: {avg_loss:.4f} ---")
