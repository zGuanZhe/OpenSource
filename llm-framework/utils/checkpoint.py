import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    """保存模型检查点 (Checkpoint)
    包含模型权重、优化器状态和当前的 Epoch。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }, path)
    print(f"检查点已保存至: {path}")

def load_checkpoint(model, path, optimizer=None):
    """加载模型检查点 (Checkpoint)"""
    if not os.path.exists(path):
        print(f"未找到检查点: {path}")
        return model, optimizer, 0
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"检查点加载成功: {path}")
    return model, optimizer, checkpoint.get('epoch', 0)
