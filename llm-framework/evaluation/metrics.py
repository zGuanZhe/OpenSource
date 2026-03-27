import torch
import math

def calculate_perplexity(loss):
    """计算困惑度 (Perplexity, PPL)
    困惑度是评估语言模型好坏的常用指标，值越小说明模型预测越准确。
    """
    return math.exp(loss)

def calculate_accuracy(predictions, labels):
    """计算简单的 Token 级准确率"""
    correct = (predictions == labels).sum().item()
    total = labels.numel()
    return correct / total if total > 0 else 0.0
