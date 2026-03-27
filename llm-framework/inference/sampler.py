import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """使用 Top-K 和/或 Top-p (Nucleus) 过滤 logits 分布
    Top-K: 仅保留概率最高的 K 个词。
    Top-p: 仅保留累积概率达到 p 的最小词集。
    """
    top_k = min(top_k, logits.size(-1))  # 安全检查
    if top_k > 0:
        # 将所有概率低于 Top-K 中最低概率的词的 logits 设为负无穷
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        # 对 logits 降序排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的词 (保留阈值之上的第一个词)
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右平移，确保即使超过阈值也至少保留一个词
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将排序后的索引散布回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits
