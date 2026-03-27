import math
import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算 RoPE 的旋转频率复数矩阵"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 转换为复数表示
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """应用旋转位置编码 (RoPE)"""
    # 将实数张量转换为复数张量
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 扩展 freqs_cis 的维度以匹配输入维度
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    
    # 复数乘法并转回实数张量
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GroupedQueryAttention(nn.Module):
    """分组查询注意力机制 (Grouped-Query Attention, 支持 MHA, GQA, MQA)
    MHA (多头注意力): num_kv_heads == num_heads
    GQA (分组查询注意力): 1 < num_kv_heads < num_heads
    MQA (多查询注意力): num_kv_heads == 1
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        # 如果未设置 num_key_value_heads，则默认为多头注意力(MHA)
        self.num_kv_heads = config.num_key_value_heads if config.num_key_value_heads is not None else config.num_attention_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        
        # 线性投影层，现代模型通常不使用偏置(bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None, use_cache=False, past_key_value=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算查询(Q)、键(K)、值(V)
        xq = self.q_proj(hidden_states)
        xk = self.k_proj(hidden_states)
        xv = self.v_proj(hidden_states)
        
        # 重塑维度用于多头计算
        xq = xq.view(batch_size, seq_length, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        
        # 应用旋转位置编码 (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # KV Cache 逻辑 (用于推理加速)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
            
        present_key_value = (xk, xv) if use_cache else None
        
        # GQA / MQA 逻辑: 扩展 K, V 头数以匹配 Q 的头数
        if self.num_key_value_groups > 1:
            xk = xk.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(batch_size, -1, self.num_heads, self.head_dim)
            xv = xv.unsqueeze(3).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(batch_size, -1, self.num_heads, self.head_dim)
            
        # 调整维度用于矩阵乘法: [batch, num_heads, seq_len, head_dim]
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(xq, xk.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # 应用掩码 (例如因果掩码 causal mask)
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # 计算注意力权重并应用 Dropout
        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        # 注意力加权求和
        output = torch.matmul(probs, xv)
        # 恢复维度
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)
        
        # 输出线性投影
        return self.out_proj(output), present_key_value
