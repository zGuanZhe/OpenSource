import torch
import torch.nn as nn
from typing import Optional, Tuple
from .layers.attention import GroupedQueryAttention, precompute_freqs_cis
from .layers.feedforward import SwiGLUFeedForward
from .layers.normalization import RMSNorm
from .heads.lm_head import LMHead

class TransformerBlock(nn.Module):
    """单个 Transformer 块 (LLaMA 风格)"""
    def __init__(self, config):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLUFeedForward(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states, freqs_cis, attention_mask=None, use_cache=False, past_key_value=None):
        # 注意力机制（使用前置归一化 Pre-Norm）
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_outputs, present_key_value = self.attention(
            hidden_states, freqs_cis, attention_mask, use_cache, past_key_value
        )
        hidden_states = residual + attn_outputs
        
        # 前馈神经网络（使用前置归一化 Pre-Norm）
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        hidden_states = residual + ff_outputs
        
        return hidden_states, present_key_value

class LLM(nn.Module):
    """大语言模型主类 (基于现代 LLaMA 架构)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        
        # 词嵌入层 (现代模型移除绝对位置编码)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer 块列表
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        # 最终归一化层
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 语言建模头
        self.lm_head = LMHead(config)
        
        # 权重共享 (Weight Tying) - 现代模型可选，这里保持独立以匹配 LLaMA
        # self.lm_head.decoder.weight = self.wte.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # 获取词嵌入
        inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(inputs_embeds)
        
        # 预计算 RoPE 频率
        # 如果有 past_key_values (KV Cache)，需要加上缓存的长度
        past_seq_len = past_key_values[0][0].shape[1] if past_key_values is not None and past_key_values[0] is not None else 0
        total_seq_len = past_seq_len + seq_length
        freqs_cis = precompute_freqs_cis(
            self.config.hidden_size // self.config.num_attention_heads, 
            total_seq_len, 
            theta=self.config.rope_theta
        ).to(device)
        
        # 提取当前步的 freqs_cis
        freqs_cis = freqs_cis[past_seq_len:total_seq_len]
        
        # 生成因果注意力掩码 (Causal Mask)
        # 注意: 这里的掩码逻辑对于训练(past_seq_len=0)和推理(seq_len=1)是正确的，
        # 但对于带padding的批处理推理(seq_len>1, past_seq_len>0)可能不完全正确，为了修复类型错误，我们暂时保留此逻辑。
        final_attention_mask: Optional[torch.Tensor]
        causal_mask = torch.tril(torch.ones((seq_length, total_seq_len), device=device))
        
        if attention_mask is not None:
            # 将填充掩码转为加性掩码
            # 确保 attention_mask 扩展到正确的维度
            expanded_padding_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_length, total_seq_len)
            additive_padding_mask = (1.0 - expanded_padding_mask) * -10000.0
            
            # 加上因果掩码 (下三角矩阵)
            additive_causal_mask = (1.0 - causal_mask.unsqueeze(0).unsqueeze(0)) * -10000.0
            final_attention_mask = additive_padding_mask + additive_causal_mask
        else:
            final_attention_mask = (1.0 - causal_mask.unsqueeze(0).unsqueeze(0)) * -10000.0
            
        presents = ()
        
        # 逐层前向传播
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=final_attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            if use_cache:
                presents = presents + (present,)
                
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        # 输出预测的 Logits
        logits = self.lm_head(hidden_states)
        
        return logits, presents
