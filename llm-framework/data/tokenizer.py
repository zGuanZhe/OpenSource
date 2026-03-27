class SimpleTokenizer:
    """一个用于演示的最简分词器 (Tokenizer) 
    实际项目中应替换为 HuggingFace 的 Tokenizer (如 BPE, SentencePiece 等)。
    """
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0  # 填充 Token
        self.eos_token_id = 1  # 结束 Token
        self.unk_token_id = 2  # 未知 Token

    def encode(self, text, max_length=None):
        # 演示用编码: 简单地将字符 ASCII 映射到整数
        tokens = [ord(c) % self.vocab_size for c in text]
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, tokens):
        # 演示用解码
        return "".join([chr(t % 256) if t < 256 else "?" for t in tokens])

    def __call__(self, text, max_length=None, padding=False, truncation=False):
        tokens = self.encode(text, max_length)
        if padding and max_length:
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        return {
            "input_ids": tokens, 
            "attention_mask": [1 if t != self.pad_token_id else 0 for t in tokens]
        }
