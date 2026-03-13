from config import ModelConfig

class RMSNormParams:

    def __init__(self, config: ModelConfig):
        self.config = config

    def total_params(self):
        return self.config.hidden_size

class AttentionParams:

    def __init__(self, config: ModelConfig):
        self.config = config

    def total_params(self):
        q_proj = self.config.hidden_size * self.config.head_size * self.config.num_attention_heads
        kv_proj = 2 * self.config.hidden_size * self.config.head_size * self.config.num_key_value_heads
        o_proj = self.config.head_size * self.config.num_attention_heads * self.config.hidden_size

        if self.config.attn_gate_type == "none":
            return q_proj + kv_proj + o_proj
        elif self.config.attn_gate_type == 'token':
            out_dim = 1
        elif self.config.attn_gate_type == 'head':
            out_dim = self.config.num_attention_heads
        elif self.config.attn_gate_type == 'channel':
            out_dim = self.config.num_attention_heads * self.config.head_size

        gate_proj = self.config.hidden_size * out_dim

        return q_proj + kv_proj + o_proj + gate_proj

class FeedForwardParams:

    def __init__(self, config: ModelConfig):
        self.config = config

    def total_params(self):
        up_proj = self.config.hidden_size * self.config.intermediate_size
        gate_proj = self.config.hidden_size * self.config.intermediate_size
        down_proj = self.config.intermediate_size * self.config.hidden_size

        return up_proj + gate_proj + down_proj

class TransformerBlockProfile:

    def __init__(self, config: ModelConfig):
        self.config = config

        self.pre_attn_rmsnorm = RMSNormParams(self.config)
        self.pre_ffn_rmsnorm = RMSNormParams(self.config)
        self.self_attn = AttentionParams(self.config)
        self.ffn = FeedForwardParams(self.config)

    def detail(self):

        attn_params = self.self_attn.total_params() 
        pre_attn_rmsnorm_params = self.pre_attn_rmsnorm.total_params()
        ffn_params = self.ffn.total_params()
        pre_ffn_rmsnorm_params = self.pre_ffn_rmsnorm.total_params()

        total_params = attn_params + pre_attn_rmsnorm_params + ffn_params + pre_ffn_rmsnorm_params

        return {
            "self_attn": attn_params,
            "pre_attn_rmsnorm": pre_attn_rmsnorm_params,
            "ffn": ffn_params,
            "pre_ffn_rmsnorm": pre_ffn_rmsnorm_params,
            "total": total_params
        }



    def total_params(self):
        return self.detail()["total"]


class EmbeddingParams:

    def __init__(self, config: ModelConfig):
        self.config = config

    def total_params(self):
        return self.config.vocab_size * self.config.hidden_size


class TransformerModelProfile:

    def __init__(self, config: ModelConfig):
        self.config = config

        self.embed_tokens = EmbeddingParams(config)
        self.block = TransformerBlockProfile(config)
        self.final_norm = RMSNormParams(config)

    def detail(self):

        embed_tokens_params = self.embed_tokens.total_params()
        block_params = self.block.total_params()
        all_block_params = block_params * self.config.num_hidden_layers
        final_norm_params = self.final_norm.total_params()

        total_params = embed_tokens_params + all_block_params + final_norm_params

        return {
            "embed_tokens": embed_tokens_params,
            "block": block_params,
            "all_block": all_block_params,
            "final_norm": final_norm_params,
            "total": total_params
        }

    def total_params(self):
        return self.detail()["total"]

