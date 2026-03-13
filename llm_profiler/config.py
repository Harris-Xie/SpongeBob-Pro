
class ModelConfig:

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        head_size,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        attn_gate_type = "none"
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.attn_gate_type = attn_gate_type

        self._validate()

    def _validate(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须被 num_key_value_heads 整除")
        
        if self.attn_gate_type not in ["none", "token", "head", "channel"]:
            raise ValueError("attn_gate_type 必须要是 none/token/head/channel 之一")
