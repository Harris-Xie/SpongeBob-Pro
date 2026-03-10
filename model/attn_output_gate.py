from torch import nn
import torch
from model.config import SpongeBobConfig

class AttnOutputGate(nn.Module):
    def __init__(self, config: SpongeBobConfig):
        super().__init__()
        self.config = config
        self.gate_type = getattr(config, 'attn_gate_type', 'none')
        self.init_bias = getattr(config, 'attn_gate_init_bias', 4.0)

        if self.gate_type == 'token':
            out_dim = 1
        elif self.gate_type == 'head':
            out_dim = self.config.num_attention_heads
        elif self.gate_type == 'channel':
            out_dim = self.config.head_size * self.config.num_attention_heads

        self.proj = nn.Linear(self.config.hidden_size, out_dim, bias=True)

        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, self.init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input:
            x: (bsz, seq_len, hidden_size)
        output:
            gate:
            - token  -> (B, 1, S, 1)
            - head   -> (B, nh, S, 1)
            - channel -> (B, nh, S, hd)
        """

        bsz, seq_len, _ = x.shape
        gate = torch.sigmoid(self.proj(x))

        if self.gate_type == 'token':
            gate = gate.squeeze(-1)[:, None, :, None] # (B, S, 1) -> (B, 1, S, 1)
        elif self.gate_type == 'head':
            gate = gate.permute(0, 2, 1).unsqueeze(-1) # (B, S, nh) -> (B, nh, S, 1)
        elif self.gate_type == 'channel':
            gate = gate.view(bsz, seq_len, self.config.num_attention_heads, self.config.head_size).permute(0, 2, 1, 3) # (B, S, nh*hd) - > (B, nh, S, hd)


        return gate