import torch as torch
import torch.nn as nn
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.swiglu import SWIGlu


class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs) 
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, **factory_kwargs)
        self.ffn = SWIGlu(d_model, d_ff, **factory_kwargs)
  

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor=None) -> torch.Tensor:
        if token_positions is None: 
            seq_len = x.size(-2)
            token_positions = torch.arange(0, seq_len, dtype=torch.long)
        output_layer_1 = x + self.attn(self.ln1(x), token_positions)
        return output_layer_1 + self.ffn(self.ln2(output_layer_1))

