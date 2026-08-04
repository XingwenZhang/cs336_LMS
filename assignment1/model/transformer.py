import torch as torch
import torch.nn as nn
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.swiglu import SWIGlu
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear


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
            token_positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        output_layer_1 = x + self.attn(self.ln1(x), token_positions)
        return output_layer_1 + self.ffn(self.ln2(output_layer_1))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model:int, num_layers: int, num_heads:int, d_ff:int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs) 
                                    for _ in range(num_layers))
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    # input: token_ids (batch_size, seq_len) 
    # intermediate: (batch, seq_len, d_model)
    # output: (batch, seq_len, vocab_size)
    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor: 
        seq_len = token_ids.size(-1)
        assert seq_len <= self.context_length 
        if token_positions is not None: 
            assert seq_len == token_positions.size(-1)

        if token_positions is None: 
            token_positions = torch.arange(0, seq_len, device=token_ids.device, dtype=torch.long)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions) 
        x = self.ln_final(x) 
        x = self.lm_head(x) 
        return x
