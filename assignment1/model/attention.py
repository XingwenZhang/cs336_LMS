import torch as torch 
from torch import einsum
import math
from cs336_basics.model.function import softmax
from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RoPE

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: 
    # q: (..., query_length, d_k) 
    # k: (..., key_length, d_k)
    # v: (..., key_length, d_v)
    # mask: (..., query_length, key_length)

    d_k = query.size(-1)
    attention_score = einsum('...ij, ...kj -> ...ik', query, key)/math.sqrt(d_k)
    if mask is not None:
        masked_score = attention_score.masked_fill(~mask, float('-inf'))
    else:
        masked_score = attention_score
    attention_weights = softmax(masked_score, -1)
    attention = einsum('...ik, ...kj -> ...ij', attention_weights, value)
    return attention

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__() 
        if d_model % num_heads != 0:
            raise ValueError(f'd_model/num_heads should be int {d_model}, {num_heads}')
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = int(self.d_model / self.num_heads)
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs) 
        self.o_proj = Linear(d_model, d_model, **factory_kwargs)
        if self.theta is not None: 
            self.rope = RoPE(self.theta, self.d_k, self.max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = torch.unflatten(query, -1, (self.num_heads, self.d_k)).transpose(-3, -2)
        key = torch.unflatten(key, -1, (self.num_heads, self.d_k)).transpose(-3, -2)
        value = torch.unflatten(value, -1, (self.num_heads, self.d_k)).transpose(-3, -2)

        q_rot = query 
        k_rot = key

        if mask is None: 
            query_len = query.size(-2)
            key_len = query.size(-2)
            mask = torch.tril(torch.ones(query_len, key_len, dtype=torch.bool))

        if token_positions is not None:
            q_rot = self.rope(query, token_positions)
            k_rot = self.rope(key, token_positions)

        assert query.shape == q_rot.shape 
        assert key.shape == k_rot.shape 

        attention = scaled_dot_product_attention(q_rot, k_rot, value, mask)

        attention = attention.transpose(-3, -2).flatten(-2, -1)  #(batch, seq, d_model)

        output = self.o_proj(attention)

        assert output.shape == x.shape 

        return output
        

