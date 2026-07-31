import torch as torch 
from torch import einsum
import math
from cs336_basics.model.function import softmax

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor: 
    # q: (..., query_length, d_k) 
    # k: (..., key_length, d_k)
    # v: (..., key_length, d_v)
    # mask: (..., query_length, key_length)

    d_k = query.size(-1)
    match = einsum('...ij, ...kj -> ...ik', query, key)/math.sqrt(d_k)
    if mask is not None:
        masked_match = match.masked_fill(~mask, float('-inf'))
    else:
        masked_match = match
    normalized_match = softmax(masked_match, -1)
    attention = einsum('...ik, ...kj -> ...ij', normalized_match, value)
    return attention
