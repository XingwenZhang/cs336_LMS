import torch as torch
import torch.nn as nn
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.swiglu import SWIGlu

def transformer_block(d_model:int, num_heads:int, d_ff:int, max_seq_len: int, theta: float, x:torch.Tensor)-> torch.Tensor:
    rms_norm = RMSNorm(d_model)
    mha = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
    output_layer_1 = x + mha(rms_norm(x))

    swiglu = SWIGlu(d_model, d_ff)

    output_layer_2 = output_layer_1 + swiglu(rms_norm(output_layer_1))
    
    return output_layer_2

