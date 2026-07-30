import torch as torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f'd_k should be even, current d_k is {d_k}')
        
        factory_kwargs = {'device': device, 'dtype': torch.float32}
        exponent = torch.arange(0, d_k, 2, **factory_kwargs) / d_k # (d_k/2, )
        inv_freq = 1 / torch.pow(theta, exponent)

        positions = torch.arange(0, max_seq_len, **factory_kwargs) # (max_seq_len, )

        angles = torch.outer(positions, inv_freq) # (max_seq_len, d_k/2)

        self.register_buffer('cos_cached', angles.cos(), persistent=False)
        self.register_buffer('sin_cached', angles.sin(), persistent=False) 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_val = self.cos_cached[token_positions]
        sin_val = self.sin_cached[token_positions] #(batch, seq, d_k/2)

        x_even = x[... , 0::2]   # (batch, seq, d_k/2)
        x_odd = x[... , 1::2]

        rot_even = x_even * cos_val - x_odd * sin_val 
        rot_odd = x_even * sin_val + x_odd * cos_val

        out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return out

        

