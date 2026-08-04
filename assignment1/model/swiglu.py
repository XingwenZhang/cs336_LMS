import torch as torch
import torch.nn as nn
from cs336_basics.model.linear import Linear


class SWIGlu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_ff = d_ff if d_ff is not None else self._up_project(d_model)
        self.w1= Linear(d_model, self.d_ff, **factory_kwargs)
        self.w2 = Linear(self.d_ff, d_model, **factory_kwargs)
        self.w3 = Linear(d_model, self.d_ff, **factory_kwargs)

    def _up_project(self, d_model, multi_of: int = 64) -> int:
        dim = int(8 * d_model / 3)
        aligned_dim = (dim + multi_of - 1 ) // multi_of * multi_of
        return aligned_dim


    def silu(self, x : torch.Tensor) -> torch.Tensor: 
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (batch, sequence, d_model)
        gate = self.w1(x)
        up = self.w3(x)

        return self.w2(self.silu(gate) * up)

        

