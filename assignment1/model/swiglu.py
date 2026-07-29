import torch as torch
import torch.nn as nn


class SWIGlu(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_ff = d_ff if d_ff is not None else self._up_project(d_model)
        self.weights1 = nn.Parameter(torch.ones(self.d_ff, d_model, **factory_kwargs))
        self.weights2 = nn.Parameter(torch.ones(d_model, self.d_ff, **factory_kwargs))
        self.weights3 = nn.Parameter(torch.ones(self.d_ff, d_model, **factory_kwargs))

    def _up_project(self, d_model, multi_of: int = 64) -> int:
        dim = int(8 * d_model / 3)
        aligned_dim = (dim + multi_of - 1 ) // multi_of * multi_of
        return aligned_dim


    def silu(self, x : torch.Tensor) -> torch.Tensor: 
        return x * torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (d_model, )
        part1 = torch.einsum('ij, ...j -> ...i', self.weights1, x)
        part3 = torch.einsum('ij, ...j -> ...i', self.weights3, x)
        part2 = torch.einsum('ji, ...i -> ...j', self.weights2, self.silu(part1) * part3)
        return part2

        

