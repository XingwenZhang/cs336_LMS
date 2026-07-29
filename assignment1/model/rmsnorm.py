import torch as torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        mean_square = torch.pow(x, 2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        result = x / rms * self.weights

        return result.to(in_dtype)

