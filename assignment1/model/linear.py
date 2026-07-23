import torch as torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super.__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = nn.Parameter(torch.randn(out_features, in_features, **factory_kwargs))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.weights.device, dtype=self.weights.dtype)
        return x @ self.weights.t() + self.bias

