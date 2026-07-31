import torch as torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0] 
        exp_x = torch.exp(x - x_max)
        return exp_x / torch.sum(exp_x, dim=self.dim, keepdim=True)


