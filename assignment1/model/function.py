import torch as torch 

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: 
    x_max = torch.max(x, dim=dim, keepdim=True)[0] 
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)