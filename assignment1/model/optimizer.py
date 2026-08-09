from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.001):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0: 
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0 < betas[0] < 1:
            raise ValueError(f"Invalid beta value of index 0: {betas[0]}")
        if not 0 < betas[1] < 1:
            raise ValueError(f"Invalid beta value of index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lambd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                t = state.get("t") + 1  
                state["t"] = t
                m = state.get("m")
                v = state.get("v")

                grad = p.grad
                with torch.no_grad(): 
                    alpha_t = alpha * math.sqrt(1.0 - math.pow(beta2, t)) / (1.0 - math.pow(beta1, t)) 
                    p.mul_(1 - alpha * lambd)
                    m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                    v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                    p.sub_(alpha_t * m / (v.sqrt() + eps))
        return loss 


def learning_rate_schedule(t: int, lr_max: float, lr_min: float, warm_up_step: int, cos_step: int):
    if lr_max < lr_min:
        raise ValueError(f"lr_max should be bigger than lr_min {lr_max}:{lr_min}")
    if cos_step < warm_up_step:
        raise ValueError(f"cos_step should be bigger than warm_up_step {cos_step}:{warm_up_step}")

    if t < warm_up_step:
        return t * lr_max / warm_up_step 
    if warm_up_step <= t <= cos_step:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - warm_up_step) / (cos_step - warm_up_step))) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float):
    params = list(params)
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return torch.Tensor(0)
    grads_norm = torch.stack([
        torch.linalg.vector_norm(
            grad,
            ord=2
        ) for grad in grads
    ])
    total_norm = torch.linalg.vector_norm(
        grads_norm, 
        ord=2
    )
    clip = torch.clamp(input=max_norm/(total_norm + 1e-6), max=1.0)

    for g in grads: 
        g.mul_(clip)

    return total_norm
