import torch as torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        nn.init.trunc_normal_(
            self.weights,
            mean = 0.0,
            std = 1, 
            a = -3.0,
            b = 3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

