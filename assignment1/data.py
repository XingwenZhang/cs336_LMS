import numpy.typing as npt
import numpy as np 
import torch as torch 

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor] :
    if len(dataset) < context_length + 1:
        raise ValueError("dataset length is not matched with context_length") 

    max_idx = len(dataset) - context_length

    start_idx = np.random.randint(0, max_idx, size=batch_size) 

    inputs_np = np.stack([dataset[i : i + context_length] for i in start_idx])
    outputs_np = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_idx])

    inputs = torch.tensor(inputs_np, dtype=torch.long, device=device)
    outputs = torch.tensor(outputs_np, dtype=torch.long, device=device)

    return (inputs, outputs)