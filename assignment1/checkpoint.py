import torch as torch
import os
import typing
from cs336_basics.common.constants import ITERATION, MODEL_STATE_DICT, OPTIMIZER_STATE_DICT

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint_data = {
        ITERATION: iteration, 
        MODEL_STATE_DICT: model.state_dict(), 
        OPTIMIZER_STATE_DICT: optimizer.state_dict()
    }

    torch.save(checkpoint_data, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint_data = torch.load(src) 
    model.load_state_dict(checkpoint_data[MODEL_STATE_DICT])
    if optimizer is not None and OPTIMIZER_STATE_DICT in checkpoint_data: 
        optimizer.load_state_dict(checkpoint_data[OPTIMIZER_STATE_DICT])

    iteration = checkpoint_data.get(ITERATION, 0)
    return iteration


