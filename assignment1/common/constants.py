from typing import Final

# Pattern used to grep the text
PAT: Final[str] = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

ENCODE_FORMAT: Final[str] = 'utf-8'

MODEL_STATE_DICT: Final[str] = 'model_state_dict'
OPTIMIZER_STATE_DICT: Final[str] = 'optimizer_state_dict'
ITERATION: Final[str] = 'iteration'
