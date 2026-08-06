import torch as torch 

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor: 
    x_max = torch.max(x, dim=dim, keepdim=True)[0] 
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

# input: logits (batch_size, seq_len, vocab_size)
#        target (batch_size, seq_len)
def cross_entropy(logits: torch.Tensor, target: torch.Tensor):
    vocab_size = logits.size(-1)
    logits = logits.reshape(-1, vocab_size)
    target = target.reshape(-1) 

    max_logits = logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits - max_logits)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    log_probs = logits - max_logits - log_sum_exp 

    n = logits.size(0) 
    selected_log_probs = log_probs[range(n), target]

    return -selected_log_probs.mean()
