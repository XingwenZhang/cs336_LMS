import torch as torch 
from cs336_basics.model.function import softmax

@torch.no_grad()
def generate(
    model, 
    input_ids: torch.Tensor, 
    max_new_tokens: int, 
    temperature: float = 1.0, 
    top_p: float = 1.0, 
    eos_token_id: int = None
):
    
    model.eval()
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        # [batch_size, vocab_size]
        next_token_logits = outputs[:, -1, :] # get the last token 
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            probs = softmax(next_token_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                for batch_idx in range(probs.shape[0]):
                    indices_to_remove = sorted_indices[batch_idx, sorted_indices_to_remove[batch_idx]]
                    probs[batch_idx, indices_to_remove] = 0.0

                # reset to 1
                probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return input_ids