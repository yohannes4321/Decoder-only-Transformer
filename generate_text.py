import torch
from decoder import  encode, decode,vocab_size,block_size
import torch
import torch
from decoder import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedFoward



def generate_from_prompt(prompt: str, max_new_tokens=200, temperature=0.7, top_k=50, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initialize the model (must match your training config)
    # m = DecoderModel(vocab_size=vocab_size, block_size=128, n_embed=512)
    
    

# Tell torch where GPTLanguageModel is defined
    with torch.serialization.safe_globals([GPTLanguageModel]):
        m = torch.load("decoder.pth", map_location=device, weights_only=False)

# Tell torch where GPTLanguageModel is defined


    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    input_tensor = prompt_ids

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > block_size:
            input_tensor = input_tensor[:, -block_size:]
        with torch.no_grad():
            logits,_ = m(input_tensor)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits[0, topk_idx[0]] = topk_vals[0]
                logits = filtered_logits
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
    

    return decode(input_tensor[0].tolist())
if __name__ =="__main__":
  prompt = """Continue the scene below in the same Shakespearean style and voice...
  Scene start:
  "First Citizen:
  Before we proceed any further, hear me speak.
  All:
  Speak, speak.
  First Citizen:
  You are all resolved rather to die than to famish?
  All:
  Resolved. resolved.

  """

  text = generate_from_prompt(prompt, max_new_tokens=200, temperature=0.7, top_k=50, device='cpu')
  print(text)
