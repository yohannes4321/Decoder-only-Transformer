import torch
from decoder import encode, decode

def generate_from_prompt(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, top_k: int = 50, device: str = None):
    """
    Generate text given a prompt string using Top-k sampling.

    Args:
        prompt (str): Input text prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature (higher = more random).
        top_k (int): Keep only the top k tokens for sampling.
        device (str): 'cuda' or 'cpu'.

    Returns:
        str: Generated text including the prompt.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    m = torch.load("decoder.pth")
    m.to(device)
    m.eval()

    # Encode prompt
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    input_tensor = prompt_ids

    # Generation loop
    for _ in range(max_new_tokens):
        if input_tensor.size(1) > m.config.block_size:
            input_tensor = input_tensor[:, -m.config.block_size:]

        with torch.no_grad():  # disable gradients
            logits = m(input_tensor)  # [1, seq_len, vocab_size]
            logits = logits[:, -1, :] / temperature

            # --- Top-k filtering ---
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, float('-inf'))  # mask everything
                filtered_logits[0, topk_idx[0]] = topk_vals[0]            # keep only top-k
                logits = filtered_logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

        # Append token
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

        # Stop at EOS
        if next_token.item() == m.config.eos_token_id:
            break

    output_ids = input_tensor[0].tolist()
    return decode(output_ids)
