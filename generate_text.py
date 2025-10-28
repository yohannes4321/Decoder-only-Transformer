import torch
import tiktoken
# --- Assume you already have your model `m` and decode function ---
# from decoder import m, decode

def generate_from_prompt(prompt: str, max_new_tokens: int = 200, temperature: float = 0.7, device: str = None):
    """
    Generate text given a prompt string.

    Args:
        prompt (str): The text prompt to start generation.
        max_new_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature (controls randomness).
        device (str): 'cuda' or 'cpu'. If None, auto-detects GPU.

    Returns:
        str: The generated text including the prompt.
    """
    enc = tiktoken.get_encoding("o200k_base")
    # 1️⃣ Determine the device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m=torch.load("decoder.pth")

    # 2️⃣ Put model on the device
    m.to(device)
    m.eval()  # Set the model to evaluation mode (disable dropout, etc.)

    # 3️⃣ Encode the prompt into token IDs
    # Here we assume your model has a tokenizer or encode function
    prompt_ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
    input_tensor = prompt_ids  # Shape: [1, prompt_length]

    # 4️⃣ Generation loop
    for _ in range(max_new_tokens):
        # 4a️⃣ Keep only the last `block_size` tokens if input is too long
        if input_tensor.size(1) > m.config.block_size:
            input_tensor = input_tensor[:, -m.config.block_size:]

        # 4b️⃣ Forward pass: get logits
        # Shape of logits: [batch_size, sequence_length, vocab_size]
        logits = m(input_tensor)
        logits = logits[:, -1, :] / temperature  # Take last token's logits and scale by temperature

        # 4c️⃣ Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # 4d️⃣ Sample the next token
        next_token = torch.multinomial(probs, num_samples=1)

        # 4e️⃣ Append sampled token to the input tensor
        input_tensor = torch.cat((input_tensor, next_token), dim=1)

        # 4f️⃣ Stop generation if EOS token is produced
        if next_token.item() == m.config.eos_token_id:
            break

    # 5️⃣ Convert final token IDs back to text
    output_ids = input_tensor[0].tolist()
    generated_text = enc.decode(output_ids)

    return generated_text
