import torch
import torch.nn.functional as F
import math
from decoder import encode, decode, get_batch, block_size, batch_size
from decoder import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedFoward
import os

# Load model safely relative to this file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "decoder.pth")
with torch.serialization.safe_globals([GPTLanguageModel]):
    model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

@torch.no_grad()
def evaluate(split="test", num_batches=100):
    model.eval()
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(split)
        x, y = x.to(device), y.to(device)
        out,_ = model(x, y)
        logits = out.logits if hasattr(out, "logits") else out
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += ce_loss.item()

    avg_loss = total_loss / num_batches
    avg_perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    print(f"{split.upper()} | Avg Loss: {avg_loss:.4f} | Perplexity: {avg_perplexity:.4f}")
    return avg_loss, avg_perplexity

if __name__ == "__main__":
    evaluate(split="test", num_batches=64)
