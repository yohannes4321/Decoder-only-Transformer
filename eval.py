import torch
import torch.nn.functional as F
import math
import pickle
from decoder import get_batch, block_size, batch_size
from decoder import GPTLanguageModel, Block, Head, MultiHeadAttention, FeedFoward

# Load the model (same as generate.py)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

@torch.no_grad()
def evaluate(split="test", num_batches=100):
    """
    Evaluate the model on a given split ('train', 'val', or 'test').
    Computes average cross-entropy loss and perplexity.
    """
    total_loss = 0.0

    for _ in range(num_batches):
        x, y = get_batch(split)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x, y)
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += ce_loss.item()

    avg_loss = total_loss / num_batches
    avg_perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    print(f"{split.upper()} | Avg Loss: {avg_loss:.4f} | Perplexity: {avg_perplexity:.4f}")
    return avg_loss, avg_perplexity


if __name__ == "__main__":
    evaluate(split="val", num_batches=64)
