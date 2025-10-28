import torch
import torch.nn.functional as F
import math
from decoder import encode, decode, get_batch
from decoder import train_data, val_data, test_data, block_size, batch_size  # make sure these exist

# Load model
m = torch.load("decoder.pth", map_location='cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = m.to(device)

@torch.no_grad()
def evaluate(model, split="test", num_batches=100):
    """
    Evaluate model on multiple batches from the given split (test/val/train)
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    for _ in range(num_batches):
        x, y = get_batch(split)
        x, y = x.to(device), y.to(device)

        # forward pass
        out = model(x, y)
        logits = out.logits if hasattr(out, "logits") else out

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        total_loss += ce_loss.item()
        batch_count += 1

    avg_ce_loss = total_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")

    print(f"{split.upper()} | Avg Loss: {avg_ce_loss:.4f} | Perplexity: {avg_perplexity:.4f}")
    return avg_ce_loss, avg_perplexity


if __name__ == "__main__":
    evaluate(model, split="test", num_batches=64)  # run over 200 random test batches
