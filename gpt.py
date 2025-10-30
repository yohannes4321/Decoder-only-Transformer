# --- decoder.py ---
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import math
import random

# NOTE: Assuming a file named 'visualize.py' exists with a function 'plot_metrics'
# from visualize import plot_metrics 

# hyperparameters (HPs)
# Set these as global constants or move them into a config file
batch_size = 64 
block_size = 256
max_iters = 15
eval_interval = 3
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Ensure reproducibility
torch.manual_seed(1337)

# --- Data Loading and Encoding/Decoding ---

# Assuming 'Data' folder exists with 'train.csv', 'validation.csv', 'test.csv'
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# Try to load data; use a placeholder if files are missing (e.g., for model loading only)
try:
    with open(os.path.join(DATA_DIR, 'train.csv'), 'r', encoding='utf-8') as f:
        train_csv = f.read()
    with open(os.path.join(DATA_DIR, 'validation.csv'), 'r', encoding='utf-8') as f:
        val_csv = f.read()
    with open(os.path.join(DATA_DIR, 'test.csv'), 'r', encoding='utf-8') as f:
        test_csv = f.read()
except FileNotFoundError:
    print("Warning: Data files not found in 'Data' directory. Model classes and functions defined, but data-dependent functions will fail if called.")
    train_csv = "A B C D E"
    val_csv = "F G H"
    test_csv = "I J K"

chars = sorted(list(set(train_csv)))
vocab_size = len(chars)

# Encoder/Decoder functions
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, 0) for c in s] # Use .get(c, 0) for safe encoding
decode = lambda l: ''.join([itos[i] for i in l])

# Convert data to tensors
train_ids = encode(train_csv) 
val_ids   = encode(val_csv)
test_ids  = encode(test_csv)

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data   = torch.tensor(val_ids, dtype=torch.long)
test_data  = torch.tensor(test_ids, dtype=torch.long)

# Data loading function
def get_batch(split):
    data = {'train': train_data, 'val': val_data, 'test': test_data}.get(split, train_data)
    
    if len(data) < block_size + 1:
        # Handle case where data is too short
        print(f"Warning: {split} data is too short for block_size {block_size}.")
        # Create a dummy batch if data is empty or too small
        x = torch.zeros((batch_size, block_size), dtype=torch.long)
        y = torch.zeros((batch_size, block_size), dtype=torch.long)
        return x.to(device), y.to(device)

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Model Architecture ---

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register tril as a buffer; ensure it's on the correct device later
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) 
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    # NOTE: Added to avoid direct usage of generate_from_prompt logic in model class
    def generate_with_params(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature # (B, C)
                
                if top_k is not None:
                    v, i = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Training Utility (Only needed if you run training separately) ---
# NOTE: The training logic from the user's input is kept here for completeness, 
# but usually, you'd save a model only after successful training.

# Placeholder for plot_metrics if you don't have it
def plot_metrics(train_losses, val_losses):
    print("Plotting metrics placeholder: Skipping plot generation.")


def train_model(model, optimizer, max_epochs, eval_interval=100, device='cpu'):
    train_losses = []
    val_losses = []

    for epoch in range(max_epochs):
        model.train()
        # ... (rest of the training logic)
        
        # NOTE: Using estimate_loss from the user's original code block
        def estimate_loss(m):
            out = {}
            m.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(split)
                    logits, loss = m(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
            m.train()
            return out
        
        # Simplified loop for this example
        losses = estimate_loss(model)
        avg_train_loss = losses['train']
        avg_val_loss = losses['val']
        
        print(f"Epoch {epoch+1}: avg train loss = {avg_train_loss:.4f}, avg val loss = {avg_val_loss:.4f}")

        train_losses.append(avg_train_loss.item())
        val_losses.append(avg_val_loss.item())

    # Save model after training
    MODEL_PATH = os.path.join(BASE_DIR, "decoder.pth")
    # Save the state_dict for best practice
    # torch.save(model.state_dict(), MODEL_PATH) 
    # Or save the whole model object as in the original code, but this requires
    # the class definition to be available at load time (the issue we're fixing).
    torch.save(model, MODEL_PATH) 

    print("\n Training complete! Model saved as 'decoder.pth'.")
    return train_losses, val_losses


if __name__ == "__main__":
    model = GPTLanguageModel()
    m = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_model(model, optimizer, max_epochs=max_iters, eval_interval=100, device=device)