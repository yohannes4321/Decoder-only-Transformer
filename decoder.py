import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from v2 import MultiHeadAttention
batch_size=32
block_size=128
max_iters=5000
eval_iterval=500
learning_rate=1e-3

tokenize=tiktoken.get_encoding("gpt2")
vocab_size = tokenize.n_vocab
n_embed = 256
device='cuda' if torch.cuda.is_available() else "cpu"
eval_iters=100
train_data=torch.tensor(tokenize.encode(train_csv),dtype=torch.long)
test_data=torch.tensor(tokenize.encode(test_csv),dtype=torch.long)
validation_data=torch.tensor(tokenize.encode(validation_csv),dtype=torch.long)
torch.manual_seed(1337)
file_path = "/content/drive/MyDrive/decoder"

with open(f"{file_path}/train.csv", "r", encoding="utf-8") as f:
    train_csv = f.read()

with open(f"{file_path}/test.csv", "r", encoding="utf-8") as f:
    test_csv = f.read()

with open(f"{file_path}/validation.csv", "r", encoding="utf-8") as f:
    validation_csv = f.read()

def get_batch(split):
  data=train_data if split =='train' else  validation_data
  ix=torch.randint(len(data)-block_size, (batch_size,))

  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:i+block_size+1] for i in ix])
  return x.to(device),y.to(device)
@torch.no_grad()
def estimate_loss():
  out={}
  model.eval()
  for split in ['train','val']:
    losses=torch.zeros(eval_iters)
    for k in range(eval_iters):
      x,y=get_batch(split)
      logits,loss=model(x,y)
      losses[k]=loss.item()
    out[split]=losses.mean()
  model.train()
  return out

class BidiagramLanguageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # to go from token_embedding to logit we need linear layer
        self.lm_head=nn.Linear(n_embed,vocab_size)
        self.postion_embedding_table=nn.Embedding(block_size,n_embed)
        self.sa_head=MultiHeadAttention(4,n_embed//4)
    def forward(self, idx, targets=None):
        B,T=idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb=self.postional_embedding_table(torch.arrage(T,device=device)) 
        x= tok_emb+pos_emb     # (B, T, n_embed)
        logits = self.lm_head(tok_emb)                  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss  # ALWAYS return a tuple

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond=idx[:, -block_size:]
            logits, loss = self(idx_cond)                # (B, T, vocab_size)
            logits = logits[:, -1, :]               # Take last time step (B, vocab_size)
            probs = F.softmax(logits, dim=-1)       # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # Append
        return idx


m = BidiagramLanguageEmbedding(vocab_size, d_model)
model = m.to(device)

optimizer=torch.optim.Adam(m.parameters(),lr=1e-3)
for steps in range(max_iters):


  if steps % eval_iterval==0:
    losses=estimate_loss()
    print(f"step {steps} train loss {losses['train']:.4f}  validation loss {losses['val']:.4f}")
  xb,yb=get_batch('train')
  logits,loss=model(xb,yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()




