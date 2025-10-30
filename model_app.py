# --- model_app.py ---
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math

# ====================================================================
# I. HYPERPARAMETERS & DATA UTILITIES (From decoder.py)
# ====================================================================

# NOTE: These hyperparameters MUST match the values used when the model was trained and saved.
# Using the values provided in your decoder.py snippet:
batch_size = 2
block_size = 3 
n_embd = 8
n_head = 1
n_layer = 1
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data setup (Simulated/Placeholder for running in a single file)
# Assumes Data folder is relative to where you run this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
DATA_DIR = os.path.join(BASE_DIR, 'Data')

# Attempt to load data for tokenization
try:
    with open(os.path.join(DATA_DIR, 'train.csv'), 'r', encoding='utf-8') as f:
        train_csv = f.read()
    chars = sorted(list(set(train_csv)))
    vocab_size = len(chars)
except FileNotFoundError:
    print("Warning: Data files not found. Using placeholder tokenization.")
    chars = sorted(list("abcdefghijklmnopqrstuvwxyz "))
    vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


# ====================================================================
# II. MODEL ARCHITECTURE (From decoder.py)
# ====================================================================

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
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
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    # Unified generate function for API
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            with torch.no_grad():
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, i = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ====================================================================
# III. MODEL LOADING WITH CUSTOM UNPICKLER (The FIX)
# ====================================================================

# This is the map the CustomUnpickler will use to locate the classes
class_map = {
    'GPTLanguageModel': GPTLanguageModel,
    'Block': Block,
    'Head': Head,
    'MultiHeadAttention': MultiHeadAttention,
    'FeedFoward': FeedFoward,
}

# The unpickler needs to look in the right place for the custom classes
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 1. If the class name is in our custom map (i.e., defined in this file), return it directly.
        if name in class_map:
            return class_map[name]
        
        # 2. The original model was saved in the training script's '__main__'.
        # By loading it here, the unpickler looks for it in 'model_app', 
        # but the class definition is now defined *within* 'model_app.py'.
        # No redirection is strictly necessary if the classes are imported/defined here, 
        # but we keep the logic for robustness against original training environment.
        if module == "__main__" or module == "decoder":
             # Since all classes are now defined directly in THIS file, 
             # the pickler should find them in the current module's namespace.
             # We let the parent class try to find it first, or use the map.
             pass
        
        return super().find_class(module, name)

# Global model instance
m = None
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        unpickler = CustomUnpickler(f)
        m = unpickler.load()
        m.to(device)
        print(f"Model loaded successfully from {MODEL_PATH} and moved to {device}.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    # Fallback to creating a new model instance
    m = GPTLanguageModel()
    m.to(device)
    print("Warning: Using a freshly initialized (untrained) GPTLanguageModel.")


# ====================================================================
# IV. GENERATION LOGIC (From generate_text.py)
# ====================================================================

def generate_from_prompt(prompt: str, max_new_tokens=200, temperature=0.7, top_k=50):
    # Use the global model instance 'm'
    global m
    if m is None:
        return "Model not initialized."
    
    m.eval()
    
    # Ensure all token IDs are valid based on the loaded vocab
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    # Use the model's unified generate function
    output_tensor = m.generate(
        prompt_ids, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_k=top_k
    )

    return decode(output_tensor[0].tolist())

# ====================================================================
# V. FASTAPI APPLICATION SETUP (From app.py)
# ====================================================================

app = FastAPI(title="GPT Language Model API")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50

@app.get("/")
def read_root():
    return {"message": "Welcome to the GPT Language Model API! Use the /generate endpoint."}

@app.post("/generate")
def generate_text_endpoint(request: GenerationRequest):
    try:
        generated_text = generate_from_prompt(
            request.prompt, 
            request.max_new_tokens, 
            request.temperature, 
            request.top_k
        )
        return {"input_prompt": request.prompt, "generated_text": generated_text}
    except Exception as e:
        return {"error": str(e), "detail": "An error occurred during text generation."}

# NOTE: The training logic has been completely removed to avoid running heavy computations
# when uvicorn loads the application.

# ====================================================================
# VI. UVICORN START COMMAND
# ====================================================================

if __name__ == "__main__":
    port=int(os.environ.get("PORT", 8000)) # Defaulting to 8000 for standard practice
    uvicorn.run("model_app:app", host="0.0.0.0", port=port, log_level="info", reload=False)