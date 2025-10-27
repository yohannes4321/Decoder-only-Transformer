import torch
import torch.nn as nn 
import torch.nn.functional as F
n_embed=512
head_size=64
class Head(nn.Module):
    """one head self Attention """
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(head_size,head_size)))
    def forward(self,x):
        Batch_size,seq_length,dimension=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei = q @ k.transpose(-2,-1) * dimension ** -0.5
        wei=wei.masked_fill(self.tril[:seq_length ,:seq_length]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1) #(B,T,T)
        v=self.value(x)
        out = wei @ v
        return out 
class MultiHeadAttention(nn.Module):
    """multiple head self attention in parallel """
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1)
    

    


