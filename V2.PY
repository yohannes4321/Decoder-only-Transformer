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
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        Batch_size,seq_length,dimension=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        wei = q @ k.transpose(-2,-1) * dimension ** -0.5
        wei=wei.masked_fill(self.tril[:seq_length ,:seq_length]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1) #(B,T,T)
        wei=self.dropout(wei)
        v=self.value(x)
        out = wei @ v
        return out 
class MultiHeadAttention(nn.Module):
    """multiple head self attention in parallel """
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embed,n_embed)
        nn.Dropout(dropout)
    def forward(self,x):
        out= torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return out
class FeedForward(nn.Module):
    """single linear layer followed by non linearity """
    def __init__(self,n_embed):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLu(),
            nn.Linear(n_embed,4*n_embed),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)
class Block(nn.Module):
    """ Transformer block communication followed by computation """
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embed)
        self.ln1=nn.LayerNorm(n_embed)
        self.ln2=nn.LayerNorm(n_embed)
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x
    

    


