import torch
import torch.nn as nn
import torch.nn.functional as F
from urllib import request
from typing import Tuple
BLOCK_SIZE = 256 # This is the number of characters we feed into the model
BATCH_SIZE = 64
EMBED_SIZE = 384
NUM_EPOCHS = 5000
NUM_HEADS = 6
NUM_LAYERS = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_data = request.urlopen('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt').read().decode('utf-8')
characters = sorted(list(set(text_data)))
vocab_size = len(characters)
stoi = {ch: i for i, ch in enumerate(characters)}
itos = {i: ch for i, ch in enumerate(characters)}

encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.query = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.value = nn.Linear(EMBED_SIZE, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)
        #self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  
        q = self.query(x) 
        wei = torch.matmul(q, k.transpose(1,2)) / (C ** 0.5)
        tril = torch.tril(torch.ones(T,T, dtype=torch.float32)).to(device)
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) 
        out = torch.matmul(wei, v)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(EMBED_SIZE, EMBED_SIZE)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.projection(out)


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, out_features * 4),
                                nn.ReLU(),
                                nn.Linear(out_features * 4, in_features),
                                nn.Dropout(0.2),)

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa_heads = MultiHeadAttention(head_size, n_head)
        self.ffwd = LinearLayer(in_features=n_embed, out_features=n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
  

class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenembedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBED_SIZE)
        self.positionembedding = nn.Embedding(num_embeddings=BLOCK_SIZE, embedding_dim=EMBED_SIZE)
        self.blocks = nn.Sequential(*[Block(EMBED_SIZE, NUM_HEADS) for _ in range(NUM_LAYERS)],
                                    nn.LayerNorm(EMBED_SIZE))
        self.sa_heads = MultiHeadAttention(head_size=EMBED_SIZE//4, num_heads=4)
        self.ffwd = LinearLayer(in_features=EMBED_SIZE, out_features=EMBED_SIZE)
        self.lm_head = nn.Linear(in_features=EMBED_SIZE, out_features=vocab_size)

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        toekn_embedded = self.tokenembedding(x)
        position_embedded = self.positionembedding(torch.arange(x.size(1), device = device))
        embedded = toekn_embedded + position_embedded
        x = self.blocks(embedded)
        x = self.ffwd(x)
        logits = self.lm_head(x)
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_temp = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_temp)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
    
def main():
    model = GPTModel().to(device)
    model.load_state_dict(torch.load('shakespeareGPT.pth'))
    model.eval()
    with torch.inference_mode():
        x = torch.randint(low  = 0, high = vocab_size, size = (1, 1)).to(device)
        x = model.generate(x, 1000)
        print(decode(x[0].cpu().numpy()))

if __name__ == "__main__":
    main()