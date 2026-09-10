"""
Looped Transformer from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_shakespeare
import os
import tempfile
import urllib.request

def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    file_path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Step 2 - CharData
import torch

class CharData:
    def __init__(self, text, n_chars=200000):
        text = text[:n_chars]

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        encoded = torch.tensor(
            self.encode(text),
            dtype=torch.int64
        )

        split_idx = int(0.9 * len(encoded))
        self.train = encoded[:split_idx]
        self.val = encoded[split_idx:]

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(self, split, block_size, batch_size, generator=None):
        data = self.train if split == "train" else self.val

        ix = torch.randint(
            len(data) - block_size,
            (batch_size,),
            generator=generator
        )

        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

        return x, y

# Step 3 - RMSNorm
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        rms = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight

# Step 4 - CausalSelfAttention
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def project_qkv(self, x):
        B, T, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        return q, k, v

    def attend(self, q, k, v):
        _, _, T, _ = q.shape
        _, _, S, _ = k.shape

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        offset = S - T
        mask = torch.tril(
            torch.ones(T, S, dtype=torch.bool, device=q.device),
            diagonal=offset
        )

        scores = scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min
        )

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous()
        out = out.view(out.shape[0], out.shape[1], self.d)

        return self.proj(out)

    def forward(self, x):
        return self.attend(*self.project_qkv(x))

# Step 5 - Block
class Block(nn.Module):
    def __init__(self, d, n_heads, mlp_mult=4):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Linear(mlp_mult * d, d)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# Step 6 - LoopedStack (not yet solved)
# TODO: implement

# Step 7 - LoopedGPT (not yet solved)
# TODO: implement

# Step 8 - parameter_breakdown (not yet solved)
# TODO: implement

# Step 9 - looped_costs (not yet solved)
# TODO: implement

# Step 10 - shared_gradient_check (not yet solved)
# TODO: implement

# Step 11 - lm_loss (not yet solved)
# TODO: implement

# Step 12 - train_lm (not yet solved)
# TODO: implement

# Step 13 - estimate_loss (not yet solved)
# TODO: implement

# Step 14 - build_models (not yet solved)
# TODO: implement

# Step 15 - compare_models (not yet solved)
# TODO: implement

# Step 16 - HaltingHead (not yet solved)
# TODO: implement

# Step 17 - act_weights (not yet solved)
# TODO: implement

# Step 18 - act_forward (not yet solved)
# TODO: implement

# Step 19 - RecursionRouter (not yet solved)
# TODO: implement

# Step 20 - mor_expert_choice (not yet solved)
# TODO: implement

# Step 21 - mor_token_choice (not yet solved)
# TODO: implement

# Step 22 - depth_report (not yet solved)
# TODO: implement

# Step 23 - PassKVCache (not yet solved)
# TODO: implement

# Step 24 - generate (not yet solved)
# TODO: implement

# Step 25 - kv_sharing_experiment (not yet solved)
# TODO: implement

