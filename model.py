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

# Step 6 - LoopedStack
class LoopedStack(nn.Module):
    def __init__(self, blocks, n_loops):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.n_loops = n_loops

    def forward(self, x, n_loops=None, return_passes=False):
        loops = self.n_loops if n_loops is None else n_loops
        passes = []

        for _ in range(loops):
            for block in self.blocks:
                x = block(x)

            if return_passes:
                passes.append(x)

        if return_passes:
            return passes

        return x

    def block_applications(self, n_loops=None):
        loops = self.n_loops if n_loops is None else n_loops
        return len(self.blocks) * loops

# Step 7 - LoopedGPT
class LoopedGPT(nn.Module):
    def __init__(self, vocab_size, d, n_heads, n_blocks, n_loops, block_size):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)

        self.stack = LoopedStack(
            [Block(d, n_heads) for _ in range(n_blocks)],
            n_loops
        )

        self.norm = RMSNorm(d)

        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

    def embed(self, idx, start=0):
        _, T = idx.shape

        positions = torch.arange(
            start,
            start + T,
            device=idx.device
        )

        return self.tok_emb(idx) + self.pos_emb(positions)

    def forward(self, idx, n_loops=None):
        x = self.embed(idx)
        x = self.stack(x, n_loops=n_loops)
        x = self.norm(x)
        return self.lm_head(x)

# Step 8 - parameter_breakdown
def parameter_breakdown(model):
    embedding = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    blocks = sum(p.numel() for p in model.stack.parameters())
    total = sum(p.numel() for p in model.parameters())

    other = total - embedding - blocks

    unrolled_total = total + (model.stack.n_loops - 1) * blocks
    embedding_share = round(embedding / total, 4)

    return {
        "embedding": int(embedding),
        "blocks": int(blocks),
        "other": int(other),
        "total": int(total),
        "unrolled_total": int(unrolled_total),
        "embedding_share": embedding_share,
    }

# Step 9 - looped_costs
def looped_costs(model, seq_len, batch, dtype_bytes=2):
    L = len(model.stack.blocks)
    K = model.stack.n_loops
    d = model.tok_emb.embedding_dim
    V = model.tok_emb.num_embeddings

    pb = parameter_breakdown(model)
    total_params = pb["total"]
    blocks = pb["blocks"]

    block_applications = L * K
    applied_params_per_token = K * blocks + d * V

    forward_flops_per_token = 2 * applied_params_per_token
    train_flops_per_token = 6 * applied_params_per_token

    kv_cache_bytes = (
        2
        * L
        * K
        * d
        * seq_len
        * batch
        * dtype_bytes
    )

    kv_cache_bytes_if_shared = (
        2
        * L
        * d
        * seq_len
        * batch
        * dtype_bytes
    )

    optimizer_moments_bytes = 2 * 4 * total_params

    return {
        "block_applications": int(block_applications),
        "applied_params_per_token": int(applied_params_per_token),
        "forward_flops_per_token": int(forward_flops_per_token),
        "train_flops_per_token": int(train_flops_per_token),
        "kv_cache_bytes": int(kv_cache_bytes),
        "kv_cache_bytes_if_shared": int(kv_cache_bytes_if_shared),
        "optimizer_moments_bytes": int(optimizer_moments_bytes),
    }

# Step 10 - shared_gradient_check
import copy
import torch.nn.functional as F

def shared_gradient_check(model, x, y):
    K = model.stack.n_loops
    L = len(model.stack.blocks)

    # Shared model: compute loss and backpropagate.
    model.zero_grad(set_to_none=True)

    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    loss.backward()

    shared_grads = [
        block.attn.qkv.weight.grad.detach().clone()
        for block in model.stack.blocks
    ]

    # Build the unrolled twin with one distinct copy of every block
    # for every loop pass.
    unrolled = copy.deepcopy(model)
    unrolled.stack = LoopedStack(
        [
            copy.deepcopy(block)
            for _ in range(K)
            for block in model.stack.blocks
        ],
        1
    )

    # deepcopy(model) copies gradients from the shared model, so clear
    # them before computing gradients for the unrolled twin.
    unrolled.zero_grad(set_to_none=True)

    unrolled_logits = unrolled(x)
    unrolled_loss = F.cross_entropy(
        unrolled_logits.reshape(-1, unrolled_logits.size(-1)),
        y.reshape(-1)
    )
    unrolled_loss.backward()

    unrolled_grads = [
        block.attn.qkv.weight.grad.detach()
        for block in unrolled.stack.blocks
    ]

    max_abs_diff = 0.0
    per_pass_grad_norms = []

    for i in range(L):
        copy_grads = [
            unrolled_grads[p * L + i]
            for p in range(K)
        ]

        per_pass_grad_norms.append([
            round(float(grad.norm()), 4)
            for grad in copy_grads
        ])

        summed_grad = torch.stack(copy_grads, dim=0).sum(dim=0)

        diff = (shared_grads[i] - summed_grad).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)

    return {
        "max_abs_diff": float(max_abs_diff),
        "matches": bool(max_abs_diff < 1e-5),
        "per_pass_grad_norms": per_pass_grad_norms,
    }

# Step 11 - lm_loss
def lm_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

# Step 12 - train_lm
def train_lm(model, data, steps, lr=3e-3, block_size=32, batch_size=16, seed=0):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    losses = []

    for _ in range(steps):
        x, y = data.get_batch(
            "train",
            block_size=block_size,
            batch_size=batch_size,
            generator=generator
        )

        optimizer.zero_grad()
        logits = model(x)
        loss = lm_loss(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return losses

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

