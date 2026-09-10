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

# Step 13 - estimate_loss
def estimate_loss(
    model,
    data,
    split="val",
    n_batches=8,
    block_size=32,
    batch_size=16,
    seed=0
):
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = data.get_batch(
                split,
                block_size=block_size,
                batch_size=batch_size,
                generator=generator
            )

            logits = model(x)
            loss = lm_loss(logits, y)
            losses.append(float(loss.item()))

    return sum(losses) / len(losses)

# Step 14 - build_models
def build_models(
    vocab_size,
    d,
    n_heads,
    n_blocks,
    n_loops,
    block_size,
    seed=0
):
    torch.manual_seed(seed)
    looped = LoopedGPT(
        vocab_size,
        d,
        n_heads,
        n_blocks,
        n_loops,
        block_size
    )

    torch.manual_seed(seed)
    unrolled = LoopedGPT(
        vocab_size,
        d,
        n_heads,
        n_blocks * n_loops,
        1,
        block_size
    )

    torch.manual_seed(seed)
    shallow = LoopedGPT(
        vocab_size,
        d,
        n_heads,
        n_blocks,
        1,
        block_size
    )

    return {
        "looped": looped,
        "unrolled": unrolled,
        "shallow": shallow,
    }

# Step 15 - compare_models
def compare_models(
    models,
    data,
    steps,
    lr=3e-3,
    block_size=32,
    batch_size=16,
    seed=0
):
    report = {}

    for name, model in models.items():
        losses = train_lm(
            model,
            data,
            steps=steps,
            lr=lr,
            block_size=block_size,
            batch_size=batch_size,
            seed=seed
        )

        val_loss = estimate_loss(
            model,
            data,
            split="val",
            n_batches=8,
            block_size=block_size,
            batch_size=batch_size,
            seed=seed
        )

        report[name] = {
            "params": int(sum(p.numel() for p in model.parameters())),
            "block_applications": int(model.stack.block_applications()),
            "train_loss_final": round(float(losses[-1]), 4),
            "val_loss": round(float(val_loss), 4),
        }

    return report

# Step 16 - HaltingHead
class HaltingHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, h):
        return torch.sigmoid(self.linear(h)).squeeze(-1)

# Step 17 - act_weights
def act_weights(halt_probs, threshold=0.99):
    cumulative = halt_probs.cumsum(dim=-1)

    reached = cumulative >= threshold
    has_reached = reached.any(dim=-1)

    # First pass whose cumulative halting probability reaches threshold.
    first_reached = reached.to(torch.int64).argmax(dim=-1)

    # If threshold is never reached, halt on the final pass.
    S = halt_probs.size(-1)
    N = torch.where(
        has_reached,
        first_reached,
        torch.full_like(first_reached, S - 1)
    )

    # Sum of halting probabilities before pass N.
    cumulative_before = torch.zeros_like(N, dtype=halt_probs.dtype)
    has_previous = N > 0

    if has_previous.any():
        cumulative_before = torch.where(
            has_previous,
            cumulative.gather(-1, (N - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
            cumulative_before
        )

    remainder = 1.0 - cumulative_before

    # Use the original probabilities before N, the remainder at N,
    # and zero after N.
    S_idx = torch.arange(S, device=halt_probs.device).view(1, 1, S)
    weights = torch.where(
        S_idx < N.unsqueeze(-1),
        halt_probs,
        torch.zeros_like(halt_probs)
    )

    weights = torch.where(
        S_idx == N.unsqueeze(-1),
        remainder.unsqueeze(-1),
        weights
    )

    n_steps = (N + 1).to(torch.int64)
    ponder = n_steps.to(halt_probs.dtype) + remainder

    return weights, n_steps, ponder

# Step 18 - act_forward
def act_forward(stack, head, x, max_loops, threshold=0.99):
    states = stack(
        x,
        n_loops=max_loops,
        return_passes=True
    )

    halt_probs = torch.stack(
        [head(state) for state in states],
        dim=-1
    )

    weights, n_steps, ponder = act_weights(
        halt_probs,
        threshold=threshold
    )

    y = sum(
        state * weights[..., s].unsqueeze(-1)
        for s, state in enumerate(states)
    )

    return y, n_steps, ponder.mean()

# Step 19 - RecursionRouter
class RecursionRouter(nn.Module):
    def __init__(self, d, n_recursions):
        super().__init__()
        self.linear = nn.Linear(d, n_recursions)

    def expert_scores(self, h, r):
        return torch.sigmoid(self.linear(h)[..., r])

    def token_probs(self, h):
        return torch.softmax(self.linear(h), dim=-1)

# Step 20 - mor_expert_choice
def mor_expert_choice(block, router, h, capacities):
    B, T, _ = h.shape

    active = torch.ones(
        B,
        T,
        dtype=torch.bool,
        device=h.device
    )

    depths = torch.zeros(
        B,
        T,
        dtype=torch.int64,
        device=h.device
    )

    for r, capacity in enumerate(capacities):
        scores = router.expert_scores(h, r)

        masked_scores = scores.masked_fill(~active, float("-inf"))

        k = min(capacity, T)
        _, top_indices = torch.topk(
            masked_scores,
            k=k,
            dim=1
        )

        selected = torch.zeros(
            B,
            T,
            dtype=torch.bool,
            device=h.device
        )
        selected.scatter_(1, top_indices, True)

        selected = selected & active

        out = block(h)

        h = torch.where(
            selected.unsqueeze(-1),
            h + scores.unsqueeze(-1) * out,
            h
        )

        depths = depths + selected.to(torch.int64)

        active = selected

    return h, depths

# Step 21 - mor_token_choice
def mor_token_choice(block, router, h, n_recursions):
    probs = router.token_probs(h)

    chosen = torch.argmax(probs, dim=-1)
    depths = chosen + 1

    gate = probs.gather(
        -1,
        chosen.unsqueeze(-1)
    ).squeeze(-1)

    for r in range(1, n_recursions + 1):
        selected = depths >= r
        out = block(h)

        h = torch.where(
            selected.unsqueeze(-1),
            h + gate.unsqueeze(-1) * out,
            h
        )

    total_tokens = depths.numel()
    balance_loss = torch.tensor(
        0.0,
        dtype=probs.dtype,
        device=probs.device
    )

    for r in range(1, n_recursions + 1):
        chosen_r = chosen == (r - 1)
        f_r = chosen_r.float().mean()
        P_r = probs[..., r - 1].mean()
        balance_loss = balance_loss + f_r * P_r

    balance_loss = n_recursions * balance_loss

    return h, depths.to(torch.int64), balance_loss

# Step 22 - depth_report
def depth_report(depths, n_recursions):
    histogram = torch.bincount(
        depths.reshape(-1).to(torch.int64),
        minlength=n_recursions + 1
    )[:n_recursions + 1]

    mean_depth = float(depths.float().mean().item())
    compute_fraction = mean_depth / n_recursions

    return {
        "histogram": histogram.tolist(),
        "mean_depth": round(mean_depth, 4),
        "compute_fraction": round(compute_fraction, 4),
    }

# Step 23 - PassKVCache
class PassKVCache:
    def __init__(self):
        self.store = {}

    def append(self, key, k, v):
        if key in self.store:
            cached_k, cached_v = self.store[key]
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        self.store[key] = (k, v)
        return k, v

    def __len__(self):
        if not self.store:
            return 0

        k, _ = next(iter(self.store.values()))
        return k.size(2)

    def n_entries(self):
        return len(self.store)


def block_step(block, x, cache, key):
    h = block.norm1(x)
    q, k, v = block.attn.project_qkv(h)

    k, v = cache.append(key, k, v)

    x = x + block.attn.attend(q, k, v)
    x = x + block.mlp(block.norm2(x))

    return x

# Step 24 - generate
@torch.no_grad()
def generate(model, idx, max_new_tokens, n_loops=None, use_cache=True):
    model.eval()

    loops = model.stack.n_loops if n_loops is None else n_loops

    if not use_cache:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.block_size:]
            logits = model(idx_cond, n_loops=loops)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    cache = PassKVCache()

    # Process the full prompt first, using its true position offset.
    start = 0
    x_new = idx
    h = model.embed(x_new, start=start)

    for p in range(loops):
        for b, block in enumerate(model.stack.blocks):
            h = block_step(block, h, cache, (p, b))

    start += x_new.size(1)

    logits = model.lm_head(model.norm(h))[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    idx = torch.cat([idx, next_token], dim=1)

    # Generate one token at a time using the per-pass, per-block cache.
    for _ in range(max_new_tokens - 1):
        x_new = idx[:, -1:]
        h = model.embed(x_new, start=start)

        for p in range(loops):
            for b, block in enumerate(model.stack.blocks):
                h = block_step(block, h, cache, (p, b))

        start += 1

        logits = model.lm_head(model.norm(h))[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

    return idx

# Step 25 - kv_sharing_experiment
def forward_shared_kv(model, idx):
    h = model.embed(idx)
    saved_kv = []

    # Pass 1: run every block normally and save its K/V.
    for block in model.stack.blocks:
        h_norm = block.norm1(h)
        q, k, v = block.attn.project_qkv(h_norm)
        saved_kv.append((k, v))

        h = h + block.attn.attend(q, k, v)
        h = h + block.mlp(block.norm2(h))

    # Later passes: compute fresh Q from the new hidden states,
    # but reuse the K/V produced during pass 1.
    for _ in range(1, model.stack.n_loops):
        for b, block in enumerate(model.stack.blocks):
            h_norm = block.norm1(h)
            q, _, _ = block.attn.project_qkv(h_norm)

            k_saved, v_saved = saved_kv[b]
            h = h + block.attn.attend(q, k_saved, v_saved)
            h = h + block.mlp(block.norm2(h))

    return model.lm_head(model.norm(h))


def kv_sharing_experiment(
    model,
    data,
    n_batches=8,
    block_size=32,
    batch_size=16,
    seed=0
):
    model.eval()
    generator = torch.Generator().manual_seed(seed)

    normal_losses = []
    shared_kv_losses = []

    with torch.no_grad():
        for _ in range(n_batches):
            x, y = data.get_batch(
                "val",
                block_size=block_size,
                batch_size=batch_size,
                generator=generator
            )

            normal_logits = model(x)
            normal_losses.append(float(lm_loss(normal_logits, y).item()))

            shared_kv_logits = forward_shared_kv(model, x)
            shared_kv_losses.append(
                float(lm_loss(shared_kv_logits, y).item())
            )

    normal_loss = sum(normal_losses) / len(normal_losses)
    shared_kv_loss = sum(shared_kv_losses) / len(shared_kv_losses)

    return {
        "normal_loss": round(normal_loss, 4),
        "shared_kv_loss": round(shared_kv_loss, 4),
        "kv_bytes_ratio": 1 / model.stack.n_loops,
    }

