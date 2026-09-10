"""
Looped Transformer from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Looped Transformer from Scratch.

Story: a character-level GPT whose block stack is applied twice with shared
weights. Count what looping saves (weights, optimizer state) and what it does
not (FLOPs, KV cache), verify the shared-block gradient, train looped, unrolled
and shallow models at matched settings, add adaptive depth with ACT and
Mixture-of-Recursions routing, decode with per-pass KV caches, and measure
the KV-sharing shortcut.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    data = CharData(load_shakespeare(), n_chars=200000)
    print(f"Tiny Shakespeare: {len(data.train):,} train / {len(data.val):,} val characters, vocab {data.vocab_size}")

    # ---- 1. A looped model and its costs ----
    models = build_models(data.vocab_size, d=48, n_heads=2, n_blocks=2, n_loops=2, block_size=32)
    looped = models["looped"]
    pb = parameter_breakdown(looped)
    costs = looped_costs(looped, seq_len=32, batch=1)
    print(f"\nlooped 2x2 model: {pb['total']:,} parameters ({pb['embedding_share']:.0%} embedding), unrolled twin would store {pb['unrolled_total']:,}")
    print(f"  {costs['block_applications']} block applications, {costs['train_flops_per_token']:,} training FLOPs/token, KV cache {costs['kv_cache_bytes']:,} B (shared across loops: {costs['kv_cache_bytes_if_shared']:,} B)")
    x, y = data.get_batch("train", 32, 4, torch.Generator().manual_seed(0))
    gc = shared_gradient_check(looped, x, y)
    print(f"  shared-block gradient = sum over passes: {gc['matches']} (max diff {gc['max_abs_diff']:.1e}); per-pass grad norms of block 0: {gc['per_pass_grad_norms'][0]}")

    # ---- 2. Looped vs unrolled vs shallow at matched settings ----
    report = compare_models(models, data, steps=150, lr=3e-3, block_size=32, batch_size=16)
    print("\nafter 150 steps (toy scale: ordering can flip across seeds):")
    for name, r in report.items():
        print(f"  {name:9s} params {r['params']:7,}  applications {r['block_applications']}  train {r['train_loss_final']:.3f}  val {r['val_loss']:.3f}")

    # ---- 3. Adaptive depth on the trained looped stack ----
    torch.manual_seed(1)
    head = HaltingHead(48)
    idx, _ = data.get_batch("val", 32, 2, torch.Generator().manual_seed(1))
    y_act, n_steps, ponder = act_forward(looped.stack, head, looped.embed(idx), max_loops=4)
    print(f"\nACT with an untrained halting head: mean steps {n_steps.float().mean():.2f} of 4, ponder cost {float(ponder):.3f}")
    router = RecursionRouter(48, 3)
    h = looped.embed(idx)
    _, depths_ec = mor_expert_choice(looped.stack.blocks[0], router, h, capacities=[24, 12, 4])
    _, depths_tc, bal = mor_token_choice(looped.stack.blocks[0], router, h, n_recursions=3)
    print(f"  MoR expert-choice depth histogram {depth_report(depths_ec, 3)['histogram']}, compute fraction {depth_report(depths_ec, 3)['compute_fraction']}")
    print(f"  MoR token-choice  depth histogram {depth_report(depths_tc, 3)['histogram']}, balance loss {float(bal):.3f}")

    # ---- 4. Decoding with per-pass caches, and the sharing shortcut ----
    prompt = torch.tensor([data.encode("ROMEO: ")])
    out = generate(looped, prompt, max_new_tokens=24)  # prompt + new tokens stay within block_size
    same = torch.equal(out, generate(looped, prompt, max_new_tokens=24, use_cache=False))
    print(f"\ncached decode == full forward: {same}")
    print("  " + repr(data.decode(out[0].tolist())))
    kv = kv_sharing_experiment(looped, data, n_batches=8, block_size=32, batch_size=16)
    print(f"  KV sharing across loops: val loss {kv['normal_loss']:.3f} -> {kv['shared_kv_loss']:.3f} at {kv['kv_bytes_ratio']:.0%} of the cache")


if __name__ == "__main__":
    main()

