# Looped Transformer from Scratch

Build a character-level GPT in PyTorch whose block stack is applied more than once with shared weights, the way Nanbeige4.2-3B runs 22 blocks twice. Measure exactly what looping saves and what it does not: parameters, optimizer state, FLOPs and the KV cache, and verify that a shared block's gradient is the sum over passes. Train looped, unrolled and shallow models at matched settings on Tiny Shakespeare, add adaptive depth with ACT halting and Mixture-of-Recursions routing, decode with per-pass KV caches, and run the KV-sharing experiment that cost Nanbeige accuracy.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_shakespeare
- [x] **2.** CharData
- [x] **3.** RMSNorm
- [x] **4.** CausalSelfAttention
- [x] **5.** Block
- [x] **6.** LoopedStack
- [x] **7.** LoopedGPT
- [x] **8.** parameter_breakdown
- [x] **9.** looped_costs
- [x] **10.** shared_gradient_check
- [x] **11.** lm_loss
- [x] **12.** train_lm
- [x] **13.** estimate_loss
- [x] **14.** build_models
- [x] **15.** compare_models
- [x] **16.** HaltingHead
- [x] **17.** act_weights
- [x] **18.** act_forward
- [x] **19.** RecursionRouter
- [x] **20.** mor_expert_choice
- [x] **21.** mor_token_choice
- [ ] **22.** depth_report
- [ ] **23.** PassKVCache
- [ ] **24.** generate
- [ ] **25.** kv_sharing_experiment

---

Built on Deep-ML.
