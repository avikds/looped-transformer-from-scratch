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
- [ ] **6.** LoopedStack
- [ ] **7.** LoopedGPT
- [ ] **8.** parameter_breakdown
- [ ] **9.** looped_costs
- [ ] **10.** shared_gradient_check
- [ ] **11.** lm_loss
- [ ] **12.** train_lm
- [ ] **13.** estimate_loss
- [ ] **14.** build_models
- [ ] **15.** compare_models
- [ ] **16.** HaltingHead
- [ ] **17.** act_weights
- [ ] **18.** act_forward
- [ ] **19.** RecursionRouter
- [ ] **20.** mor_expert_choice
- [ ] **21.** mor_token_choice
- [ ] **22.** depth_report
- [ ] **23.** PassKVCache
- [ ] **24.** generate
- [ ] **25.** kv_sharing_experiment

---

Built on Deep-ML.
