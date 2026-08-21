# ASLA vs MLA benchmark report

- Profile: **full**
- Total parameter delta: **0.000259%**
- Best mean eval loss: **ASLA-P1**
- Best mean train throughput: **MLA**
- Lowest mean peak VRAM: **MLA**

## Mean results

| arch    |   eval_loss |   train_throughput_tok_s |   peak_vram_gib |   perplexity |   next_token_accuracy |
|:--------|------------:|-------------------------:|----------------:|-------------:|----------------------:|
| ASLA-P1 |     8.80535 |                  50781.2 |         1.18716 |      6814.04 |              0.161662 |
| MLA     |     8.95602 |                  67953.1 |         1.18235 |      7754.92 |              0.160156 |

## Interpretation guardrails

- ASLA compressed-state cache is analytical; source benchmark explicitly did not realize a KV-cache/static-paged-cache kernel.
- MLA optimized latent decode cache is analytical/paper-faithful here; attention prefill path materializes K/V for ordinary SDPA.
- Attention-only scaling measures prefill, not optimized autoregressive decode.
- 3 paired seed(s) are used in this profile; for strong statistical claims, run at least 5–10 seeds.
- ASLA routing/basis path uses generic PyTorch einsum while MLA uses dense Linear projections; custom fused kernels could materially change throughput.
- This is a ~9.28M proxy model; conclusions may not transfer monotonically to large-scale LLMs.
