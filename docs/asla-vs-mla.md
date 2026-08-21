# ASLA-P1 vs MLA

## Frozen means

| Metric | ASLA-P1 | MLA | Notes |
|---|---:|---:|---|
| Eval loss ↓ | **8.8053** | 8.9560 | ASLA mean is pulled down by seed 2025 |
| Perplexity ↓ | **6814.0** | 7754.9 | follows the eval-loss mean |
| Next-token accuracy ↑ | **0.16166** | 0.16016 | very small absolute gap |
| Train throughput ↑ | 50,781 tok/s | **67,953 tok/s** | MLA is ~33.8% faster |
| Peak VRAM ↓ | 1.1872 GiB | **1.1824 GiB** | effectively tied |
| Tail CE std ↓ | 0.1223 | **0.08514** | lower variation for MLA |
| Prefill p50 @ 2K ↓ | 13.155 ms | **4.347 ms** | MLA is ~3.03× faster here |
| Analytical compressed state ↓ | **44** | 93 | ASLA is smaller on paper; neither optimized decode cache is benchmarked |
| Total params | 9,278,304 | 9,278,328 | parameter matched |

## Quality

The mean favors ASLA-P1, but three seeds do not give a stable quality ranking. The paired eval-loss deltas `(ASLA - MLA)` are `+0.00116`, `-0.45094`, and `-0.00224`. Seeds `1234` and `3407` are near ties. Most of the mean difference comes from seed `2025`.

That makes the quality result worth following up, especially at more seeds and larger scale, but too weak for a broad superiority claim.

## Runtime

MLA is consistently faster in the frozen implementation. Training throughput is about 33.8% higher, and the 2K attention-only prefill latency is about 3.03× lower.

The implementation paths are not equally optimized. ASLA-P1 uses routing plus generic tensor contractions; MLA is mostly dense projection work that maps well to existing kernels. A fused ASLA kernel could move these numbers. The current benchmark only tells us what these implementations do on a T4 with this PyTorch/SDPA stack.

## Cache accounting

The analytical compressed states are `44` elements/token/layer for ASLA-P1 and `93` for MLA in this parameterization. Those values are useful for reasoning about decode-cache potential.

They are not the K/V memory used by the current training/prefill path. The current materialized path is `224` elements/token/layer for ASLA-P1 and `448` for MLA. A proper decode benchmark should implement each architecture's intended cache directly and measure latency, memory, bandwidth and generation throughput together.
