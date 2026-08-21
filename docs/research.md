# Research notes

The benchmark uses papers for two things: MLA's structure/cache accounting, and the interpretation of measured attention speed.

## DeepSeek-V2 / MLA

**DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**  
https://arxiv.org/abs/2405.04434

DeepSeek-V2 introduced Multi-head Latent Attention. The part used here is the joint low-rank K/V latent together with decoupled RoPE. For an optimized MLA decode path, the cache can be expressed as the KV latent plus the shared positional-key state rather than full per-head K/V tensors.

This repo uses that structure for the MLA proxy and uses `d_c + d_h^R` for analytical cache accounting. It does not copy DeepSeek-V2's production dimensions.

## FlashAttention

**FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**  
https://arxiv.org/abs/2205.14135

The useful lesson for this benchmark is methodological: FLOPs or state size alone do not determine wall-clock attention speed. Moving data through the GPU memory hierarchy can be the bottleneck.

## FlashAttention-2

**FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**  
https://arxiv.org/abs/2307.08691

FlashAttention-2 shows how work partitioning and occupancy can move attention throughput substantially. That is why the runtime result in this repo is described as an implementation result. ASLA-P1's generic routed `einsum` path should not be treated as the architecture's final speed limit.

## How those references affect the claims here

- MLA equations and analytical cache accounting are tied to DeepSeek-V2.
- ASLA-P1 equations come from the supplied experimental implementation and its reported dimensions: latent `40`, two KV heads, four routing bases and query rank `64`.
- Training throughput and prefill latency are measured values for the code path used in the frozen run.
- Compressed cache size is shown separately because neither architecture's optimized decode cache is exercised by the frozen prefill/training benchmark.
