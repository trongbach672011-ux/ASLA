# Limitations

This benchmark answers a narrow question at small scale. The following points matter when reading the plots or reusing the numbers.

1. **Three paired seeds.** The lower ASLA-P1 mean eval loss is mostly caused by seed `2025`. Two seeds are near ties. A larger seed count is needed for a reliable quality comparison.
2. **Small proxy model.** Both models are about 9.28M parameters. Attention behavior can change with depth, width, context length and training scale.
3. **Synthetic data.** The dataset stresses recurrence, copy and retrieval patterns. It does not test broad language, coding, reasoning or instruction-following ability.
4. **No optimized ASLA decode cache.** The `44` elements/token/layer figure is analytical. The current path materializes K/V before SDPA.
5. **No optimized MLA decode benchmark either.** The `93` elements/token/layer value follows the latent plus decoupled-RoPE accounting, while the frozen prefill path also materializes K/V.
6. **Prefill and decode are different workloads.** The latency sweep does not tell us token-by-token generation speed with a persistent cache.
7. **Different kernel friendliness.** ASLA-P1 uses routing and generic `einsum` work; MLA is mostly dense projections. A fused ASLA kernel could change the speed ranking.
8. **Hardware/software specific timings.** Latency and VRAM were measured on a Tesla T4 with the recorded PyTorch/SDPA stack. Other GPUs or kernels may behave differently.
9. **No downstream benchmark suite.** There is no MMLU-style, code, long-context or reasoning evaluation in this repo.
10. **Rerun code is a reconstruction.** The frozen data came from the supplied benchmark artifact. The repository reconstructs the reported architecture and settings, but does not claim bit-for-bit identity with the original notebook environment.
