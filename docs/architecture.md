# Architecture

The Transformer backbone is fixed across both runs. The experiment only swaps the attention block.

```mermaid
flowchart LR
  X[hidden state h_t] --> A
  X --> M

  subgraph A[ASLA-P1]
    A1[Q projection\n224 → 64 → 224]
    A2[token latent z_t\n224 → 40]
    A3[router r_t\n224 → 4]
    A4[K/V basis banks\n4 × 40 × 2 × 56]
    A5[materialize K/V\n2 KV heads]
    A6[RoPE + GQA SDPA]
    A7[output projection]
    A1 --> A6
    A2 --> A4
    A3 --> A4
    A4 --> A5 --> A6 --> A7
  end

  subgraph M[MLA proxy]
    M1[Q latent c^Q\n224 → 65]
    M2[content Q + Q RoPE]
    M3[KV latent c^KV\n224 → 69]
    M4[content K + V]
    M5[shared RoPE key\n224 → 24]
    M6[materialize K/V\n4 heads in training path]
    M7[SDPA]
    M8[output projection]
    M1 --> M2 --> M7
    M3 --> M4 --> M6 --> M7
    M5 --> M7 --> M8
  end
```

ASLA-P1 reconstructs K/V from a 40-dimensional token latent and a four-way router over learned basis banks. The supplied P1 setup has two KV heads, then uses grouped-query attention for four query heads.

The MLA proxy compresses K/V into a shared latent, reconstructs content K/V, and keeps a separate RoPE key path. That follows the main MLA structure introduced in DeepSeek-V2, with dimensions chosen here to match ASLA-P1's parameter budget as closely as possible.

## Parameter match

| Item | ASLA-P1 | MLA |
|---|---:|---:|
| Attention params/layer | 124,544 | 124,550 |
| Total params | 9,278,304 | 9,278,328 |
| Total parameter delta | — | **0.000259%** |

A 24-parameter difference over the full model is small enough that capacity is unlikely to explain the runtime gap by itself.

The cache numbers used elsewhere in the repo should not be read from this diagram as measured memory. Both training/prefill paths materialize K/V for SDPA. The smaller latent-state figures are analytical decode-cache accounting; see [`formulas.md`](formulas.md) and [`methodology.md`](methodology.md).
