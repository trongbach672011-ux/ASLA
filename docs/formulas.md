# Formulas and dimensions

Notation used below:

- hidden state: `h_t ∈ R^d`
- `d = 224`
- query heads: `H = 4`
- head size: `d_h = 56`

## ASLA-P1

### Query path

The query projection uses a rank-64 bottleneck:

\[
c_t^Q=W_{DQ}h_t
\]

\[
q_t=W_{UQ}\,\mathrm{SiLU}(c_t^Q)
\]

with `d_q = 64`.

### Routed latent K/V

Each token is compressed to a 40-dimensional latent:

\[
z_t=W_Zh_t,\qquad z_t\in\mathbb{R}^{40}
\]

A second projection produces a four-way routing distribution:

\[
r_t=\mathrm{softmax}(W_Rh_t),\qquad r_t\in\mathbb{R}^{4}
\]

Let `B^K` and `B^V` be learned basis tensors. K/V are reconstructed as

\[
k_{t,h,j}=\sum_{b=1}^{4}\sum_{\ell=1}^{40}
z_{t,\ell}r_{t,b}B^K_{b,\ell,h,j}
\]

\[
v_{t,h,j}=\sum_{b=1}^{4}\sum_{\ell=1}^{40}
z_{t,\ell}r_{t,b}B^V_{b,\ell,h,j}
\]

The supplied P1 configuration has two KV heads and four query heads, so SDPA uses grouped-query attention.

### Auxiliary loss

The training code includes a small routing-balance term and a basis regularizer:

\[
L_{aux}=10^{-3}B\sum_{b=1}^{B}(\bar r_b-1/B)^2+10^{-4}L_{basis}
\]

with `B = 4` routing bases.

The eval cross-entropy reported in this repo excludes `L_aux`.

### Analytical compressed state

A decode implementation that stores only the token latent and routing state would keep

\[
C_{ASLA}=d_z+B=40+4=44
\]

elements per token per layer.

That path is not implemented in the frozen benchmark. Before SDPA, the current code materializes K/V, giving `224` elements/token/layer in the cache-accounting table.

## MLA proxy

The MLA equations follow the low-rank KV compression and decoupled-RoPE structure from DeepSeek-V2.

### Joint KV compression

\[
c_t^{KV}=W^{DKV}h_t
\]

\[
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
\]

The benchmark uses `d_c = 69` for the KV latent.

### Query compression

\[
c_t^Q=W^{DQ}h_t
\]

The query latent rank is `65`, followed by projections for the content and positional query parts.

### Decoupled RoPE

\[
q_{t,i}=[q_{t,i}^{C};q_{t,i}^{R}]
\]

\[
k_{t,i}=[k_{t,i}^{C};k_t^{R}]
\]

The positional key `k_t^R` is shared across heads. Its dimension in this benchmark is `24`.

### Attention

\[
o_{t,i}=\sum_{j\le t}
\mathrm{softmax}_j\left(
\frac{q_{t,i}^{\top}k_{j,i}}{\sqrt{d_h}}
\right)v_{j,i}
\]

### Analytical MLA cache

Using the latent plus the shared RoPE key gives

\[
C_{MLA}=d_c+d_h^R=69+24=93
\]

elements per token per layer.

Again, that is cache accounting for an optimized decode path. The training/prefill code materializes full K/V before SDPA, so the measured path does not realize the 93-element state.
