from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ASLA_KV_HEADS,
    ASLA_LATENT,
    ASLA_Q_RANK,
    D_MODEL,
    FFN_MULT,
    HEAD_DIM,
    MLA_KV_RANK,
    MLA_Q_RANK,
    N_HEADS,
    N_LAYERS,
    NUM_BASES,
    ROPE_DIM,
    VOCAB,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        return x * scale * self.weight


def _rope(x: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to the complete last dimension; dimension must be even."""
    dim = x.size(-1)
    if dim % 2:
        raise ValueError("RoPE dimension must be even")
    half = dim // 2
    pos = torch.arange(x.size(-2), device=x.device, dtype=torch.float32)
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10_000 ** (idx / max(half, 1)))
    # Same mild frequency stretch used by the supplied ASLA source.
    factors = 1.0 + 3.0 * idx / max(half - 1, 1)
    angle = pos[:, None] * inv_freq[None, :] / factors[None, :]
    cos = angle.cos().to(x.dtype)[None, None]
    sin = angle.sin().to(x.dtype)[None, None]
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)


def _causal_padding_mask(valid: torch.Tensor, length: int) -> torch.Tensor:
    causal = torch.ones((length, length), dtype=torch.bool, device=valid.device).tril()
    return causal[None, None] & valid[:, None, None, :]


class ASLAP1Attention(nn.Module):
    """ASLA-P1 proxy reconstructed from the supplied benchmark source.

    K/V are produced from a token latent z and a token-dependent soft routing
    vector over learned K/V basis tensors. The current PyTorch path materializes
    K/V before SDPA; the 44-element compressed cache is therefore analytical.
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(D_MODEL, ASLA_Q_RANK, bias=False),
            nn.SiLU(),
            nn.Linear(ASLA_Q_RANK, N_HEADS * HEAD_DIM, bias=False),
        )
        self.kv = nn.Linear(D_MODEL, ASLA_LATENT, bias=False)
        self.route = nn.Linear(D_MODEL, NUM_BASES, bias=False)
        self.k_basis = nn.Parameter(
            torch.randn(NUM_BASES, ASLA_LATENT, ASLA_KV_HEADS, HEAD_DIM) * 0.02
        )
        self.v_basis = nn.Parameter(
            torch.randn(NUM_BASES, ASLA_LATENT, ASLA_KV_HEADS, HEAD_DIM) * 0.02
        )
        self.out = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.aux_loss = torch.tensor(0.0)

    def _basis_loss(self) -> torch.Tensor:
        losses = []
        for basis in (self.k_basis, self.v_basis):
            flat = F.normalize(basis.float().reshape(NUM_BASES, -1), dim=-1)
            gram = flat @ flat.T
            eye = torch.eye(NUM_BASES, device=basis.device)
            losses.append((gram - eye).square().mean())
        return torch.stack(losses).mean()

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        valid = (
            torch.ones((b, t), dtype=torch.bool, device=x.device)
            if attention_mask is None
            else attention_mask.bool()
        )

        q = self.q(x).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        latent = self.kv(x)
        route = self.route(x).softmax(-1)
        k = torch.einsum("btl,btr,rlhd->bthd", latent, route, self.k_basis).transpose(1, 2)
        v = torch.einsum("btl,btr,rlhd->bthd", latent, route, self.v_basis).transpose(1, 2)

        q_rope, q_tail = q[..., :ROPE_DIM], q[..., ROPE_DIM:]
        k_rope, k_tail = k[..., :ROPE_DIM], k[..., ROPE_DIM:]
        q = torch.cat((_rope(q_rope), q_tail), dim=-1)
        k = torch.cat((_rope(k_rope), k_tail), dim=-1)

        mask = _causal_padding_mask(valid, t)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=(ASLA_KV_HEADS != N_HEADS)
        )
        out = self.out(attended.transpose(1, 2).reshape(b, t, -1))

        usage = route.float().mean((0, 1))
        self.aux_loss = (
            1e-3 * NUM_BASES * (usage - 1.0 / NUM_BASES).square().sum()
            + 1e-4 * self._basis_loss()
        )
        return out * valid.unsqueeze(-1)


class MLAAttention(nn.Module):
    """DeepSeek-V2-style MLA proxy with decoupled RoPE.

    The dimensions are chosen so this attention layer is parameter-parity with
    ASLA-P1: 124,550 vs 124,544 parameters per layer.
    """

    def __init__(self):
        super().__init__()
        content_dim = HEAD_DIM - ROPE_DIM
        # Biases on the two down-projections account for the exact 134-parameter
        # parity adjustment present in the supplied benchmark report.
        self.q_down = nn.Linear(D_MODEL, MLA_Q_RANK, bias=True)
        self.q_content = nn.Linear(MLA_Q_RANK, N_HEADS * content_dim, bias=False)
        self.q_rope = nn.Linear(MLA_Q_RANK, N_HEADS * ROPE_DIM, bias=False)

        self.kv_down = nn.Linear(D_MODEL, MLA_KV_RANK, bias=True)
        self.k_content = nn.Linear(MLA_KV_RANK, N_HEADS * content_dim, bias=False)
        self.v_up = nn.Linear(MLA_KV_RANK, N_HEADS * HEAD_DIM, bias=False)
        self.k_rope = nn.Linear(D_MODEL, ROPE_DIM, bias=False)
        self.out = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        valid = (
            torch.ones((b, t), dtype=torch.bool, device=x.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        content_dim = HEAD_DIM - ROPE_DIM

        c_q = self.q_down(x)
        q_c = self.q_content(c_q).view(b, t, N_HEADS, content_dim).transpose(1, 2)
        q_r = self.q_rope(c_q).view(b, t, N_HEADS, ROPE_DIM).transpose(1, 2)
        q_r = _rope(q_r)

        c_kv = self.kv_down(x)
        k_c = self.k_content(c_kv).view(b, t, N_HEADS, content_dim).transpose(1, 2)
        v = self.v_up(c_kv).view(b, t, N_HEADS, HEAD_DIM).transpose(1, 2)
        k_r = self.k_rope(x).view(b, t, 1, ROPE_DIM).transpose(1, 2)
        k_r = _rope(k_r).expand(-1, N_HEADS, -1, -1)

        q = torch.cat((q_c, q_r), dim=-1)
        k = torch.cat((k_c, k_r), dim=-1)
        mask = _causal_padding_mask(valid, t)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = self.out(attended.transpose(1, 2).reshape(b, t, -1))
        return out * valid.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.n1 = RMSNorm(D_MODEL)
        self.n2 = RMSNorm(D_MODEL)
        self.attn = ASLAP1Attention() if arch == "ASLA-P1" else MLAAttention()
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_MULT * D_MODEL),
            nn.GELU(),
            nn.Linear(FFN_MULT * D_MODEL, D_MODEL),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid = (
            torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            if attention_mask is None
            else attention_mask.bool()
        )
        x = x + self.attn(self.n1(x), valid)
        x = x + self.ffn(self.n2(x))
        return x * valid.unsqueeze(-1)


class TinyLM(nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        if arch not in {"ASLA-P1", "MLA"}:
            raise ValueError(f"Unknown architecture: {arch}")
        self.arch = arch
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block(arch) for _ in range(N_LAYERS)])
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.head.weight = self.embed.weight

    def forward(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        include_aux: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        valid = (
            torch.ones_like(tokens, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        x = self.embed(tokens) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid)
        logits = self.head(self.norm(x))

        if targets is None:
            return None, logits
        masked_targets = targets.masked_fill(~valid, -100)
        ce = F.cross_entropy(
            logits.reshape(-1, VOCAB), masked_targets.reshape(-1), ignore_index=-100
        )
        if include_aux and self.arch == "ASLA-P1":
            aux = torch.stack([b.attn.aux_loss.to(ce.device) for b in self.blocks]).mean()
            ce = ce + aux
        return ce, logits


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def attention_parameters(model: TinyLM) -> int:
    return count_parameters(model.blocks[0].attn)
