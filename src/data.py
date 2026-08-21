from __future__ import annotations

from pathlib import Path
import torch

from .config import SEQ_LEN, VOCAB

COPY_MARK = 31_990
KV_DEF = 31_991
KV_QUERY = 31_992
SEP = 31_993
DATA_MAX = 30_000


def _local_fill(g: torch.Generator, n: int) -> torch.Tensor:
    x = torch.empty(n, dtype=torch.long)
    x[0:2] = torch.randint(1, DATA_MAX, (2,), generator=g)
    a = int(torch.randint(3, 97, (1,), generator=g)) | 1
    b = int(torch.randint(3, 89, (1,), generator=g)) | 1
    c = int(torch.randint(1, 997, (1,), generator=g))
    for t in range(2, n):
        x[t] = 1 + ((a * int(x[t - 1]) + b * int(x[t - 2]) + c + 17 * t) % (DATA_MAX - 1))
    return x


def _make_row(g: torch.Generator, seq_len: int = SEQ_LEN) -> torch.Tensor:
    if seq_len != 256:
        # Full structured tasks are positioned for the published 256-token run.
        # Smoke mode slices the prebuilt 256-token rows instead of regenerating.
        raise ValueError("Structured row generator is fixed to seq_len=256")
    x = _local_fill(g, seq_len + 1)
    src0, copy_len, dst0 = 8, 56, 136
    block = torch.randint(1, DATA_MAX, (copy_len,), generator=g)
    x[src0 : src0 + copy_len] = block
    x[dst0 - 1] = COPY_MARK
    x[dst0 : dst0 + copy_len] = block

    keys = torch.randperm(4000, generator=g)[:10] + 20_001
    vals = torch.randint(10_001, 18_000, (10,), generator=g)
    p = 72
    for key, val in zip(keys, vals):
        x[p : p + 4] = torch.tensor([KV_DEF, int(key), int(val), SEP])
        p += 4
    order = torch.randperm(10, generator=g)
    p = 198
    for idx in order:
        key, val = keys[idx], vals[idx]
        x[p : p + 4] = torch.tensor([KV_QUERY, int(key), int(val), SEP])
        p += 4
    return x


def make_split(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([_make_row(g) for _ in range(rows)])


def load_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=True)
    if int(cache["vocab"]) != VOCAB:
        raise ValueError("Unexpected vocabulary in benchmark cache")
    return cache
