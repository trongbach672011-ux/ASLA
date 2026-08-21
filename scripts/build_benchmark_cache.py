import torch
from pathlib import Path

VOCAB = 32000
SEQ_LEN = 256
TRAIN_ROWS = 3584
EVAL_ROWS = 512

# Reserved task markers near the top of the vocabulary.
COPY_MARK = 31990
KV_DEF = 31991
KV_QUERY = 31992
SEP = 31993
DATA_MAX = 30000


def _local_fill(g: torch.Generator, n: int) -> torch.Tensor:
    """Deterministic nonlinear recurrence with row-specific seeds."""
    x = torch.empty(n, dtype=torch.long)
    x[0:2] = torch.randint(1, DATA_MAX, (2,), generator=g)
    a = int(torch.randint(3, 97, (1,), generator=g)) | 1
    b = int(torch.randint(3, 89, (1,), generator=g)) | 1
    c = int(torch.randint(1, 997, (1,), generator=g))
    for t in range(2, n):
        x[t] = 1 + ((a * int(x[t-1]) + b * int(x[t-2]) + c + 17*t) % (DATA_MAX - 1))
    return x


def _make_row(g: torch.Generator) -> torch.Tensor:
    # Need SEQ_LEN+1 tokens because LM input/target are shifted by one.
    x = _local_fill(g, SEQ_LEN + 1)

    # Long-range induction/copy: a random 56-token span is repeated ~120 tokens later.
    src0, copy_len = 8, 56
    dst0 = 136
    block = torch.randint(1, DATA_MAX, (copy_len,), generator=g)
    x[src0:src0+copy_len] = block
    x[dst0-1] = COPY_MARK
    x[dst0:dst0+copy_len] = block

    # Key-value retrieval: definitions early, queries late. The value after a query
    # is predictable only by retrieving the matching key-value pair.
    keys = torch.randperm(4000, generator=g)[:10] + 20001
    vals = torch.randint(10001, 18000, (10,), generator=g)
    p = 72
    for key, val in zip(keys, vals):
        x[p:p+4] = torch.tensor([KV_DEF, int(key), int(val), SEP])
        p += 4

    order = torch.randperm(10, generator=g)
    p = 198
    for idx in order:
        key, val = keys[idx], vals[idx]
        x[p:p+4] = torch.tensor([KV_QUERY, int(key), int(val), SEP])
        p += 4
    return x


def make_split(rows: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.stack([_make_row(g) for _ in range(rows)])


def main():
    path = Path(__file__).resolve().parents[1] / 'data' / 'realdata_cache.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        'vocab': VOCAB,
        'train_tokens': make_split(TRAIN_ROWS, 0xA51A1234),
        'eval_tokens': make_split(EVAL_ROWS, 0x4D1A2026),
        'benchmark_meta': {
            'kind': 'deterministic_structured_lm_v1',
            'seq_len': SEQ_LEN,
            'copy_marker': COPY_MARK,
            'kv_def_marker': KV_DEF,
            'kv_query_marker': KV_QUERY,
            'sep_marker': SEP,
        },
    }
    torch.save(cache, path)
    print('Wrote', path.resolve())
    print('Train:', tuple(cache['train_tokens'].shape), 'Eval:', tuple(cache['eval_tokens'].shape))


if __name__ == '__main__':
    main()
