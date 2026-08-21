from __future__ import annotations

LOWER_IS_BETTER = {
    "quality": "eval_loss_mean",
    "prefill_latency": "prefill_latency_ms_p50_seq2048",
    "vram_efficiency": "peak_vram_gib_mean",
    "stability": "tail_ce_std_mean",
    "cache_efficiency_theoretical": "theoretical_cache_elements_per_token_layer",
}
HIGHER_IS_BETTER = {"throughput": "train_throughput_tok_s_mean"}


def ratio_to_best(rows: list[dict]) -> list[dict]:
    """Return 0..1 ratio-to-best scores without exaggerating two-model gaps.

    For lower-is-better metrics: score = best / value.
    For higher-is-better metrics: score = value / best.
    The best model on each axis is exactly 1.0; the other retains its relative
    magnitude instead of collapsing to 0 as two-point min-max normalization would.
    """
    out = [dict(row) for row in rows]
    for score_name, metric in LOWER_IS_BETTER.items():
        best = min(float(row[metric]) for row in rows)
        for row in out:
            row[f"radar_{score_name}"] = best / float(row[metric])
    for score_name, metric in HIGHER_IS_BETTER.items():
        best = max(float(row[metric]) for row in rows)
        for row in out:
            row[f"radar_{score_name}"] = float(row[metric]) / best
    return out
