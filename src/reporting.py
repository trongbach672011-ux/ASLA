from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

from .config import ASLA_THEORETICAL_CACHE, MLA_THEORETICAL_CACHE
from .scoring import ratio_to_best

SUMMARY_METRICS = [
    "eval_loss",
    "perplexity",
    "next_token_accuracy",
    "copy_span_accuracy",
    "train_throughput_tok_s",
    "p50_step_seconds",
    "p95_step_seconds",
    "peak_vram_gib",
    "tail_ce_std",
    "total_params",
    "attention_params_per_layer",
]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prefill_2k(prefill_rows: list[dict]) -> dict[str, float]:
    found = {
        row["arch"]: float(row["latency_ms_p50"])
        for row in prefill_rows
        if int(row["seq_len"]) == 2048
    }
    if set(found) != {"ASLA-P1", "MLA"}:
        raise ValueError("Need seq_len=2048 prefill latency for ASLA-P1 and MLA")
    return found


def summarize_runs(runs: list[dict], prefill_rows: list[dict]) -> list[dict]:
    prefill = _prefill_2k(prefill_rows)
    arches = sorted({row["arch"] for row in runs}, key=lambda x: (x != "ASLA-P1", x))
    summary = []
    for arch in arches:
        subset = [row for row in runs if row["arch"] == arch]
        out: dict[str, object] = {"arch": arch, "n_seeds": len(subset)}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out["prefill_latency_ms_p50_seq2048"] = prefill[arch]
        out["theoretical_cache_elements_per_token_layer"] = (
            ASLA_THEORETICAL_CACHE if arch == "ASLA-P1" else MLA_THEORETICAL_CACHE
        )
        summary.append(out)
    return ratio_to_best(summary)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
