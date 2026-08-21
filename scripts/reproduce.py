#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reporting import read_csv, summarize_runs, write_csv


def plot_radar(rows: list[dict], output: Path) -> None:
    import matplotlib.pyplot as plt

    axes = [
        ("Quality\n(eval loss)", "radar_quality"),
        ("Train\nthroughput", "radar_throughput"),
        ("Prefill latency\n@ 2K", "radar_prefill_latency"),
        ("Peak VRAM\nefficiency", "radar_vram_efficiency"),
        ("Training\nstability", "radar_stability"),
        ("Theoretical\ncache efficiency", "radar_cache_efficiency_theoretical"),
    ]
    labels = [x[0] for x in axes]
    keys = [x[1] for x in axes]
    angles = [2 * math.pi * i / len(labels) for i in range(len(labels))]
    closed_angles = angles + angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for row in rows:
        values = [float(row[k]) for k in keys]
        closed = values + values[:1]
        line = ax.plot(closed_angles, closed, linewidth=2, label=row["arch"])[0]
        ax.fill(closed_angles, closed, alpha=0.10)
    ax.set_xticks(angles, labels)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_title("ASLA-P1 vs MLA — normalized trade-off (ratio-to-best)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate(rows: list[dict]) -> None:
    by_arch = {row["arch"]: row for row in rows}
    if set(by_arch) != {"ASLA-P1", "MLA"}:
        raise SystemExit("Expected exactly ASLA-P1 and MLA rows")
    if int(by_arch["ASLA-P1"]["n_seeds"]) != 3 or int(by_arch["MLA"]["n_seeds"]) != 3:
        raise SystemExit("Frozen result set is expected to contain 3 paired seeds per architecture")
    p_asla = float(by_arch["ASLA-P1"]["total_params_mean"])
    p_mla = float(by_arch["MLA"]["total_params_mean"])
    delta_pct = abs(p_asla - p_mla) / p_asla * 100
    if delta_pct > 0.001:
        raise SystemExit(f"Parameter parity check failed: delta={delta_pct:.6f}%")
    for row in rows:
        radar = [float(v) for k, v in row.items() if k.startswith("radar_")]
        if not radar or any(not (0.0 <= v <= 1.000001) for v in radar):
            raise SystemExit(f"Invalid radar score for {row['arch']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild summary CSV and radar figure from frozen raw benchmark data.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw")
    parser.add_argument("--results-csv", type=Path, default=ROOT / "results.csv")
    parser.add_argument("--radar", type=Path, default=ROOT / "assets" / "radar_tradeoff.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    runs = read_csv(args.raw_dir / "runs.csv")
    prefill = read_csv(args.raw_dir / "attention_prefill_scaling.csv")
    summary = summarize_runs(runs, prefill)
    validate(summary)
    write_csv(args.results_csv, summary)
    if not args.no_plot:
        plot_radar(summary, args.radar)

    print(f"wrote {args.results_csv}")
    if not args.no_plot:
        print(f"wrote {args.radar}")


if __name__ == "__main__":
    main()
