#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEQ_LEN
from src.data import COPY_MARK, KV_QUERY, load_cache
from src.models import TinyLM, attention_parameters, count_parameters


PROFILES = {
    "smoke": {"steps": 2, "batch": 1, "seq_len": 64, "eval_batches": 1, "seeds": [1234]},
    "full": {"steps": 2000, "batch": 8, "seq_len": 256, "eval_batches": 8, "seeds": [1234, 2025, 3407]},
}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def batch_at(tokens: torch.Tensor, index: int, batch_size: int, seq_len: int, device: torch.device):
    start = (index * batch_size) % (len(tokens) - batch_size + 1)
    rows = tokens[start : start + batch_size, : seq_len + 1].to(device, non_blocking=True)
    return rows[:, :-1], rows[:, 1:]


def same_shape_state(source: TinyLM, target: TinyLM) -> None:
    """Copy identically named/typed common parameters for paired initialization."""
    src = source.state_dict()
    dst = target.state_dict()
    for key, value in src.items():
        if key in dst and dst[key].shape == value.shape:
            dst[key] = value.detach().clone()
    target.load_state_dict(dst)


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def aux_value(model: TinyLM) -> float:
    if model.arch != "ASLA-P1":
        return 0.0
    return float(torch.stack([block.attn.aux_loss.detach().float().cpu() for block in model.blocks]).mean())


def eval_metrics(model: TinyLM, tokens: torch.Tensor, batch_size: int, seq_len: int, batches: int, device: torch.device):
    model.eval()
    losses, correct, total = [], 0, 0
    copy_correct, copy_total = 0, 0
    retrieval_correct, retrieval_total = 0, 0
    with torch.inference_mode():
        for i in range(batches):
            x, y = batch_at(tokens, i, batch_size, seq_len, device)
            with autocast_context(device):
                loss, logits = model(x, y, include_aux=False)
            losses.append(float(loss.detach().float().cpu()))
            pred = logits.argmax(-1)
            correct += int((pred == y).sum().item())
            total += y.numel()

            # Repeated-token proxy used by the supplied source benchmark.
            repeated = (x[:, :, None] == x[:, None, :]).tril(-1).any(-1)
            repeated[:, -1] = False
            copy_correct += int(((pred == y) & repeated).sum().item())
            copy_total += int(repeated.sum().item())

            query_positions = x == KV_QUERY
            if query_positions.any():
                # The structured row layout places key then value after KV_QUERY.
                value_positions = torch.zeros_like(query_positions)
                value_positions[:, 2:] = query_positions[:, :-2]
                retrieval_correct += int(((pred == y) & value_positions).sum().item())
                retrieval_total += int(value_positions.sum().item())

    mean_loss = statistics.mean(losses)
    return {
        "eval_loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "next_token_accuracy": correct / max(total, 1),
        "copy_span_accuracy": copy_correct / max(copy_total, 1),
        "retrieval_accuracy": retrieval_correct / max(retrieval_total, 1),
    }


def correctness_checks(model: TinyLM, device: torch.device) -> dict:
    model.eval()
    with torch.inference_mode():
        probe = torch.randint(0, 32_000, (1, 16), device=device)
        mask = torch.tensor([[1] * 9 + [0] * 7], device=device)
        _, short = model(probe[:, :9])
        _, padded = model(probe, attention_mask=mask)
        pad_diff = float((short - padded[:, :9]).abs().max().cpu())

        _, base = model(probe[:, :9])
        changed = probe[:, :9].clone()
        # Alter a future token; positions before it must remain unchanged.
        changed[:, 8] = (changed[:, 8] + 7) % 32_000
        _, edited = model(changed)
        causal_diff = float((base[:, :8] - edited[:, :8]).abs().max().cpu())
    return {
        "padding_pass": pad_diff < 2e-4,
        "padding_max_diff": pad_diff,
        "causal_pass": causal_diff < 2e-4,
        "causal_max_diff": causal_diff,
    }


def train_one(initial: TinyLM, train_tokens: torch.Tensor, eval_tokens: torch.Tensor, cfg: dict, device: torch.device):
    model = deepcopy(initial).to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    optimizer_kwargs = {"lr": 3e-4, "weight_decay": 0.01}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    checks = correctness_checks(model, device)
    if not checks["padding_pass"] or not checks["causal_pass"]:
        raise RuntimeError(f"Correctness check failed for {model.arch}: {checks}")

    step_times: list[float] = []
    ce_history: list[float] = []
    model.train()
    for step in range(cfg["steps"]):
        x, y = batch_at(train_tokens, step, cfg["batch"], cfg["seq_len"], device)
        optimizer.zero_grad(set_to_none=True)
        cuda_sync(device)
        t0 = time.perf_counter()
        with autocast_context(device):
            loss, _ = model(x, y, include_aux=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        cuda_sync(device)
        step_times.append(time.perf_counter() - t0)
        ce_history.append(float(loss.detach().float().cpu()) - aux_value(model))

    evals = eval_metrics(
        model, eval_tokens, cfg["batch"], cfg["seq_len"], cfg["eval_batches"], device
    )
    tail = ce_history[max(0, int(len(ce_history) * 0.9)) :]
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    )
    mean_step = statistics.mean(step_times)
    out = {
        "arch": model.arch,
        "total_params": count_parameters(model),
        "attention_params_per_layer": attention_parameters(model),
        "train_tokens": cfg["steps"] * cfg["batch"] * cfg["seq_len"],
        "mean_step_seconds": mean_step,
        "p50_step_seconds": percentile(step_times, 0.50),
        "p95_step_seconds": percentile(step_times, 0.95),
        "train_throughput_tok_s": cfg["batch"] * cfg["seq_len"] / mean_step,
        "peak_vram_gib": peak,
        "tail_ce_std": statistics.stdev(tail) if len(tail) > 1 else 0.0,
        **evals,
        "correctness": checks,
    }
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def write_runs(path: Path, rows: list[dict]) -> None:
    fields = [
        "arch", "seed", "total_params", "attention_params_per_layer", "train_tokens",
        "mean_step_seconds", "p50_step_seconds", "p95_step_seconds",
        "train_throughput_tok_s", "peak_vram_gib", "tail_ce_std", "eval_loss",
        "perplexity", "next_token_accuracy", "copy_span_accuracy", "retrieval_accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired ASLA-P1 vs MLA proxy benchmark.")
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "realdata_cache.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "rerun")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()

    cfg = dict(PROFILES[args.profile])
    if args.steps is not None:
        cfg["steps"] = args.steps
    if args.seeds:
        cfg["seeds"] = args.seeds
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    cache = load_cache(args.cache)
    train_tokens = cache["train_tokens"]
    eval_tokens = cache["eval_tokens"]
    rows, checks = [], []

    for seed in cfg["seeds"]:
        torch.manual_seed(seed)
        asla = TinyLM("ASLA-P1")
        torch.manual_seed(seed + 1)
        mla = TinyLM("MLA")
        same_shape_state(asla, mla)
        initial = {"ASLA-P1": asla, "MLA": mla}

        for arch in ("ASLA-P1", "MLA"):
            result = train_one(initial[arch], train_tokens, eval_tokens, cfg, device)
            result["seed"] = seed
            checks.append({"seed": seed, "arch": arch, **result.pop("correctness")})
            rows.append(result)
            print(
                f"{arch} seed={seed} eval={result['eval_loss']:.4f} "
                f"throughput={result['train_throughput_tok_s']:.1f} tok/s"
            )

    write_runs(args.out_dir / "runs.csv", rows)
    report = {
        "profile": args.profile,
        "config": cfg,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "correctness": checks,
        "note": "Rerun implementation reconstructed from supplied source/report; frozen published numbers live in results/raw.",
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out_dir / 'runs.csv'}")


if __name__ == "__main__":
    main()
