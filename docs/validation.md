# Validation

Validation for this packaged revision was run on 2026-08-21 with:

```bash
python -m compileall -q src scripts tests
PYTHONPATH=. python -m unittest discover -s tests -v
python scripts/reproduce.py
```

Result: `5/5` unit tests passed, `compileall` completed without errors, and the reproduce script regenerated both `results.csv` and `assets/radar_tradeoff.png`.

The test suite checks:

- total and per-layer attention parameter counts;
- the reported parameter-parity delta;
- analytical ASLA/MLA cache accounting;
- frozen summary values and radar score ranges;
- the frozen 2K prefill latency values.

A separate documentation check also compared the headline numbers against `results.csv` and verified local Markdown links.

A CPU smoke benchmark is useful after model-code changes because it exercises forward/backward, the optimizer path, causal masking, padding invariance and report writing without requiring a GPU. No model code changed during this prose-only revision, so the existing benchmark code and frozen evidence were left intact.
