from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ASLA_THEORETICAL_CACHE, MLA_THEORETICAL_CACHE
from src.models import TinyLM, attention_parameters, count_parameters
from src.reporting import read_csv, summarize_runs


class RepoTests(unittest.TestCase):
    def test_exact_parameter_counts(self):
        asla = TinyLM("ASLA-P1")
        mla = TinyLM("MLA")
        self.assertEqual(count_parameters(asla), 9_278_304)
        self.assertEqual(count_parameters(mla), 9_278_328)
        self.assertEqual(attention_parameters(asla), 124_544)
        self.assertEqual(attention_parameters(mla), 124_550)

    def test_parameter_parity(self):
        a, b = 9_278_304, 9_278_328
        delta_pct = abs(a - b) / a * 100
        self.assertLess(delta_pct, 0.001)
        self.assertAlmostEqual(delta_pct, 0.00025866796345539014, places=12)

    def test_cache_accounting(self):
        self.assertEqual(ASLA_THEORETICAL_CACHE, 44)
        self.assertEqual(MLA_THEORETICAL_CACHE, 93)

    def test_frozen_summary_and_radar_scores(self):
        raw = ROOT / "results" / "raw"
        rows = summarize_runs(
            read_csv(raw / "runs.csv"),
            read_csv(raw / "attention_prefill_scaling.csv"),
        )
        by_arch = {r["arch"]: r for r in rows}
        self.assertEqual(set(by_arch), {"ASLA-P1", "MLA"})
        self.assertEqual(by_arch["ASLA-P1"]["n_seeds"], 3)
        self.assertEqual(by_arch["MLA"]["n_seeds"], 3)
        self.assertAlmostEqual(float(by_arch["ASLA-P1"]["eval_loss_mean"]), 8.805346250534058)
        self.assertAlmostEqual(float(by_arch["MLA"]["eval_loss_mean"]), 8.95602031548818)
        for row in rows:
            scores = [float(v) for k, v in row.items() if k.startswith("radar_")]
            self.assertTrue(scores)
            self.assertTrue(all(0.0 <= x <= 1.0 + 1e-9 for x in scores))

    def test_prefill_latency_matches_frozen_data(self):
        raw = ROOT / "results" / "raw"
        rows = summarize_runs(
            read_csv(raw / "runs.csv"),
            read_csv(raw / "attention_prefill_scaling.csv"),
        )
        by_arch = {r["arch"]: r for r in rows}
        self.assertAlmostEqual(float(by_arch["ASLA-P1"]["prefill_latency_ms_p50_seq2048"]), 13.155012999959581)
        self.assertAlmostEqual(float(by_arch["MLA"]["prefill_latency_ms_p50_seq2048"]), 4.346855000221694)


if __name__ == "__main__":
    unittest.main()
