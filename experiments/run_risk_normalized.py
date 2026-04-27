#!/usr/bin/env python3
"""Risk-normalised effect sizes on the frontier benchmark (reviewer Q4).

For each model and each scenario s, define
    Δ_norm(s) = (A_s - H_s) / (1 - H_s),
the fraction of the remaining safety headroom captured by agent framing.
Only scenarios with H_s < 1 contribute (saturated cells are undefined).

Loads paper_ie/results/frontier_iwpe/benchmark_raw.json and prints a small
table aligned with Appendix~\\ref{app:risk_norm}.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "results" / "frontier_iwpe" / "benchmark_raw.json"

KEEP = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-large-2411",
]
LABELS = {
    "anthropic/claude-3.5-sonnet":     "Claude-3.5-Sonnet",
    "openai/gpt-4o":                   "GPT-4o",
    "meta-llama/llama-3.3-70b-instruct": "Llama-3.3-70B",
    "mistralai/mistral-large-2411":    "Mistral-Large",
}


def main() -> None:
    rows_all = json.loads(RAW.read_text())
    print(f"{'Model':<22} {'n_sc':>5} {'H':>7} {'A':>7} {'Δ(pp)':>8}"
          f" {'Δ_norm':>10} {'n_unsat':>8}")
    for mdl in KEEP:
        rows = [r for r in rows_all if r["model"] == mdl]
        per: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"H": [], "A": []})
        for r in rows:
            side = "H" if r["cond"] in ("C_HT", "C_HJ") else "A"
            per[r["id"]][side].append(int(r["leaked"]))
        H_means, A_means, norm = [], [], []
        for hd in per.values():
            if not hd["H"] or not hd["A"]:
                continue
            h = float(np.mean(hd["H"]))
            a = float(np.mean(hd["A"]))
            H_means.append(h)
            A_means.append(a)
            if h < 1.0:
                norm.append((a - h) / (1.0 - h))
        H = float(np.mean(H_means))
        A = float(np.mean(A_means))
        print(f"{LABELS[mdl]:<22} {len(H_means):>5} {H:>7.3f} {A:>7.3f}"
              f" {(A - H) * 100:>+8.1f} {float(np.mean(norm)):>+10.3f}"
              f" {len(norm):>8}")


if __name__ == "__main__":
    main()
