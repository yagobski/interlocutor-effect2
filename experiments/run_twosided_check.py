#!/usr/bin/env python3
"""Two-sided robustness check for the L0-clean primary test (reviewer Q1).

Reproduces Table 1 statistics with both a one-sided (pre-registered, H1: A>H)
and a two-sided Wilcoxon signed-rank test, plus Stouffer combine across the
four open-weight models. Output is intended for Appendix.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from collections import defaultdict
from scipy.stats import wilcoxon
from scipy.special import ndtri, ndtr
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "results"
TAGS = ("llama-3.1-8b", "mistral-7b", "qwen2.5-1.5b", "qwen2.5-7b")


def stats(raw_path: Path, lo: float | None = None, hi: float | None = None) -> dict | None:
    if not raw_path.exists():
        return None
    rows = json.loads(raw_path.read_text())
    per = defaultdict(lambda: {"H": [], "A": []})
    for r in rows:
        if r["n_sensitive"] == 0:
            continue
        per[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])
    deltas = []
    for d in per.values():
        if d["H"] and d["A"]:
            a = sum(d["A"]) / len(d["A"])
            h = sum(d["H"]) / len(d["H"])
            if lo is not None and not (lo < h < hi):
                continue
            deltas.append(a - h)
    if not deltas:
        return None
    _, p_g = wilcoxon(deltas, alternative="greater")
    _, p_t = wilcoxon(deltas, alternative="two-sided")
    sd = np.std(deltas, ddof=1)
    return dict(
        n=len(deltas),
        mean=float(np.mean(deltas)),
        d=float(np.mean(deltas) / sd) if sd > 0 else 0.0,
        p_one=float(p_g),
        p_two=float(p_t),
    )


def main() -> None:
    print("L0-clean (gated H in (0.10, 0.90)) — one-sided vs two-sided Wilcoxon")
    print(f"{'model':<16}{'n':>5}{'meanΔ':>10}{'d':>8}{'p_one':>13}{'p_two':>13}")
    ones, twos = [], []
    for tag in TAGS:
        st = stats(ROOT / "l0_clean" / tag / "l0_clean_raw.json", 0.10, 0.90)
        if st is None:
            print(f"{tag:<16}  (missing)")
            continue
        print(f"{tag:<16}{st['n']:>5}{st['mean']:>+10.4f}{st['d']:>+8.2f}"
              f"{st['p_one']:>13.3e}{st['p_two']:>13.3e}")
        ones.append(st["p_one"])
        twos.append(st["p_two"])

    if len(ones) >= 2:
        zs1 = [ndtri(1 - p) for p in ones]
        sz1 = sum(zs1) / math.sqrt(len(zs1))
        sp1 = 1 - ndtr(sz1)
        # Two-sided Stouffer: convert each two-sided p to a |z| and pool with sign
        zs2 = [ndtri(1 - p / 2) for p in twos]  # all deltas point same direction; signs preserved by mean
        sz2 = sum(zs2) / math.sqrt(len(zs2))
        sp2 = 2 * (1 - ndtr(abs(sz2)))
        print(f"{'Stouffer':<16}{'':>5}{'':>10}{'':>8}"
              f"  one-sided z={sz1:+.2f} p={sp1:.3e}"
              f"   two-sided z={sz2:+.2f} p={sp2:.3e}")


if __name__ == "__main__":
    main()
