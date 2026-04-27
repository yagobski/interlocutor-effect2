"""Bootstrap 95% CIs for Cohen's d (paired) and Holm-Bonferroni correction.

Reads each model's L0-clean per-scenario means (after H gate (0.10, 0.90))
and reports d, 95% bootstrap CI, raw and Holm-adjusted p, on the four
pre-registered open-weight models.

Usage: python paper_ie/experiments/run_bootstrap_ci.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "l0_clean"

MODELS = [
    "llama-3.1-8b",
    "mistral-7b",
    "qwen2.5-1.5b",
    "qwen2.5-7b",
]

H_LO, H_HI = 0.10, 0.90
B = 10000
RNG = np.random.default_rng(42)


def per_scenario_means(rows):
    """Aggregate H/A leakage per scenario across seeds & paraphrases."""
    h_acc, a_acc = defaultdict(list), defaultdict(list)
    for r in rows:
        sid = r["scenario_id"]
        cond = r.get("condition") or r.get("frame")
        leak = r.get("leak", r.get("leak_ratio"))
        if cond in ("L0_HUMAN", "H", "human"):
            h_acc[sid].append(leak)
        elif cond in ("L0_AGENT", "A", "agent"):
            a_acc[sid].append(leak)
    sids = sorted(set(h_acc) & set(a_acc))
    H = np.array([np.mean(h_acc[s]) for s in sids])
    A = np.array([np.mean(a_acc[s]) for s in sids])
    return sids, H, A


def cohens_d_paired(d):
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def bootstrap_ci_d(diffs, B=B, alpha=0.05, rng=RNG):
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(B, n))
    samples = diffs[idx]
    sd = samples.std(axis=1, ddof=1)
    sd[sd == 0] = np.nan
    ds = samples.mean(axis=1) / sd
    lo, hi = np.nanpercentile(ds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        adj_p = (m - rank) * pvals[idx]
        running = max(running, adj_p)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    out_rows = []
    for tag in MODELS:
        path = RESULTS / tag / "l0_clean_raw.json"
        if not path.exists():
            print(f"[skip] {tag}: missing {path}")
            continue
        rows = json.loads(path.read_text())
        sids, H, A = per_scenario_means(rows)
        gate = (H > H_LO) & (H < H_HI)
        Hg, Ag = H[gate], A[gate]
        diffs = Ag - Hg
        n = len(diffs)
        d = cohens_d_paired(diffs)
        ci_lo, ci_hi = bootstrap_ci_d(diffs)
        if n >= 2 and diffs.std(ddof=1) > 0:
            stat = wilcoxon(diffs, alternative="greater", zero_method="wilcox")
            p = float(stat.pvalue)
        else:
            p = float("nan")
        out_rows.append(
            dict(
                model=tag,
                n=n,
                mean_delta=float(diffs.mean()),
                d=d,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                p_one=p,
            )
        )
    pvals = np.array([r["p_one"] for r in out_rows])
    p_holm = holm(pvals)
    print(
        "\n L0-clean (gate H in (0.10, 0.90))  —  bootstrap d CIs (B=10000) + Holm correction over 4 models\n"
    )
    print(
        f"{'model':<14} {'n':>3}  {'meanΔ':>7}  {'d':>6}  {'95% CI':>16}  {'p_one':>10}  {'p_Holm':>10}"
    )
    for r, ph in zip(out_rows, p_holm):
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(
            f"{r['model']:<14} {r['n']:>3}  {r['mean_delta']:+.4f}  {r['d']:+.2f}  {ci:>16}  {r['p_one']:.3e}  {float(ph):.3e}"
        )

    # Stouffer
    zs = np.array([math.copysign(1, r["d"]) * abs(_p_to_z(r["p_one"])) for r in out_rows])
    z_st = zs.sum() / math.sqrt(len(zs))
    from scipy.stats import norm

    p_st = 1 - norm.cdf(z_st)
    print(
        f"\n Stouffer (4 models): z = {z_st:+.2f}, one-sided p = {p_st:.3e}"
    )

    out_path = RESULTS.parent / "stats" / "l0_bootstrap_ci.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            dict(rows=out_rows, p_holm=p_holm.tolist(), stouffer_z=z_st, stouffer_p=p_st),
            indent=2,
        )
    )
    print(f"\n wrote {out_path}")


def _p_to_z(p):
    from scipy.stats import norm

    p = max(min(p, 1 - 1e-15), 1e-15)
    return float(norm.ppf(1 - p))


if __name__ == "__main__":
    main()
