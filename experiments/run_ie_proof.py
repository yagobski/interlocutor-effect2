#!/usr/bin/env python3
"""
run_ie_proof.py — ONE-COMMAND reproducibility entry point for the IE proof.

Runs both the main L0-clean test and the phrasing-neutral control for a
single model, then prints a one-page summary. This is the script a reviewer
(or the user) launches to verify the main claim end-to-end.

Usage:
    python run_ie_proof.py --model qwen2.5-1.5b    # one model (≈ 20-40 min)
    python run_ie_proof.py --all                   # all four (≈ 4-6 h)
    python run_ie_proof.py --report-only           # stats on existing results

The pre-registration must already exist (see run_l0_clean.py --calibrate).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import PROJECT_ROOT

HERE = Path(__file__).parent
PY = sys.executable

OPEN_WEIGHT_TAGS = ("llama-3.1-8b", "mistral-7b", "qwen2.5-1.5b", "qwen2.5-7b")
RESULTS_ROOT = Path(PROJECT_ROOT) / "results"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"[FAIL] command exited {r.returncode}")


def run_model(tag: str) -> None:
    _run([PY, "run_l0_clean.py", "--run", "--model", tag])
    _run([PY, "run_l0_neutral.py", "--model", tag])


def _stats_for(raw_path: Path, hr_lo: float | None = None, hr_hi: float | None = None) -> dict | None:
    """Compute paired-scenario stats from raw rows.

    If hr_lo/hr_hi are given, apply the pre-registered headroom gate
    (mean H-frame leak ratio strictly inside (hr_lo, hr_hi)). This gate
    matches the paper's Table 1 row counts and Stouffer combine.
    """
    from scipy.stats import wilcoxon
    import numpy as np
    if not raw_path.exists():
        return None
    rows = json.loads(raw_path.read_text())
    per_scn: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"H": [], "A": []})
    for r in rows:
        if r["n_sensitive"] == 0:
            continue
        per_scn[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])
    deltas = []
    pos = neg = 0
    for sid, d in per_scn.items():
        if d["H"] and d["A"]:
            a = sum(d["A"]) / len(d["A"])
            h = sum(d["H"]) / len(d["H"])
            if hr_lo is not None and hr_hi is not None:
                if not (hr_lo < h < hr_hi):
                    continue
            delta = a - h
            deltas.append(delta)
            if delta > 0: pos += 1
            elif delta < 0: neg += 1
    if not deltas:
        return None
    _, p = wilcoxon(deltas, alternative="greater")
    sd = np.std(deltas, ddof=1)
    d = (np.mean(deltas) / sd) if sd > 0 else 0.0
    return dict(n=len(deltas), pos=pos, neg=neg, mean=float(np.mean(deltas)), p=float(p), d=float(d))


def report() -> None:
    from scipy.special import ndtri, ndtr
    import math

    print("\n" + "=" * 70)
    print("IE PROOF — one-page report")
    print("=" * 70)

    # (experiment_name, subdir, raw_filename, gate_lo, gate_hi)
    # L0-clean: pre-registered headroom gate H in (0.10, 0.90) — matches paper Table 1.
    # L0-neutral: ungated (paper Appendix table reports n=51 per model).
    experiments = [
        ("L0-clean (main: varied phrasing) — gated H∈(0.10,0.90)", "l0_clean", "l0_clean_raw.json", 0.10, 0.90),
        ("L0-neutral (control: identical phrasing) — ungated", "l0_neutral", "l0_neutral_raw.json", None, None),
    ]
    for experiment, subdir, fname, lo, hi in experiments:
        print(f"\n{experiment}")
        print("-" * 70)
        print(f"  {'model':<16} {'n':>3} {'A>H':>5} {'A<H':>5} {'meanΔ':>8} {'d':>6} {'p':>10}")
        per_p = []
        for tag in OPEN_WEIGHT_TAGS:
            st = _stats_for(RESULTS_ROOT / subdir / tag / fname, lo, hi)
            if st is None:
                print(f"  {tag:<16} {'—':>3}")
                continue
            print(f"  {tag:<16} {st['n']:>3} {st['pos']:>5} {st['neg']:>5} "
                  f"{st['mean']:>+8.4f} {st['d']:>+6.2f} {st['p']:>10.2e}")
            per_p.append(st["p"])
        if len(per_p) >= 2:
            zs = [ndtri(1 - pv) for pv in per_p]
            sz = sum(zs) / math.sqrt(len(zs))
            sp = 1 - ndtr(sz)
            print(f"  {'Stouffer':<16} {'':>3} {'':>5} {'':>5} {'':>8} z={sz:>+5.2f} p={sp:>10.2e}")

    print("\nInterpretation key:")
    print("  * L0-clean = main IE existence test (51 scenarios, 5 seeds, 5 paraphrases).")
    print("  * L0-neutral = same scenarios, ONE identical user request for H and A.")
    print("    If the effect survives L0-neutral with d>0.3 and p<0.01, the IE is")
    print("    attributable to the system-prompt identity contrast alone, not to")
    print("    co-varying user-side phrasing.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=OPEN_WEIGHT_TAGS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report()
    elif args.all:
        for tag in OPEN_WEIGHT_TAGS:
            run_model(tag)
        report()
    elif args.model:
        run_model(args.model)
        report()
    else:
        ap.print_help()
