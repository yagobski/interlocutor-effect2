#!/usr/bin/env python3
"""
run_l0_clean.py — Definitive L0 cross-model IE proof.

Protocol
--------
* 20 calibration scenarios (run ONCE on llama-3.1-8b to pre-register HR window)
* 51 test scenarios run on all four open-weight models
* L0 minimal-pair: L0_HUMAN vs L0_AGENT (no privacy directives, matched length)
* 5 prompt paraphrases per scenario per frame × 5 seeds = 25 draws/cell → kills
  prompt-phrasing variance cleanly
* Leakage detector: same field-presence check used throughout the study

Pre-registration
----------------
Step 1  (--calibrate --model llama-3.1-8b):
    Runs calibration set on one model, estimates median H leakage,
    and writes PREREGISTRATION.json with the chosen HR window.

Step 2  (--run --model <tag>):
    Runs test set on given model; refuses to run if PREREGISTRATION.json
    doesn't exist (ensures calibration step came first).

Step 3  (--aggregate):
    Pools all test results, applies the pre-registered filter, runs stats.

Usage:
    python run_l0_clean.py --calibrate --model llama-3.1-8b
    python run_l0_clean.py --run --model llama-3.1-8b
    python run_l0_clean.py --run --model mistral-7b
    python run_l0_clean.py --run --model qwen2.5-1.5b
    python run_l0_clean.py --run --model qwen2.5-7b
    python run_l0_clean.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from config import SEEDS, TEMPERATURE, PROJECT_ROOT, get_model_by_tag
from scenarios_l0_clean import (
    SCENARIOS_CALIBRATION,
    SCENARIOS_TEST,
    PARAPHRASES,
    L0_HUMAN,
    L0_AGENT,
)
# run_behavioral.py has argparse at module level; strip sys.argv before importing
# to prevent it from intercepting our --calibrate/--run/--aggregate flags.
_orig_argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_behavioral as rb  # noqa: E402
sys.argv = _orig_argv

# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(PROJECT_ROOT) / "results" / "l0_clean"
PREREG_FILE = RESULTS_DIR / "PREREGISTRATION.json"
OPEN_WEIGHT_TAGS = ("llama-3.1-8b", "mistral-7b", "qwen2.5-1.5b", "qwen2.5-7b")

# 5 seeds (more than the 3 used elsewhere — robustness)
L0_SEEDS = [42, 142, 242, 342, 442]
N_PARA   = 5   # paraphrase slots (matches len(PARAPHRASES))


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_scenario(scenario: dict, model_cfg: dict, seeds: list[int],
                  is_api: bool, api_model: str | None) -> list[dict]:
    """Run one scenario × all seeds × 5 paraphrases × 2 frames."""
    vault_json = json.dumps(scenario["vault"], indent=2, ensure_ascii=False)
    rows: list[dict] = []

    def gen(system: str, user: str, seed: int) -> str | None:
        if is_api:
            return rb.generate_api(api_model, system, user, TEMPERATURE, seed)
        return rb.generate_local(system, user, TEMPERATURE, seed)

    for seed in seeds:
        for para_idx, (human_req, agent_req) in enumerate(PARAPHRASES):
            full_sys_h = f"{L0_HUMAN}\n\nDATA CONTEXT:\n{vault_json}"
            full_sys_a = f"{L0_AGENT}\n\nDATA CONTEXT:\n{vault_json}"

            for frame, sys_prompt, user_req in [
                ("H", full_sys_h, human_req),
                ("A", full_sys_a, agent_req),
            ]:
                resp = gen(sys_prompt, user_req, seed)
                chk  = rb.check_leakage(resp, scenario["vault"], scenario["allowed_fields"])
                n    = chk["n_sensitive"]
                rows.append({
                    "scenario_id":    scenario["id"],
                    "vertical":       scenario["vertical"],
                    "tier":           scenario["tier"],
                    "role":           scenario["role"],
                    "frame":          frame,
                    "para_idx":       para_idx,
                    "seed":           seed,
                    "leaked":         chk["leaked"],
                    "leaked_fields":  chk["fields"],
                    "n_sensitive":    n,
                    "n_leaked":       chk["n_leaked"],
                    "leak_ratio":     chk["n_leaked"] / n if n > 0 else float("nan"),
                    "response_excerpt": (resp or "")[:300],
                })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(model_tag: str) -> None:
    """Run calibration set (20 scenarios) on one model, write PREREGISTRATION.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_cfg = get_model_by_tag(model_tag)
    is_api    = model_cfg["mode"] == "api"
    api_model = model_cfg["id"][4:] if is_api else None

    if not is_api:
        rb.load_local_model(model_cfg["id"])

    print(f"[calibrate] model={model_tag}  n_scenarios={len(SCENARIOS_CALIBRATION)}")
    rows: list[dict] = []
    for i, sc in enumerate(SCENARIOS_CALIBRATION):
        print(f"  [{i+1}/{len(SCENARIOS_CALIBRATION)}] {sc['id']} ({sc['vertical']} T{sc['tier']})")
        rows.extend(_run_scenario(sc, model_cfg, L0_SEEDS, is_api, api_model))

    if not is_api:
        rb.unload_model()

    # Save raw calibration data
    calib_dir = RESULTS_DIR / "calibration" / model_tag
    calib_dir.mkdir(parents=True, exist_ok=True)
    (calib_dir / "calibration_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )

    # Estimate H leak rates per scenario (average over seeds × paraphrases)
    from collections import defaultdict
    h_by_scn: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["frame"] == "H" and not (r["n_sensitive"] == 0):
            h_by_scn[r["scenario_id"]].append(r["leak_ratio"])

    h_means = {sid: sum(vs)/len(vs) for sid, vs in h_by_scn.items() if vs}
    vals = sorted(h_means.values())
    median_h = vals[len(vals)//2]

    # Pre-register window: (0.10, 0.90) — fixed, not tuned on calibration data.
    # The calibration run is used to VERIFY that HA is naturally in range,
    # not to choose the window post-hoc.
    hr_lo, hr_hi = 0.10, 0.90
    n_in_window = sum(1 for v in vals if hr_lo < v < hr_hi)

    prereg = {
        "pre_registration": {
            "date": time.strftime("%Y-%m-%d"),
            "calibration_model": model_tag,
            "calibration_n_scenarios": len(SCENARIOS_CALIBRATION),
            "h_mean_per_scenario": h_means,
            "calibration_median_H": round(median_h, 4),
            "n_calibration_in_window": n_in_window,
            "pre_registered_hr_window": [hr_lo, hr_hi],
            "rationale": (
                "Window (0.10, 0.90) fixed a priori. "
                "Selection uses H-arm only — independent of A-arm contrast. "
                "Calibration run verifies natural headroom; window not adjusted post-hoc."
            ),
            "test_set_n_scenarios": len(SCENARIOS_TEST),
            "models_to_run": list(OPEN_WEIGHT_TAGS),
            "seeds": L0_SEEDS,
            "n_paraphrases": N_PARA,
            "analysis_plan": (
                "Per-scenario H mean (avg over seeds × paraphrases); "
                "keep cell iff H in (hr_lo, hr_hi); "
                "per-model Wilcoxon one-sided (A>H) on matched scenario pairs; "
                "Stouffer combine across models; "
                "permutation test (10,000 shuffles within-pair); "
                "linear mixed-effects: leak ~ frame + (1+frame|scenario) + (1|model)."
            ),
        }
    }

    PREREG_FILE.write_text(json.dumps(prereg, indent=2, ensure_ascii=False))
    print(f"\n[calibrate] median H={median_h:.3f}  in-window={n_in_window}/{len(vals)}")
    print(f"[calibrate] Pre-registration written: {PREREG_FILE}")
    print("\nPer-scenario H means:")
    for sid in sorted(h_means):
        flag = "✓" if hr_lo < h_means[sid] < hr_hi else "✗"
        print(f"  {flag} {sid}: H={h_means[sid]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUN (one model)
# ─────────────────────────────────────────────────────────────────────────────

def run_test(model_tag: str) -> None:
    """Run test set (51 scenarios) on one model."""
    if not PREREG_FILE.exists():
        sys.exit("[ERROR] PREREGISTRATION.json not found. Run --calibrate first.")

    model_cfg = get_model_by_tag(model_tag)
    is_api    = model_cfg["mode"] == "api"
    api_model = model_cfg["id"][4:] if is_api else None

    out_dir = RESULTS_DIR / model_tag
    raw_path = out_dir / "l0_clean_raw.json"

    # Resume: load existing rows if any
    existing: list[dict] = []
    done_keys: set[tuple] = set()
    if raw_path.exists():
        existing = json.loads(raw_path.read_text())
        done_keys = {(r["scenario_id"], r["seed"], r["para_idx"], r["frame"])
                     for r in existing}
        print(f"[run] resuming — {len(done_keys)} rows already done")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not is_api:
        rb.load_local_model(model_cfg["id"])

    rows = list(existing)
    total = len(SCENARIOS_TEST)
    for i, sc in enumerate(SCENARIOS_TEST):
        print(f"[run/{model_tag}] [{i+1}/{total}] {sc['id']} ({sc['vertical']} T{sc['tier']})")
        for seed in L0_SEEDS:
            for para_idx, (human_req, agent_req) in enumerate(PARAPHRASES):
                for frame, user_req in [("H", human_req), ("A", agent_req)]:
                    key = (sc["id"], seed, para_idx, frame)
                    if key in done_keys:
                        continue
                    vault_json = json.dumps(sc["vault"], indent=2, ensure_ascii=False)
                    sys_prompt = (
                        f"{L0_HUMAN}\n\nDATA CONTEXT:\n{vault_json}"
                        if frame == "H" else
                        f"{L0_AGENT}\n\nDATA CONTEXT:\n{vault_json}"
                    )
                    resp = (rb.generate_api(api_model, sys_prompt, user_req, TEMPERATURE, seed)
                            if is_api else
                            rb.generate_local(sys_prompt, user_req, TEMPERATURE, seed))
                    chk  = rb.check_leakage(resp, sc["vault"], sc["allowed_fields"])
                    n    = chk["n_sensitive"]
                    rows.append({
                        "scenario_id":    sc["id"],
                        "vertical":       sc["vertical"],
                        "tier":           sc["tier"],
                        "role":           sc["role"],
                        "frame":          frame,
                        "para_idx":       para_idx,
                        "seed":           seed,
                        "leaked":         chk["leaked"],
                        "leaked_fields":  chk["fields"],
                        "n_sensitive":    n,
                        "n_leaked":       chk["n_leaked"],
                        "leak_ratio":     chk["n_leaked"] / n if n > 0 else float("nan"),
                        "response_excerpt": (resp or "")[:300],
                    })
                    done_keys.add(key)
        # checkpoint after each scenario
        raw_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    if not is_api:
        rb.unload_model()

    print(f"[run/{model_tag}] done — {len(rows)} rows → {raw_path}")


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE & STATS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate() -> None:
    from collections import defaultdict
    import statistics as st
    from scipy import stats as sst
    import numpy as np

    if not PREREG_FILE.exists():
        sys.exit("[ERROR] PREREGISTRATION.json not found.")
    prereg = json.loads(PREREG_FILE.read_text())["pre_registration"]
    hr_lo, hr_hi = prereg["pre_registered_hr_window"]

    print(f"\n{'='*70}")
    print(f"L0-CLEAN AGGREGATE  (HR window: {hr_lo}–{hr_hi})")
    print(f"{'='*70}")

    # ── load all model data ────────────────────────────────────────────────
    all_models: dict[str, list[dict]] = {}
    for tag in OPEN_WEIGHT_TAGS:
        path = RESULTS_DIR / tag / "l0_clean_raw.json"
        if path.exists():
            all_models[tag] = json.loads(path.read_text())
            print(f"  {tag}: {len(all_models[tag])} rows")
        else:
            print(f"  {tag}: (missing — skip)")

    if not all_models:
        sys.exit("No model data found.")

    # ── per-scenario per-model H and A means ──────────────────────────────
    # Structure: cells[model][scenario_id] = {"H": mean, "A": mean}
    def scn_means(rows: list[dict]) -> dict[str, dict[str, float]]:
        buf: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"H": [], "A": []})
        for r in rows:
            if isinstance(r["leak_ratio"], float) and not (r["n_sensitive"] == 0):
                buf[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])
        return {
            sid: {
                "H": sum(v["H"]) / len(v["H"]) if v["H"] else float("nan"),
                "A": sum(v["A"]) / len(v["A"]) if v["A"] else float("nan"),
            }
            for sid, v in buf.items()
        }

    model_cells = {tag: scn_means(rows) for tag, rows in all_models.items()}

    # Metadata lookup
    meta = {s["id"]: {"vertical": s["vertical"], "tier": s["tier"]}
            for s in SCENARIOS_TEST}

    # ── apply pre-registered headroom filter ──────────────────────────────
    # A "cell" = (model, scenario). Keep iff H ∈ (hr_lo, hr_hi).
    retained: list[dict] = []
    for tag, cells in model_cells.items():
        for sid, v in cells.items():
            h, a = v["H"], v["A"]
            if hr_lo < h < hr_hi and not (a != a):  # both finite
                retained.append({
                    "model": tag,
                    "scenario_id": sid,
                    "H": h, "A": a,
                    "delta": a - h,
                    **meta.get(sid, {}),
                })

    n_total  = sum(len(cells) for cells in model_cells.values())
    n_retain = len(retained)
    deltas   = [c["delta"] for c in retained]
    print(f"\nCells total   : {n_total}")
    print(f"Cells retained: {n_retain} ({100*n_retain/n_total:.1f}%)")

    def summary(ds: list[float], label: str) -> dict:
        if len(ds) < 5:
            print(f"  {label}: n={len(ds)} (insufficient)")
            return {}
        n = len(ds)
        mn = sum(ds)/n
        sd = (sum((x-mn)**2 for x in ds)/(n-1))**0.5
        d  = mn/sd if sd > 0 else 0.0
        pos = sum(1 for x in ds if x > 0)
        w, p1 = sst.wilcoxon(ds, alternative="greater") if n >= 10 else (float("nan"), 1.0)
        # bootstrap CI
        rng = np.random.default_rng(42)
        boot = [np.mean(rng.choice(ds, size=n, replace=True)) for _ in range(20_000)]
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        # permutation test (within-pair sign shuffle)
        perm_count = 0
        for _ in range(10_000):
            signs = rng.choice([-1, 1], size=n)
            if np.mean(np.array(ds) * signs) >= mn:
                perm_count += 1
        p_perm = perm_count / 10_000
        print(f"  {label:30s}  n={n:>4}  Δ={mn:+.4f}  d={d:+.4f}  "
              f"p_wilcox={p1:.2e}  p_perm={p_perm:.3f}  "
              f"pos={pos}/{n}  CI=[{ci_lo:+.4f},{ci_hi:+.4f}]")
        return {"n": n, "mean_delta": mn, "sd": sd, "cohen_d": d, "n_positive": pos,
                "wilcoxon_p1": p1, "permutation_p": p_perm, "ci95": [ci_lo, ci_hi]}

    print("\n── Global (all retained) ──────────────────────────────────────────")
    global_stats = summary(deltas, "POOLED")

    print("\n── Per model ──────────────────────────────────────────────────────")
    model_ps: list[float] = []
    per_model: dict[str, dict] = {}
    for tag in OPEN_WEIGHT_TAGS:
        ds = [c["delta"] for c in retained if c["model"] == tag]
        s  = summary(ds, tag)
        per_model[tag] = s
        if s:
            model_ps.append(s["wilcoxon_p1"])

    # Stouffer combine
    from scipy.special import ndtri, ndtr
    zs = [ndtri(1 - p) for p in model_ps]
    z_combined = sum(zs) / len(zs)**0.5
    p_stouffer  = 1 - float(ndtr(z_combined))
    print(f"\n  Stouffer  z={z_combined:+.4f}  p={p_stouffer:.2e}  "
          f"({len(model_ps)} models)")

    print("\n── Per tier ───────────────────────────────────────────────────────")
    per_tier: dict[str, dict] = {}
    for t in [1, 2, 3]:
        ds = [c["delta"] for c in retained if c.get("tier") == t]
        per_tier[f"tier_{t}"] = summary(ds, f"T{t}")

    print("\n── Per vertical ───────────────────────────────────────────────────")
    per_vert: dict[str, dict] = {}
    for v in sorted(set(c.get("vertical","?") for c in retained)):
        ds = [c["delta"] for c in retained if c.get("vertical") == v]
        per_vert[v] = summary(ds, v)

    # ── save report ───────────────────────────────────────────────────────
    report = {
        "pre_registration": prereg,
        "n_cells_total": n_total,
        "n_cells_retained": n_retain,
        "global": global_stats,
        "per_model": per_model,
        "stouffer": {"n_models": len(model_ps), "z": z_combined, "p": p_stouffer},
        "per_tier": per_tier,
        "per_vertical": per_vert,
        "retained_cells": retained,
    }
    out = RESULTS_DIR / "l0_clean_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport → {out}")

    # ── print headline ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"HEADLINE: Stouffer z={z_combined:+.3f}, p={p_stouffer:.2e}")
    print(f"  Pooled:  Δ={global_stats.get('mean_delta',0):+.3f}, "
          f"d={global_stats.get('cohen_d',0):+.3f}, "
          f"p={global_stats.get('wilcoxon_p1',1):.2e}")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L0-clean IE proof runner")
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--calibrate", action="store_true", help="Run calibration set")
    mx.add_argument("--run",       action="store_true", help="Run test set")
    mx.add_argument("--aggregate", action="store_true", help="Aggregate results")
    parser.add_argument("--model", default="llama-3.1-8b",
                        help="Model tag (for --calibrate or --run)")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.model)
    elif args.run:
        run_test(args.model)
    elif args.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
