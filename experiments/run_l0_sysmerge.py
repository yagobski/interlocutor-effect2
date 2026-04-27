#!/usr/bin/env python3
"""
run_l0_sysmerge.py — Causal ablation of the system-role channel.

Motivation
----------
Gemma-2-9B yielded a null result on the L0 minimal-pair test (d=-0.02, p=0.57),
in contrast to strong positive effects on Llama-3.1-8B, Mistral-7B, and
Qwen2.5-1.5B. Gemma's chat template does not support a `system` role, so the
identity contrast (HUMAN vs AGENT) had to be delivered inside the user turn.

This script runs the SAME L0 minimal-pair protocol as run_l0_clean.py, but
FORCES the system-merge format (identity injected into user turn) on models
that normally support the system role. If the IE collapses under this format
on Llama / Mistral, it causally implicates the system channel — independent
of architecture or training.

Design: Identical to run_l0_clean.py (same 51 test scenarios, 5 seeds, 5
paraphrases, 2 frames), only the chat-template format differs.

Output: results/l0_sysmerge/<model>/l0_sysmerge_raw.json

Usage:
    python run_l0_sysmerge.py --run --model llama-3.1-8b
    python run_l0_sysmerge.py --run --model mistral-7b
    python run_l0_sysmerge.py --aggregate
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from config import TEMPERATURE, PROJECT_ROOT, get_model_by_tag
from scenarios_l0_clean import (
    SCENARIOS_TEST,
    PARAPHRASES,
    L0_HUMAN,
    L0_AGENT,
)

_orig_argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_behavioral as rb  # noqa: E402
sys.argv = _orig_argv

RESULTS_DIR = Path(PROJECT_ROOT) / "results" / "l0_sysmerge"
L0_SEEDS = [42, 142, 242, 342, 442]
ABLATION_TAGS = ("llama-3.1-8b", "mistral-7b", "qwen2.5-1.5b", "qwen2.5-7b")

# ─────────────────────────────────────────────────────────────────────────────
# FORCED SYSTEM-MERGE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_local_sysmerge(system: str, user: str, temperature=0.0, seed=42):
    """Same as rb.generate_local, but ALWAYS merges system into user turn."""
    import torch
    messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
    text = rb._tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = rb._tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    ).to(rb._device)
    torch.manual_seed(seed)
    if rb._device == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    elif rb._device == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        with torch.no_grad():
            out = rb._model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.9 if temperature > 0 else None,
                pad_token_id=rb._tokenizer.eos_token_id,
            )
        return rb._tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
    except Exception as e:
        print(f"    [GEN ERROR] {e}")
        return None


def _run_scenario(scenario: dict, seeds: list[int]) -> list[dict]:
    vault_json = json.dumps(scenario["vault"], indent=2, ensure_ascii=False)
    rows: list[dict] = []
    for seed in seeds:
        for para_idx, (human_req, agent_req) in enumerate(PARAPHRASES):
            full_sys_h = f"{L0_HUMAN}\n\nDATA CONTEXT:\n{vault_json}"
            full_sys_a = f"{L0_AGENT}\n\nDATA CONTEXT:\n{vault_json}"
            for frame, sys_prompt, user_req in [
                ("H", full_sys_h, human_req),
                ("A", full_sys_a, agent_req),
            ]:
                resp = generate_local_sysmerge(sys_prompt, user_req, TEMPERATURE, seed)
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
# RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_test(model_tag: str) -> None:
    model_cfg = get_model_by_tag(model_tag)
    if model_cfg["mode"] != "local":
        raise SystemExit(f"sysmerge ablation is only meaningful for local models (got {model_tag})")
    out_dir = RESULTS_DIR / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rb.load_local_model(model_cfg["id"])
    print(f"[sysmerge] model={model_tag}  n_scenarios={len(SCENARIOS_TEST)}  (system merged into user turn)")

    rows: list[dict] = []
    for i, sc in enumerate(SCENARIOS_TEST):
        print(f"  [{i+1}/{len(SCENARIOS_TEST)}] {sc['id']} ({sc['vertical']} T{sc['tier']})")
        rows.extend(_run_scenario(sc, L0_SEEDS))
        # checkpoint every 10 scenarios
        if (i + 1) % 10 == 0:
            (out_dir / "l0_sysmerge_raw.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False)
            )

    rb.unload_model()
    (out_dir / "l0_sysmerge_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    print(f"[sysmerge] wrote {len(rows)} rows to {out_dir}/l0_sysmerge_raw.json")


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate() -> None:
    import numpy as np
    from scipy.stats import wilcoxon, norm

    # Apply same pre-registered headroom filter: H in (0.10, 0.90)
    HR_LO, HR_HI = 0.10, 0.90

    # Also try to compare to the original (non-merged) l0_clean results.
    clean_dir = Path(PROJECT_ROOT) / "results" / "l0_clean"

    print(f"{'model':<16}{'n_raw':>7}{'n_filt':>8}{'Δ(A-H)':>10}{'d':>8}{'p':>12}{'vs_clean_d':>14}")
    print("-" * 80)

    all_zs: list[float] = []
    models_with_data: list[str] = []
    _per_model_report: dict = {}

    for tag in ABLATION_TAGS:
        f = RESULTS_DIR / tag / "l0_sysmerge_raw.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        per_scn = collections.defaultdict(lambda: {"H": [], "A": []})
        for r in rows:
            if r["n_sensitive"] == 0:
                continue
            per_scn[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])

        scn_deltas: list[float] = []
        scn_h: list[float] = []
        for sid, d in per_scn.items():
            if not d["H"] or not d["A"]:
                continue
            h = float(np.mean(d["H"]))
            a = float(np.mean(d["A"]))
            scn_deltas.append(a - h)
            scn_h.append(h)

        # Filter by headroom
        keep = [(de, hh) for de, hh in zip(scn_deltas, scn_h) if HR_LO < hh < HR_HI]
        n_raw = len(scn_deltas)
        n_filt = len(keep)
        if n_filt < 5:
            print(f"{tag:<16}{n_raw:>7}{n_filt:>8}  insufficient data after filter")
            continue

        deltas = np.array([d for d, _ in keep])
        mean_delta = float(deltas.mean())
        cohen_d = mean_delta / float(deltas.std(ddof=1)) if deltas.std(ddof=1) > 0 else 0.0
        try:
            stat, p = wilcoxon(deltas, alternative="greater")
        except ValueError:
            stat, p = float("nan"), 1.0

        # Compare to l0_clean Cohen's d
        clean_d_str = "n/a"
        clean_file = clean_dir / tag / "l0_clean_raw.json"
        if clean_file.exists():
            c_rows = json.loads(clean_file.read_text())
            c_scn = collections.defaultdict(lambda: {"H": [], "A": []})
            for r in c_rows:
                if r["n_sensitive"] == 0:
                    continue
                c_scn[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])
            c_deltas = []
            for sid, d in c_scn.items():
                if not d["H"] or not d["A"]:
                    continue
                h = float(np.mean(d["H"])); a = float(np.mean(d["A"]))
                if HR_LO < h < HR_HI:
                    c_deltas.append(a - h)
            if len(c_deltas) >= 5:
                ca = np.array(c_deltas)
                cd = float(ca.mean() / ca.std(ddof=1)) if ca.std(ddof=1) > 0 else 0.0
                clean_d_str = f"{cd:+.3f}"

        # Stouffer contribution: signed z from one-sided p
        z_one = norm.isf(max(p, 1e-300))
        all_zs.append(z_one)
        models_with_data.append(tag)
        _per_model_report[tag] = {
            "n_raw": int(n_raw),
            "n": int(n_filt),
            "mean_delta": float(mean_delta),
            "cohen_d": float(cohen_d),
            "wilcoxon_p1": float(p),
        }

        print(f"{tag:<16}{n_raw:>7}{n_filt:>8}{mean_delta:>+10.3f}{cohen_d:>+8.3f}{p:>12.3e}{clean_d_str:>14}")

    if all_zs:
        z_stouffer = float(np.sum(all_zs) / np.sqrt(len(all_zs)))
        p_stouffer = float(norm.sf(z_stouffer))
        print("-" * 80)
        print(f"Stouffer (sysmerge, {len(all_zs)} models): z={z_stouffer:+.3f}  p={p_stouffer:.3e}")
        # Persist a self-describing report so downstream tools can read it.
        report = {
            "headroom_window": [HR_LO, HR_HI],
            "per_model": _per_model_report,
            "stouffer": {"z": z_stouffer, "p": p_stouffer, "n_models": len(all_zs)},
        }
        (RESULTS_DIR / "l0_sysmerge_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        print(f"  → {RESULTS_DIR / 'l0_sysmerge_report.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    if args.run:
        if not args.model:
            raise SystemExit("--run requires --model")
        run_test(args.model)
    elif args.aggregate:
        aggregate()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
