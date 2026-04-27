#!/usr/bin/env python3
"""
run_l0_neutral.py — phrasing-neutral control for L0-clean.

Uses the same 51 test scenarios and L0_HUMAN/L0_AGENT system prompts, but
replaces the 5 paraphrase pairs (which differ between H and A) with ONE
identical neutral user request. This isolates the system-prompt identity
contrast from any user-phrasing confound.

Usage:
    python run_l0_neutral.py --model <tag>
    python run_l0_neutral.py --aggregate
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from config import TEMPERATURE, PROJECT_ROOT, get_model_by_tag
from scenarios_l0_neutral import (
    SCENARIOS_TEST, L0_HUMAN, L0_AGENT, NEUTRAL_REQUEST,
)

_orig_argv = sys.argv[:]
sys.argv = sys.argv[:1]
import run_behavioral as rb  # noqa: E402
sys.argv = _orig_argv

RESULTS_DIR = Path(PROJECT_ROOT) / "results" / "l0_neutral"
OPEN_WEIGHT_TAGS = ("llama-3.1-8b", "mistral-7b", "qwen2.5-1.5b", "qwen2.5-7b")
SEEDS = [42, 142, 242, 342, 442]


def run_model(tag: str) -> None:
    cfg = get_model_by_tag(tag)
    is_api = cfg["mode"] == "api"
    api_model = cfg["id"][4:] if is_api else None

    out_dir = RESULTS_DIR / tag
    raw_path = out_dir / "l0_neutral_raw.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    done_keys: set[tuple] = set()
    if raw_path.exists():
        existing = json.loads(raw_path.read_text())
        done_keys = {(r["scenario_id"], r["seed"], r["frame"]) for r in existing}
        print(f"[resume] {len(done_keys)} rows already done")

    if not is_api:
        rb.load_local_model(cfg["id"])

    rows = list(existing)
    for i, sc in enumerate(SCENARIOS_TEST):
        print(f"[{tag}] [{i+1}/{len(SCENARIOS_TEST)}] {sc['id']} ({sc['vertical']} T{sc['tier']})")
        vault_json = json.dumps(sc["vault"], indent=2, ensure_ascii=False)
        for seed in SEEDS:
            for frame, base_sys in [("H", L0_HUMAN), ("A", L0_AGENT)]:
                key = (sc["id"], seed, frame)
                if key in done_keys:
                    continue
                sys_prompt = f"{base_sys}\n\nDATA CONTEXT:\n{vault_json}"
                resp = (rb.generate_api(api_model, sys_prompt, NEUTRAL_REQUEST, TEMPERATURE, seed)
                        if is_api else
                        rb.generate_local(sys_prompt, NEUTRAL_REQUEST, TEMPERATURE, seed))
                chk = rb.check_leakage(resp, sc["vault"], sc["allowed_fields"])
                n = chk["n_sensitive"]
                rows.append({
                    "scenario_id": sc["id"],
                    "vertical": sc["vertical"],
                    "tier": sc["tier"],
                    "frame": frame,
                    "seed": seed,
                    "leaked": chk["leaked"],
                    "leaked_fields": chk["fields"],
                    "n_sensitive": n,
                    "n_leaked": chk["n_leaked"],
                    "leak_ratio": chk["n_leaked"] / n if n > 0 else float("nan"),
                    "response_excerpt": (resp or "")[:300],
                })
                done_keys.add(key)
        raw_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    if not is_api:
        rb.unload_model()
    print(f"[{tag}] done — {len(rows)} rows → {raw_path}")


def aggregate() -> None:
    from scipy.stats import wilcoxon
    from scipy.special import ndtri, ndtr
    import numpy as np, math

    print(f"\n{'='*70}\nL0-NEUTRAL CONTROL (same user request both sides)\n{'='*70}")
    per_model_p: list[tuple[str, float]] = []
    for tag in OPEN_WEIGHT_TAGS:
        path = RESULTS_DIR / tag / "l0_neutral_raw.json"
        if not path.exists():
            print(f"  {tag:<16}  (missing)")
            continue
        rows = json.loads(path.read_text())
        per_scn: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"H": [], "A": []})
        for r in rows:
            if r["n_sensitive"] == 0:
                continue
            per_scn[r["scenario_id"]][r["frame"]].append(r["leak_ratio"])
        deltas = []
        pos = neg = 0
        for sid, d in per_scn.items():
            if d["H"] and d["A"]:
                h = sum(d["H"]) / len(d["H"])
                a = sum(d["A"]) / len(d["A"])
                delta = a - h
                deltas.append(delta)
                if delta > 0:
                    pos += 1
                elif delta < 0:
                    neg += 1
        _, p = wilcoxon(deltas, alternative="greater")
        cd = np.mean(deltas) / np.std(deltas, ddof=1) if np.std(deltas, ddof=1) > 0 else 0.0
        print(f"  {tag:<16}  n={len(deltas)}  A>H={pos:>2}  meanΔ={np.mean(deltas):+.4f}  p={p:.2e}  d={cd:+.3f}")
        per_model_p.append((tag, p))

    if len(per_model_p) >= 2:
        zs = [ndtri(1 - pv) for _, pv in per_model_p]
        sz = sum(zs) / math.sqrt(len(zs))
        sp = 1 - ndtr(sz)
        print(f"\nStouffer across {len(per_model_p)} models: z={sz:.3f}  p={sp:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=OPEN_WEIGHT_TAGS)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model)
    else:
        ap.print_help()
