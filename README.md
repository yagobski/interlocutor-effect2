# The Interlocutor Effect

**Anonymous code release** for the NeurIPS 2026 submission *"The Interlocutor Effect: How Agentic Framing Changes Privacy Behaviour in Instruction-Tuned LLMs"* (under double-blind review).

This repository contains experiment runners, scenario sets, raw per-model outputs, and statistical pipelines needed to reproduce every numerical claim in the paper. The paper PDF is not included here to comply with double-blind submission policy.

---

## Quick start

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python ≥ 3.11 recommended. The pinned `requirements.txt` records the exact package versions used to produce all results.

For gated models (Llama-3.1-8B, Mistral-7B-Instruct-v0.3) accept the licence on Hugging Face, then:

```bash
export HF_TOKEN=<your_hf_token>
```

Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct require no token.

### 2. One-command reproduction (primary evidence)

```bash
cd experiments
python run_ie_proof.py --all
```

Runs the pre-registered **L0-clean tagged-vault minimal pair** plus the **phrasing-neutral control** on all four pre-registered open-weight models and prints the statistical report matching Table 1 and Table 3 of the paper. Walltime: roughly 4–6 hours on a single GPU or Apple M-series Mac.

Partial runs:

```bash
python run_ie_proof.py --model llama-3.1-8b   # single model
python run_ie_proof.py --report-only          # re-aggregate already-written outputs
```

### 3. Verify against shipped results

Every JSON in `results/` was produced by the final experiment runs. To regenerate statistics without re-running models:

```bash
cd experiments && python run_ie_proof.py --report-only
```

---

## Models evaluated

| Model | HF identifier | Token? |
|---|---|---|
| Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` | No |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | No |
| Llama-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Yes |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Yes |

All open-weight runs use `torch.bfloat16`. Hardware requirements per model are roughly 6 GB (Qwen-1.5B), 18 GB (Qwen-7B / Mistral-7B), and 20 GB (Llama-8B). CPU-only is possible but slow.

---

## Repository layout

```
experiments/
  run_ie_proof.py             One-command entry point (§4.1)
  run_l0_clean.py             Pre-registered L0 tagged-vault minimal pair (§4.1)
  run_l0_neutral.py           Phrasing-neutral control — system-prompt axis (§4.1)
  run_l0_sysmerge.py          System-prompt → user-turn merge ablation (§4.1)
  run_l0_policy.py            Symmetric data-minimisation defence script
                              (not reported in paper; supplied for follow-up)
  run_bootstrap_ci.py         Bootstrap CIs + Holm correction (§4.1)
  run_twosided_check.py       Two-sided robustness check (App. B)
  run_orthogonal_v2.py        2×2×2 orthogonal ablation (§4.3)
  run_matched_context.py      Matched-context identity isolation (App. H)
  run_dissociation.py         IIV five-test battery (§4.4 / App. C)
  run_mechanistic.py          Extended mechanistic analysis (§4.4 / App. C)
  run_iiv_framing.py          IIV framing-gradient bridge (App. D)
  run_persona_vectors.py      Persona-vectors baseline (App. F)
  run_behavioral.py           Cross-model A2A + framing-level / implicit / persona (App. A)
  run_a2a_llama.py            A2A protocol replication (App. A)
  run_risk_normalized.py      Risk-normalised frontier effect sizes (§4.5 / App. E)
  validate_detector.py        Leakage-detector validation (App. J)
  scenarios_l0_clean.py       51 tagged-vault test scenarios + 20 calibration scenarios
  scenarios_l0_neutral.py     Scenarios for phrasing-neutral control
  scenarios_neurips.py        40 legacy scenarios (A2A, framing gradient, persona)
  scenarios_orthogonal.py     100 held-out scenarios for 2×2×2 ablation
  config.py                   Model registry + shared utilities

results/
  l0_clean/                   PRIMARY — pre-registered tagged-vault minimal pair
    PREREGISTRATION.json      Pre-registration record (SHA-256 in paper)
    calibration/              Calibration run (Llama-3.1-8B, 20 scenarios)
    llama-3.1-8b/             Per-model raw draws (l0_clean_raw.json)
    mistral-7b/
    qwen2.5-1.5b/
    qwen2.5-7b/
    l0_clean_report.json      Aggregated per-model statistics
  l0_neutral/                 Phrasing-neutral control (§4.1 / App. G)
    <model>/l0_neutral_raw.json
  l0_sysmerge/                System-prompt → user-turn merge ablation (§4.1)
    <model>/l0_sysmerge_raw.json
  frontier_iwpe/              Frontier-model replication (§4.5 / App. E)
    benchmark_raw.json        Raw per-trial API responses
    frontier_replication_stats.json
  stats/
    l0_bootstrap_ci.json      Bootstrap CIs + Holm correction
  <model>/                    Per-model legacy outputs for appendix analyses
    behavioral_raw.json       Cross-model A2A (App. A)
    stats.json
    behavioral_framing_levels_raw.json   Framing gradient (App. B)
    stats_framing_levels.json
    behavioral_implicit_raw.json         Implicit MCP framing (App. B)
    stats_implicit.json
    behavioral_orthogonal_v2_raw.json    2×2×2 ablation (§4.3 / App. I)
    stats_orthogonal_v2.json
    behavioral_matched_context_raw.json  Matched-context (App. H)
    stats_matched_context.json
    behavioral_persona_control_raw.json  Persona specificity (App. F)
    stats_persona_control.json
    dissociation_safety.json             IIV five-test battery (§4.4 / App. C)
    iiv_framing_gradient.json            IIV framing gradient (App. D)
    persona_vectors.json                 Persona vectors (App. F)
    results.json                         Mechanistic analysis raw
    a2a_behavioral_raw.json              A2A replication (App. A)
    a2a_stats.json
  detector_validation.json    Detector validation (App. J)
  detector_validation_summary.txt

requirements.txt
LICENSE
```

---

## Script → paper section → output

### Primary evidence (pre-registered, §4.1)

| Script | Paper | Output |
|---|---|---|
| `run_l0_clean.py --calibrate --model llama-3.1-8b` | §4.1 calibration | `results/l0_clean/PREREGISTRATION.json` |
| `run_l0_clean.py --run --model <tag>` | §4.1 Table 1 | `results/l0_clean/<tag>/l0_clean_raw.json` |
| `run_l0_neutral.py --model <tag>` | §4.1 / App. G | `results/l0_neutral/<tag>/l0_neutral_raw.json` |
| `run_l0_sysmerge.py --model <tag>` | §4.1 sysmerge ablation | `results/l0_sysmerge/<tag>/l0_sysmerge_raw.json` |
| `run_bootstrap_ci.py` | §4.1 bootstrap CIs + Holm | `results/stats/l0_bootstrap_ci.json` |
| `run_twosided_check.py` | App. B two-sided check | `results/stats/l0_twosided.json` |
| **`run_ie_proof.py --all`** | **Single entry point** | all of the above |

### Causal ablations (§4.3)

| Script | Paper | Output |
|---|---|---|
| `run_orthogonal_v2.py` | §4.3 2×2×2 ablation | `results/<tag>/behavioral_orthogonal_v2_raw.json`, `stats_orthogonal_v2.json` |
| `run_matched_context.py` | App. H matched-context | `results/<tag>/behavioral_matched_context_raw.json`, `stats_matched_context.json` |

### Representational analysis (§4.4)

| Script | Paper | Output |
|---|---|---|
| `run_dissociation.py --model <tag>` | §4.4 / App. C IIV T1–T5 | `results/<tag>/dissociation_safety.json` |
| `run_mechanistic.py --model <tag>` | §4.4 / App. C extended | `results/<tag>/results.json` |
| `run_iiv_framing.py --model <tag>` | App. D IIV framing gradient | `results/<tag>/iiv_framing_gradient.json` |

### Frontier replication (§4.5)

| Script | Paper | Output |
|---|---|---|
| `run_risk_normalized.py` | §4.5 / App. E | `results/stats/risk_normalized.json` (reads `frontier_iwpe/benchmark_raw.json`) |

### Appendix secondary evidence

| Script | Paper | Output |
|---|---|---|
| `run_behavioral.py` | App. A cross-model A2A | `results/<tag>/behavioral_raw.json`, `stats.json` |
| `run_a2a_llama.py` | App. A A2A replication | `results/<tag>/a2a_behavioral_raw.json`, `a2a_stats.json` |
| `run_behavioral.py --mode framing_levels` | App. B framing gradient | `results/<tag>/stats_framing_levels.json` |
| `run_behavioral.py --mode implicit` | App. B implicit MCP | `results/<tag>/stats_implicit.json` |
| `run_behavioral.py --mode persona_control` | App. F persona specificity | `results/<tag>/stats_persona_control.json` |
| `run_persona_vectors.py` | App. F persona vectors | `results/<tag>/persona_vectors.json` |
| `validate_detector.py` | App. J detector validation | `results/detector_validation.json` |

---

## Headline numbers (all reproducible from this repo)

### Primary L0-clean (Table 1, §4.1)

Pre-registered tagged-vault minimal pair. 51 test scenarios, headroom window `H ∈ (0.10, 0.90)`. All runs at `T=0`, 5 seeds × 5 paraphrase pairs per scenario per model.

| Model | n | Ā−H̄ | Cohen's d | 95% CI | Wilcoxon p |
|---|---|---|---|---|---|
| Llama-3.1-8B | 46 | +0.110 | +0.76 | [+0.46, +1.15] | 2.0×10⁻⁵ |
| Mistral-7B | 42 | +0.208 | +1.32 | [+0.99, +1.85] | 5.0×10⁻⁸ |
| Qwen2.5-1.5B | 50 | +0.226 | +1.41 | [+1.07, +1.89] | 3.4×10⁻⁹ |
| Qwen2.5-7B | 40 | +0.010 | +0.05 | [−0.28, +0.37] | 0.42 |
| **Stouffer (4 models)** | | | **z = 7.70** | | **p = 6.7×10⁻¹⁵** |

### Phrasing-neutral control (§4.1 / App. G)

Identical user request on both sides; system prompt identity only. Llama-3.1-8B: `d = 0.93`, `p = 7.2×10⁻⁷`. Pooled Stouffer `z = 3.81`, `p = 6.8×10⁻⁵`.

### System-prompt → user-turn merge (§4.1)

Llama-3.1-8B: `d = 1.41` (n=44). Mistral-7B: `d = 1.32` (n=42). Stouffer `z = 7.56`, `p = 2.0×10⁻¹⁴`. The dedicated system-prompt channel is not required.

### Orthogonal 2×2×2 ablation (§4.3)

Interlocutor main effect survives Holm correction on Qwen2.5-1.5B (`F=8.40`, `p_corr=0.023`) and Llama-3.1-8B (`F=12.05`, `p_corr=0.003`). Interlocutor × Extraction interaction null on all three models (`p_corr=1.0`): identity is additive with extraction directives.

### Frontier replication (§4.5, Table 2)

| Model | Human | Agent | Δ (pp) | d | Wilcoxon p |
|---|---|---|---|---|---|
| Claude-3.5-Sonnet | 84.7% | 93.5% | +8.78 | 0.365 | 7×10⁻⁷ |
| GPT-4o | 87.2% | 93.0% | +5.86 | 0.243 | 7×10⁻⁴ |
| Llama-3.3-70B | 68.7% | 75.5% | +6.76 | 0.202 | 5×10⁻³ |
| Mistral-Large | 92.0% | 94.5% | +2.50 | 0.133 | 4×10⁻² |

Risk-normalised (`Δ_norm`): Claude +0.66, GPT-4o +0.57, Llama-3.3-70B +0.43, Mistral-Large +0.64. Agent framing closes 43–66% of residual safety headroom.

### Representational analysis (§4.4 / App. C)

IIV linear separability accuracy: 1.000 on all four ≤8B models. Refusal orthogonality `|cos| ∈ [0.105, 0.202]`. Each model passes 4/5 IIV dissociation tests.

### Detector validation (App. J)

Field-level F₁ = 0.993 on 87 adversarial cases. Cohen's κ = 0.991 on 792 real model outputs.

---

## Pre-registration

`results/l0_clean/PREREGISTRATION.json` — committed before any L0-clean test run. Records: calibration model (Llama-3.1-8B), calibration window (0.10, 0.90), 51 test scenarios, seeds `{42, 142, 242, 342, 442}`, 5 paraphrase pairs, and the statistical plan. SHA-256 of this file is reported in the paper.

---

## Anonymity

This repository is anonymised for double-blind review. It contains no author names, affiliations, contact information, or identifying metadata. Synthetic PII in scenario vaults is fictitious; no real individuals are identifiable.

---

## Licence

MIT — see [LICENSE](LICENSE).
