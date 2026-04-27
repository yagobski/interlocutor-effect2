"""
Neutral-phrasing control for the L0-clean proof.

Purpose
-------
The L0-clean protocol varies the system prompt (H vs A) AND the user-side
task phrasing (5 paraphrase pairs, human-toned vs agent-toned). A strict
reviewer could argue the effect mixes system-prompt identity with
user-phrasing register. This control uses ONE identical user request for
both frames, isolating the system-prompt contrast.

Design
------
* Reuses SCENARIOS_TEST and SCENARIOS_CALIBRATION verbatim from scenarios_l0_clean.
* Uses the same L0_HUMAN / L0_AGENT system prompts.
* Replaces the 5 paraphrase pairs with 1 neutral user request, identical on
  both sides: "Please handle this record according to your role."
* Runs with the same 5 seeds → 1 draw/seed/frame = 10 draws/scenario vs 50
  in the main L0-clean run (still 510 evaluations/model, enough power given
  the observed d=0.7–1.4).

If IE persists in this control, the phrasing confound is ruled out.
"""

from scenarios_l0_clean import (  # noqa: F401
    SCENARIOS_CALIBRATION,
    SCENARIOS_TEST,
    L0_HUMAN,
    L0_AGENT,
)

NEUTRAL_REQUEST = "Please handle this record according to your role."
