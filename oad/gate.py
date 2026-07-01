"""
The fixed, auditable gate. Given a strictly-matched regularity and the extracted
slot values, it decides ACT (autonomous replay) vs ABSTAIN (route to the LLM).
Every abstention carries an inspectable reason -- this is what keeps the safety
auditable and is the source of the "zero silent failures with a reason" result.

The gate is *not* a learned model and reads *no* model-internal confidence. It is
a fixed priority of checks:

  1. slot-extraction-method  -- refuse replay if the action cannot be derived
                                deterministically from slots (any semantic param).
  2. template-skeleton       -- a strict anchor match is required (near-misses
                                never reach the gate as strict; they abstain at
                                assignment).
  3. novel-content-word      -- refuse replay if a slot value falls outside the
                                learned slot-content profile (token count / shape).
  4. maturity + hysteresis   -- the Beta-Binomial acting bar must be cleared, and
                                the slot must have enough distinct confirmed
                                support before acting on unseen values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .confirm import Thresholds
from .regularity import Regularity

Tokens = Tuple[str, ...]


@dataclass
class GateDecision:
    act: bool
    reason: str  # "act" if acting; otherwise the abstention cause


def decide(reg: Regularity, slots: Dict[int, Tokens], thr: Thresholds, k_slot: int) -> GateDecision:
    # 1. slot-extraction-method
    if reg.any_param_unfillable():
        return GateDecision(False, "slot_extraction_method")
    # 3. novel-content-word (slot content profile)
    if not reg.slot_content_ok(slots):
        return GateDecision(False, "novel_content_word")
    # 4a. maturity (Beta-Binomial acting bar)
    if not thr.may_act(reg.post):
        return GateDecision(False, "immature")
    # 4b. hysteresis: need enough distinct confirmed slot support before acting on
    #     unseen values (only relevant when the regularity has slots).
    if reg.profiles and reg.min_slot_support() < k_slot:
        return GateDecision(False, "insufficient_support")
    return GateDecision(True, "act")
