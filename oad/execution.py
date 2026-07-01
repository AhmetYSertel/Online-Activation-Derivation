"""
execution.py -- an HONEST execution-confirmation channel (ROADMAP Phase A2).

Why this module exists
----------------------
The packaged "independent" confirmation channel is `teacher.make_execute`, which
returns `produced == ground_truth(text)` (+/- eps). That is independent of the
*teacher* but NOT of the *referee*: both read the same ground-truth oracle. So the
109 -> 0 "fix" rests on handing the system an oracle that already knows the answer.
Independence there is structural, not informational (see LIMITATIONS.md #2).

This module supplies a confirmation signal that has NO access to ground_truth. A
`ToolEnvironment` decides whether executing an action *succeeds* purely from
per-tool input schemas -- the validity contract a real backing tool would enforce
(a geocoder rejects "1945"; a calendar rejects "2026"; a contact book rejects
"99problems"; a smart-home hub rejects "12"). A call "succeeds" iff it is
schema-valid and would not raise. Correctness by a stricter semantic criterion is
NOT consulted.

Honesty of the result
---------------------
This is strictly more independent than `== ground_truth`, but it is still
hand-specified, and -- crucially -- it is deliberately *imperfect*: some wrong
actions are schema-valid and will be confirmed (e.g. a long but well-formed
reminder, or get_weather("istanbul") produced for a paraphrased prompt). That is
the point. A pure execution signal recovers most of the safety but cannot catch a
wrong action that executes cleanly; characterizing that residue is a finding, not a
bug. The eps knob models a tool that returns a plausible-but-wrong result instead
of erroring, for the Phase-D noisy-executor sweep.

Nothing here imports the workload ground truth or the harness referee.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

Action = Dict[str, object]

# Independent world knowledge the *tools* would have -- authored from the tool's
# point of view, not copied from the workload's ground-truth labeller.
KNOWN_DAYS = frozenset(
    "today tomorrow yesterday monday tuesday wednesday thursday friday "
    "saturday sunday".split()
)
KNOWN_DEVICES = frozenset(
    "door window garage blinds lights gate curtains thermostat lock".split()
)


def _toks(value: object) -> List[str]:
    return str(value).strip().split()


# A validator answers: would this tool ACCEPT these params (not raise)?  It says
# nothing about whether the tool is the *right* one for the request.
def _v_weather(p: Dict[str, object]) -> bool:
    t = _toks(p.get("city", ""))
    return len(t) == 1 and t[0].isalpha()        # a single alphabetic place token


def _v_calendar(p: Dict[str, object]) -> bool:
    t = _toks(p.get("day", ""))
    return len(t) == 1 and t[0].lower() in KNOWN_DAYS


def _v_contact(p: Dict[str, object]) -> bool:
    t = _toks(p.get("name", ""))
    return 1 <= len(t) <= 2 and all(w.isalpha() for w in t)


def _v_smarthome(p: Dict[str, object]) -> bool:
    t = _toks(p.get("target", ""))
    return len(t) == 1 and t[0].lower() in KNOWN_DEVICES


def _v_notes(p: Dict[str, object]) -> bool:
    # a notes search accepts any non-empty query (permissive by nature)
    return len(_toks(p.get("query", ""))) >= 1


def _v_reminder(p: Dict[str, object]) -> bool:
    # a reminder accepts any non-empty free text -- so a long-but-well-formed
    # reminder is NOT rejected. This is an intentional honesty gap.
    return len(_toks(p.get("text", ""))) >= 1


@dataclass
class ToolEnvironment:
    """Maps tool name -> param validator. Unknown tools and declines do not execute
    as a successful tool call."""
    validators: Dict[str, Callable[[Dict[str, object]], bool]] = field(
        default_factory=lambda: {
            "get_weather": _v_weather,
            "get_calendar": _v_calendar,
            "lookup_contact": _v_contact,
            "smart_home": _v_smarthome,
            "search_notes": _v_notes,
            "create_reminder": _v_reminder,
        }
    )

    def succeeds(self, action: Optional[Action]) -> bool:
        if not action:
            return False
        tool = str(action.get("tool", ""))
        if tool.startswith("__"):          # a decline / fallback is not a tool execution
            return False
        v = self.validators.get(tool)
        if v is None:                      # unknown tool -> the call would raise
            return False
        try:
            return bool(v(action.get("params", {}) or {}))
        except Exception:
            return False


def make_honest_execute(env: Optional[ToolEnvironment] = None, eps: float = 0.0,
                        rng: Optional[random.Random] = None
                        ) -> Callable[[str, Action], bool]:
    """Returns execute(text, produced) -> bool with NO ground-truth access.

    A schema-valid action confirms. A schema-invalid action confirms only with
    probability eps (eps>0 models a tool that returns a plausible-but-wrong result
    instead of erroring -- the Phase-D noisy-executor channel). eps=0 is a clean,
    honest executor.
    """
    env = env or ToolEnvironment()
    rng = rng or random.Random(0)

    def execute(text: str, produced: Action) -> bool:
        if env.succeeds(produced):
            return True
        return rng.random() < eps

    return execute
