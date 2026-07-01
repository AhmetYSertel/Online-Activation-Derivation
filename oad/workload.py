"""
Synthetic workload generator (teacher-independent in the sense that the engine
never sees how inputs are produced).

================================ READ THIS CAVEAT ================================
The "teacher" built on top of this workload (teacher.OracleTeacher) is a
GROUND-TRUTH ORACLE, *not* a real frozen LLM. It returns the workload's exact
correct action for every well-formed input, so it never makes the mistakes a real
LLM teacher would: label noise, paraphrase confusion, tool-selection errors,
hallucinated parameters. Likewise the referee (harness.Referee) is backed by the
SAME oracle, so its reliability is 1.0 by construction -- the independence between
teacher and referee here is STRUCTURAL (separate code paths / objects), not
INFORMATIONAL.

Consequence for claims: results obtained with the oracle teacher validate the
MECHANISM (does online derivation, gating, slot induction, and the safety asymmetry
behave as designed). They do NOT establish transfer to real agent traffic. To make
a transfer claim, swap in teacher.OpenAITeacher (a real GPT model) and/or
teacher.NoisyTeacher (injected label noise), and use an external referee. See
LIMITATIONS.md.
=================================================================================

The families and adversarial probes are also authored by the same hand as the
system, so the adversarial set is illustrative, not adversarially optimized. A
held-out, third-party, or red-teamed probe set is future work.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import skeleton as sk

Action = Dict[str, object]
Tokens = Tuple[str, ...]
FALLBACK: Action = {"tool": "__llm_other__", "params": {}}


@dataclass
class Family:
    name: str
    phrasing: str                 # e.g. "weather in {x}"
    tool: str
    param: str                    # action parameter the slot flows to (copy)
    vocab: List[str]              # slot fills (each may be multi-word)
    min_tok: int
    max_tok: int

    def skeleton(self) -> List[sk.Segment]:
        parts = self.phrasing.split("{x}")
        segs: List[sk.Segment] = []
        sid = 0
        for i, part in enumerate(parts):
            toks = sk.tokenize(part)
            if toks:
                segs.append(("lit", tuple(toks)))
            if i < len(parts) - 1:
                segs.append(("slot", sid))
                sid += 1
        return segs

    def valid_fill(self, value: Tokens) -> bool:
        # A valid family instance: right slot length, alphabetic content. A digit or
        # an over-long span makes it NOT a member of this family (its ground truth is
        # the fallback) -- this is what makes adversarial probes surface-distinguishable.
        if not (self.min_tok <= len(value) <= self.max_tok):
            return False
        return all(t.isalpha() for t in value)

    def action_for(self, value: Tokens) -> Action:
        return {"tool": self.tool, "params": {self.param: " ".join(value)}}

    def render(self, fill: str) -> str:
        return self.phrasing.replace("{x}", fill)


def default_families() -> List[Family]:
    return [
        Family("weather", "weather in {x}", "get_weather", "city",
               ["istanbul", "tokyo", "berlin", "paris", "cairo", "lisbon",
                "oslo", "madrid", "delhi", "lima"], 1, 1),
        Family("calendar", "show my calendar for {x}", "get_calendar", "day",
               ["today", "tomorrow", "monday", "tuesday", "friday", "sunday"], 1, 1),
        Family("notes", "search my notes about {x}", "search_notes", "query",
               ["alpha", "physics", "budget", "thesis", "kpop", "memory"], 1, 1),
        Family("reminder", "remind me to {x}", "create_reminder", "text",
               ["call mom", "buy milk", "walk dog", "email boss", "pay rent"], 1, 3),
        Family("contact", "look up contact {x}", "lookup_contact", "name",
               ["alice", "bob", "carol", "dave", "erin", "frank"], 1, 1),
        Family("smarthome", "open the {x}", "smart_home", "target",
               ["door", "window", "garage", "blinds"], 1, 1),
    ]


# A boundary family: shares the "open the {x}" anchor structure but a different tool.
# Used only in the slot-induction split experiment.
BOUNDARY_FAMILY = Family("banking", "open the account", "banking", "__none__", [], 0, 0)


class Workload:
    def __init__(self, seed: int = 0, families: Optional[List[Family]] = None,
                 include_boundary: bool = False):
        self.rng = random.Random(seed)
        self.families = families or default_families()
        self._match_set = list(self.families)
        if include_boundary:
            self._match_set = self.families + [BOUNDARY_FAMILY]

    # ---------------------------------------------------------- ground truth
    def ground_truth(self, text: str) -> Action:
        """The reference correct action for an input. Used to BUILD the oracle teacher
        and the referee (as separate objects). Returns FALLBACK ("ask the LLM") for
        anything that is not a valid family instance."""
        toks = tuple(sk.tokenize(text))
        best: Optional[Action] = None
        best_lit = -1
        for fam in self._match_set:
            skel = fam.skeleton()
            m = sk.match(skel, toks)
            lit = sum(len(p) for k, p in skel if k == "lit")
            if fam is BOUNDARY_FAMILY:
                if toks == tuple(sk.tokenize("open the account")) and lit > best_lit:
                    best = {"tool": "banking", "params": {"acct": "default"}}
                    best_lit = lit
                continue
            if m is None:
                continue
            slot_val = m.get(0, ())
            if not fam.valid_fill(slot_val):
                continue
            if lit > best_lit:
                best = fam.action_for(slot_val)
                best_lit = lit
        return best if best is not None else dict(FALLBACK)

    # -------------------------------------------------------------- streams
    def stream(self, n: int, zipf_s: float = 1.1) -> List[str]:
        """Sequential call stream; family chosen by Zipfian weights (skewed traffic),
        fill chosen uniformly from the family vocab."""
        k = len(self.families)
        weights = [1.0 / ((i + 1) ** zipf_s) for i in range(k)]
        out: List[str] = []
        for _ in range(n):
            fam = self.rng.choices(self.families, weights=weights, k=1)[0]
            fill = self.rng.choice(fam.vocab)
            out.append(fam.render(fill))
        return out

    def in_domain_probes(self, m: int = 60) -> List[str]:
        out = []
        for _ in range(m):
            fam = self.rng.choice(self.families)
            out.append(fam.render(self.rng.choice(fam.vocab)))
        return out

    def adversarial_probes(self) -> List[Tuple[str, str]]:
        """(text, kind) pairs that look related but are not valid family instances.
        CAVEAT: illustrative, hand-authored, not adversarially optimized."""
        return [
            ("weather in 1945", "deviant_slot_digit"),
            ("weather in nineteen forty five", "deviant_slot_len"),
            ("weather in istanbul please cancel", "slot_injection"),
            ("show my calendar for 2026", "deviant_slot_digit"),
            ("look up contact 99problems", "deviant_slot_digit"),
            ("weather forecast for istanbul", "anchor_break"),
            ("open the 12", "deviant_slot_digit"),
            ("remind me to do a b c d e", "deviant_slot_len"),
        ]

    def ood_probes(self) -> List[str]:
        return [
            "translate hello into french",
            "what is the stock price of apple",
            "convert ten dollars to euros",
            "who won the world cup",
            "define photosynthesis",
            "book a flight to rome",
        ]
