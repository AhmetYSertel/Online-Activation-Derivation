"""
The engine ties the pieces together: it stores regularities, assigns inputs,
runs the online loop, and -- the crux -- decides on each confirmed observation
whether to *generalize* an existing regularity (promote a slot) or *split*
(spawn a new regularity at a cluster boundary).

Three confirmation channels are kept architecturally separate:

  * production confirmation  -> updates the Beta-Binomial posterior when the
                                system replays and the action executes.
  * learning confirmation    -> folds a confirmed observation into structure
                                (generalize / spawn). Governed by action-
                                consistency (replay must reproduce every example's
                                action) and hysteresis.
  * evaluation referee       -> the harness. NEVER read by the engine; only the
                                experiment harness scores outputs with it.

The "teacher" supplies the correct action for an input (the frozen LLM as the
*source* of I/O). The system learns from it on abstention; it never inspects the
referee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import skeleton as sk
from .confirm import Thresholds
from .gate import decide, GateDecision
from .regularity import Action, Regularity

Tokens = Tuple[str, ...]
Teacher = Callable[[str], Action]


@dataclass
class StepResult:
    text: str
    decision: str            # "act" | "abstain"
    action: Optional[Action]
    reason: str              # on abstain: the gate coherence reason ("no_match" if unmatched)
    rid: Optional[int]
    harness_correct: Optional[bool]
    matured_now: bool = False
    learn_outcome: Optional[str] = None   # "fold_strict" | "generalize" | "spawn" | None
    acting_post_mean: Optional[float] = None  # acting regularity's believed success prob (when acting)


def _n_literals(skeleton) -> int:
    return sum(len(p) for k, p in skeleton if k == "lit")


def _reproduces_all(reg_examples: List[Tuple[Tokens, Action]], skeleton) -> bool:
    """A candidate skeleton is action-consistent iff a regularity built on it can
    deterministically replay every example's action. This is the promote-vs-split
    test: city variation reproduces; a different-tool 'boundary' example does not."""
    tmp = Regularity(rid=-1)
    tmp.examples = list(reg_examples)
    tmp.rebuild()
    # rebuild() recomputed its own skeleton; force the candidate skeleton instead,
    # then re-derive template/profiles against it.
    tmp.skeleton = skeleton
    tmp._derive_template()
    tmp._build_profiles()
    if tmp.any_param_unfillable():
        return False
    for tokens, action in reg_examples:
        m = sk.match(skeleton, tokens)
        if m is None:
            return False
        if tmp.replay(m) != action:
            return False
    return True


class Engine:
    def __init__(self, thr: Optional[Thresholds] = None, k_slot: int = 2,
                 self_train: bool = False, use_gate: bool = True,
                 post_decay: float = 1.0, allow_fallthrough: bool = False,
                 confirm_learning: bool = False):
        self.thr = thr or Thresholds()
        self.k_slot = k_slot
        # self_train=False is the safe asymmetry: the system's own acted outputs feed
        # the posterior (production) but NEVER reshape structure (learning). Setting it
        # True turns the asymmetry off -- acted outputs are folded back, which is the
        # self-poisoning path exercised in the Phase 2 scissors ablation.
        self.self_train = self_train
        # use_gate=False bypasses the coherence checks (acts on any fillable most-
        # specific match). Only used for the eager "no-gate" baseline / ablation.
        self.use_gate = use_gate
        # recency decay for every regularity's posterior (1.0 = no forgetting, the
        # Phase 1 default; < 1.0 lets confidence adapt to drift in Phase 2).
        self.post_decay = post_decay
        # allow_fallthrough=False is the safe default: decide on the single most-
        # specific match only. True re-enables falling through to a more general
        # (less specific) match when the specific one abstains -- the ablation that
        # shows specificity preference is load-bearing against boundary silent failures.
        self.allow_fallthrough = allow_fallthrough
        # confirm_learning=False reproduces the packaged behavior: a teacher label is
        # folded into structure on the strength of the teacher alone. =True binds
        # template-induction to the INDEPENDENT confirmation channel (`execute`): a
        # teacher label is learned only if executing it actually succeeds. This stops
        # the engine from folding a confidently-wrong teacher label (e.g. an invalid
        # slot the teacher accepts) into a healthy regularity, where it would otherwise
        # ride that regularity's posterior. Meaningful only with an execute() that is
        # independent of the teacher (a real executor / world signal); with the default
        # teacher-backed execute it is a no-op (the tautology confirms everything).
        self.confirm_learning = confirm_learning
        self.regs: List[Regularity] = []
        self._next_rid = 0

    def _new_rid(self) -> int:
        r = self._next_rid
        self._next_rid += 1
        return r

    # ----------------------------------------------------------- assignment
    def _strict_matches(self, tokens: Tokens) -> List[Tuple[Regularity, Dict[int, Tokens]]]:
        out = []
        for reg in self.regs:
            m = reg.try_match(tokens)
            if m is not None:
                out.append((reg, m))
        # specificity preference: more literal anchors first, then higher posterior.
        out.sort(key=lambda rm: (_n_literals(rm[0].skeleton), rm[0].post.mean()), reverse=True)
        return out

    # ------------------------------------------------------------- folding
    def _find_generalizable(self, tokens: Tokens, action: Action) -> Optional[Regularity]:
        best: Optional[Regularity] = None
        best_lit = -1
        for reg in self.regs:
            if reg.tool != action["tool"]:
                continue  # different tool => boundary, not a slot
            rep = reg.examples[0][0]
            cand = sk.induce_skeleton([list(rep), list(tokens)])
            if not any(k == "lit" for k, _ in cand):
                continue  # would collapse everything to slots
            if sk.match(cand, rep) is None or sk.match(cand, tokens) is None:
                continue
            new_examples = reg.examples + [(tokens, action)]
            if not _reproduces_all(new_examples, cand):
                continue  # action-inconsistent => boundary
            lit = _n_literals(cand)
            if lit > best_lit:
                best_lit = lit
                best = reg
        return best

    def _learn(self, tokens: Tokens, action: Action,
               strict: List[Tuple[Regularity, Dict[int, Tokens]]]) -> Tuple[str, int]:
        # A. consistent strict match -> fold (refines profiles / supports unseen values).
        # The fold is allowed ONLY if the augmented example set stays template-
        # consistent (every param still traces deterministically to a slot). Under a
        # perfect oracle this guard is always satisfied -- only well-formed family
        # instances are ever taught -- so it is a no-op there. Under an HONEST execution
        # signal it is load-bearing: the executor may confirm a teacher action whose
        # param does not match the captured slot (e.g. get_weather("istanbul") for
        # "weather in istanbul please cancel", where the slot is "istanbul please
        # cancel"); folding that would poison the template into an unfillable/semantic
        # state and silently kill the whole regularity's acting. So a structurally
        # foreign example is refused here and left to spawn / defer instead.
        for reg, _ in strict:
            if reg.tool == action["tool"]:
                cand = reg.examples + [(tokens, action)]
                cand_skel = sk.induce_skeleton([list(t) for t, _ in cand])
                if not _reproduces_all(cand, cand_skel):
                    continue  # foreign: do not fold into this regularity
                reg.add_example(tokens, action)
                reg.post.update(True)
                return ("fold_strict", reg.rid)
        # B. near-miss generalizable -> promote slot
        near = self._find_generalizable(tokens, action)
        if near is not None:
            near.add_example(tokens, action)
            near.post.update(True)
            return ("generalize", near.rid)
        # C. boundary / novel -> spawn (born literal, over-specific)
        reg = Regularity(rid=self._new_rid())
        reg.post.decay = self.post_decay
        reg.add_example(tokens, action)
        reg.post.update(True)
        self.regs.append(reg)
        return ("spawn", reg.rid)

    # ---------------------------------------------------------------- step
    def step(self, text: str, teacher: Teacher,
             harness_score: Optional[Callable[[str, Action], bool]] = None,
             execute: Optional[Callable[[str, Action], bool]] = None,
             is_derivable: Optional[Callable[[Action], bool]] = None) -> StepResult:
        # A teacher action is "derivable" if it names a real tool; a punt to the
        # model (e.g. tool "__llm_other__") is not a template to be learned.
        if is_derivable is None:
            is_derivable = lambda a: not str(a.get("tool", "")).startswith("__")
        # The production-confirmation channel. Default: clean (matches the teacher).
        # Phase 2 passes a noisy execute() to stress the posterior.
        if execute is None:
            execute = lambda t, p: p == teacher(t)

        tokens = tuple(sk.tokenize(text))
        strict = self._strict_matches(tokens)  # most-specific first

        # Safe default: decide on the SINGLE most-specific match. Never fall through
        # from a specific-but-immature regularity to a more general one -- a more
        # specific cluster has claimed this input, and acting via a broader cluster is
        # exactly what would cause a boundary silent failure. With allow_fallthrough,
        # we iterate to the next match instead (the ablation).
        candidates = strict if self.allow_fallthrough else strict[:1]
        gate_reason = "no_match"
        for reg, slots in candidates:
            if self.use_gate:
                g = decide(reg, slots, self.thr, self.k_slot)
            else:
                # no-gate baseline: act on any fillable match (still must be replayable)
                ok = (not reg.any_param_unfillable()) and self.thr.may_act(reg.post)
                g = GateDecision(ok, "act" if ok else "immature")
            if g.act:
                produced = reg.replay(slots)
                exec_success = execute(text, produced)        # production confirmation
                reg.post.update(exec_success)
                if self.self_train and exec_success:
                    # asymmetry OFF: fold the system's own output back into structure
                    reg.add_example(tokens, produced)
                hc = harness_score(text, produced) if harness_score else None
                return StepResult(text, "act", produced, "act", reg.rid, hc,
                                  acting_post_mean=reg.post.mean())
            gate_reason = g.reason  # remember the (most-specific) abstention reason

        # ABSTAIN: the teacher (LLM) supplies the action. Learn only if derivable AND
        # (when confirm_learning is on) the action is independently confirmed by the
        # execute channel -- never on teacher self-agreement. An unconfirmed label is
        # still returned for this step (the system used the LLM), but it is NOT folded
        # into structure, so the regularity's slot domain stays clean and the input
        # remains a permanent defer zone rather than a future silent failure.
        action = teacher(text)
        learn_outcome: Optional[str] = None
        rid: Optional[int] = None
        confirmed_for_learning = (not self.confirm_learning) or execute(text, action)
        if is_derivable(action) and confirmed_for_learning:
            learn_outcome, rid = self._learn(tokens, action, strict)
        hc = harness_score(text, action) if harness_score else None
        reg = next((r for r in self.regs if r.rid == rid), None) if rid is not None else None
        matured = bool(reg and self.thr.may_act(reg.post)
                       and (not reg.profiles or reg.min_slot_support() >= self.k_slot))
        return StepResult(text, "abstain", action, gate_reason, rid, hc,
                          matured_now=matured, learn_outcome=learn_outcome)

    # --------------------------------------------------------------- views
    def summary(self) -> List[dict]:
        rows = []
        for r in self.regs:
            rows.append({
                "rid": r.rid,
                "tool": r.tool,
                "skeleton": sk.skeleton_str(r.skeleton),
                "n_examples": len(r.examples),
                "slots": sk.n_slots(r.skeleton),
                "slot_support": r.min_slot_support(),
                "post_mean": round(r.post.mean(), 3),
                "may_act": self.thr.may_act(r.post),
            })
        return rows
