"""
A *regularity* is the unit the system induces (the thing metaphorically called a
"cluster" -- note there is no explicit clustering). It bundles:

    * an induced skeleton (from skeleton.induce_skeleton over its examples),
    * an action template: tool + per-parameter derivation linking each parameter
      to a slot via copy / transform, or marking it const / semantic,
    * a learned slot-content profile per slot (token-count range + token shapes),
      used by the novel-content-word coherence check,
    * a Beta-Binomial success posterior (confirm.Posterior).

Everything here is symbolic + statistical. No trained model, no embeddings.
Action emission is deterministic template instantiation -> identical input yields
identical action by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import skeleton as sk
from .confirm import Posterior

Action = Dict[str, object]  # {"tool": str, "params": {name: str}}
Tokens = Tuple[str, ...]

# Deterministic transforms a parameter value may be derived through. 'copy' is the
# identity (handled separately). These let a parameter trace to a slot even when it
# is not a verbatim copy. (In the synthetic workload params are verbatim copies, so
# transforms stay latent -- as the slot-extraction-method check does in this regime.)
TRANSFORMS: Dict[str, Callable[[str], str]] = {
    "title": lambda s: s.title(),
    "upper": lambda s: s.upper(),
    "nospace": lambda s: s.replace(" ", ""),
}


def _shape(tok: str) -> str:
    if any(c.isdigit() for c in tok):
        return "num"
    if tok.isalpha():
        return "alpha"
    return "other"


@dataclass
class SlotProfile:
    min_len: int = 10**9
    max_len: int = 0
    shapes: set = field(default_factory=set)

    def observe(self, value: Tokens) -> None:
        self.min_len = min(self.min_len, len(value))
        self.max_len = max(self.max_len, len(value))
        for t in value:
            self.shapes.add(_shape(t))

    def accepts(self, value: Tokens) -> bool:
        if not (self.min_len <= len(value) <= self.max_len):
            return False
        return all(_shape(t) in self.shapes for t in value)


# Per-parameter derivation descriptor.
# kind in {"copy", "transform", "const", "semantic"}
@dataclass
class ParamDeriv:
    kind: str
    slot_id: Optional[int] = None
    transform: Optional[str] = None
    const: Optional[str] = None


@dataclass
class Regularity:
    rid: int
    examples: List[Tuple[Tokens, Action]] = field(default_factory=list)
    skeleton: List[sk.Segment] = field(default_factory=list)
    tool: Optional[str] = None
    template: Dict[str, ParamDeriv] = field(default_factory=dict)
    profiles: Dict[int, SlotProfile] = field(default_factory=dict)
    post: Posterior = field(default_factory=Posterior)
    # hysteresis: distinct slot-values observed per slot before structure is trusted
    slot_support: Dict[int, set] = field(default_factory=dict)
    # cap on retained examples (keep the seed + most recent) so long streams stay O(1)
    max_examples: int = 60

    # ----------------------------------------------------------------- build
    def add_example(self, tokens: Tokens, action: Action) -> None:
        self.examples.append((tokens, action))
        if len(self.examples) > self.max_examples:
            # keep examples[0] (the seed defining the skeleton reference) + recent tail
            self.examples = [self.examples[0]] + self.examples[-(self.max_examples - 1):]
        self.rebuild()

    def rebuild(self) -> None:
        inputs = [list(t) for t, _ in self.examples]
        self.skeleton = sk.induce_skeleton(inputs)
        self.tool = self.examples[0][1]["tool"] if self.examples else None
        self._derive_template()
        self._build_profiles()

    def _slot_values_per_example(self) -> List[Optional[Dict[int, Tokens]]]:
        out = []
        for tokens, _ in self.examples:
            out.append(sk.match(self.skeleton, tokens))
        return out

    def _derive_template(self) -> None:
        self.template = {}
        if not self.examples:
            return
        per = self._slot_values_per_example()
        params = self.examples[0][1]["params"]
        slot_ids = [sid for sid, _ in enumerate(
            [s for s in self.skeleton if s[0] == "slot"])]

        for name in params:
            # gather (param_value, slot_values) across examples
            pv = [ex[1]["params"].get(name) for ex in self.examples]
            deriv = ParamDeriv(kind="semantic")

            # try copy from some slot j (consistent across all examples)
            explained = False
            for j in slot_ids:
                ok = True
                for k, m in enumerate(per):
                    if m is None or j not in m:
                        ok = False
                        break
                    if " ".join(m[j]) != pv[k]:
                        ok = False
                        break
                if ok:
                    deriv = ParamDeriv(kind="copy", slot_id=j)
                    explained = True
                    break

            # try transform from some slot j
            if not explained:
                for j in slot_ids:
                    for tname, fn in TRANSFORMS.items():
                        ok = True
                        for k, m in enumerate(per):
                            if m is None or j not in m or fn(" ".join(m[j])) != pv[k]:
                                ok = False
                                break
                        if ok:
                            deriv = ParamDeriv(kind="transform", slot_id=j, transform=tname)
                            explained = True
                            break
                    if explained:
                        break

            # try const (same value everywhere, not slot-linked)
            if not explained and len(set(pv)) == 1 and pv[0] is not None:
                deriv = ParamDeriv(kind="const", const=pv[0])

            self.template[name] = deriv

    def _build_profiles(self) -> None:
        self.profiles = {}
        self.slot_support = {}
        per = self._slot_values_per_example()
        for m in per:
            if m is None:
                continue
            for sid, val in m.items():
                self.profiles.setdefault(sid, SlotProfile()).observe(val)
                self.slot_support.setdefault(sid, set()).add(" ".join(val))

    # ----------------------------------------------------------------- query
    def try_match(self, tokens: Tokens) -> Optional[Dict[int, Tokens]]:
        return sk.match(self.skeleton, tokens)

    def all_params_semantic(self) -> bool:
        if not self.template:
            return False
        return all(d.kind == "semantic" for d in self.template.values())

    def any_param_unfillable(self) -> bool:
        # semantic params cannot be filled deterministically at replay time
        return any(d.kind == "semantic" for d in self.template.values())

    def slot_content_ok(self, slots: Dict[int, Tokens]) -> bool:
        for sid, val in slots.items():
            prof = self.profiles.get(sid)
            if prof is None or not prof.accepts(val):
                return False
        return True

    def replay(self, slots: Dict[int, Tokens]) -> Optional[Action]:
        if self.any_param_unfillable():
            return None
        params: Dict[str, str] = {}
        for name, d in self.template.items():
            if d.kind == "copy":
                params[name] = " ".join(slots[d.slot_id])
            elif d.kind == "transform":
                params[name] = TRANSFORMS[d.transform](" ".join(slots[d.slot_id]))
            elif d.kind == "const":
                params[name] = d.const
            else:
                return None
        return {"tool": self.tool, "params": params}

    def min_slot_support(self) -> int:
        """Fewest distinct values any slot has been confirmed with (hysteresis)."""
        if not self.slot_support:
            return 0
        return min(len(v) for v in self.slot_support.values())
