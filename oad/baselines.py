"""
baselines.py -- ROADMAP Phase C: a real cost-reduction peer to compare OAD against
on the cost-safety axis. OAD's own LIMITATIONS.md #4 states always-call/eager are not
real competitors; the real peer is a semantic (embedding) action cache.

SemanticCache (GPTCache-style): embed the query; if its cosine similarity to a stored
key is >= threshold, SERVE the stored action with no model call (a "hit"); otherwise
call the teacher and store (query, action) (a "miss" = model call). It has NO gate and
NO slot induction -- a hit returns the cached action *verbatim*, including its
parameter. So on slot-varying traffic it serves a STALE parameter, and on an
irrelevance probe that embeds near a cached tool-call it serves that call instead of
declining. Those are exactly the silent failures OAD's gate + slot induction avoid.

The threshold traces the cache's cost-safety curve; OAD is a single operating point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Action = Dict[str, object]


class MiniLMEncoder:
    """all-MiniLM-L6-v2 sentence embeddings, normalized, memoized per text."""
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model)
        self._memo: Dict[str, np.ndarray] = {}

    def __call__(self, text: str) -> np.ndarray:
        v = self._memo.get(text)
        if v is None:
            v = self._m.encode([text], normalize_embeddings=True)[0].astype(np.float32)
            self._memo[text] = v
        return v


@dataclass
class SemanticCache:
    threshold: float
    encoder: Callable[[str], np.ndarray]
    keys: List[np.ndarray] = field(default_factory=list)
    acts: List[Action] = field(default_factory=list)
    _mat: Optional[np.ndarray] = None

    def _nearest(self, v: np.ndarray) -> Tuple[float, int]:
        if not self.keys:
            return -1.0, -1
        if self._mat is None or self._mat.shape[0] != len(self.keys):
            self._mat = np.vstack(self.keys)
        sims = self._mat @ v                      # all keys are normalized
        j = int(np.argmax(sims))
        return float(sims[j]), j

    def step(self, text: str, teacher: Callable[[str], Action]) -> Tuple[str, Action]:
        v = self.encoder(text)
        sim, j = self._nearest(v)
        if sim >= self.threshold:
            return "hit", self.acts[j]            # served from cache, no model call
        a = teacher(text)                          # miss -> model call
        self.keys.append(v); self.acts.append(a); self._mat = None
        return "miss", a


def run_exact_cache(stream: List[str], teacher: Callable[[str], Action],
                    score: Callable[[str, Action], bool]) -> dict:
    """Exact-match (verbatim) cache: serves only on an identical prior input. Safe by
    construction on this workload (a repeat is identical), but cannot generalize to an
    unseen slot value -- so on an open slot space it keeps calling the model. The
    reference point that shows OAD's win is GENERALIZATION, not just caching."""
    seen: Dict[str, Action] = {}
    hits = misses = correct = silent = 0
    for t in stream:
        if t in seen:
            a = seen[t]; hits += 1; ok = score(t, a); correct += ok; silent += (not ok)
        else:
            a = teacher(t); seen[t] = a; misses += 1; correct += score(t, a)
    n = len(stream)
    return {"call_rate": round(100.0 * misses / n, 2),
            "accuracy": round(100.0 * correct / n, 2),
            "silent_rate": round(100.0 * silent / n, 2), "silent_failures": silent}


def run_semantic_cache(stream: List[str], teacher: Callable[[str], Action],
                       score: Callable[[str, Action], bool], threshold: float,
                       encoder: Callable[[str], np.ndarray]) -> dict:
    cache = SemanticCache(threshold=threshold, encoder=encoder)
    hits = misses = correct = silent = 0
    for t in stream:
        kind, a = cache.step(t, teacher)
        ok = score(t, a)
        correct += ok
        if kind == "hit":
            hits += 1
            silent += (not ok)                     # served a wrong cached action
        else:
            misses += 1
    n = len(stream)
    return {
        "threshold": threshold,
        "call_rate": round(100.0 * misses / n, 2),
        "accuracy": round(100.0 * correct / n, 2),
        "silent_rate": round(100.0 * silent / n, 2),
        "silent_failures": silent,
    }
