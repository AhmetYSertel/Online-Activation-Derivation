"""
The referee: the out-of-loop evaluator. It scores produced actions against the
correct action. The ENGINE NEVER CALLS THIS -- only experiment drivers do. Keeping
the referee in a separate module from both the engine and the teacher is deliberate:
the engine has no import path to it.

================================ READ THIS CAVEAT ================================
In this synthetic setup the referee is backed by the SAME ground-truth oracle as the
OracleTeacher. So its reliability on any gold set is 1.0 -- but that is because the
oracle defines truth, not because the referee is a validated independent judge. The
independence here is STRUCTURAL (separate object, no engine access), not
INFORMATIONAL. With a real LLM teacher you MUST replace this with an execution-based
or human-validated referee, and calibrate() would then report a meaningful (< 1.0)
agreement that could catch criterion errors. See LIMITATIONS.md.
=================================================================================
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .workload import Action


class Referee:
    def __init__(self, ground_truth: Callable[[str], Action]):
        self._gt = ground_truth

    def score(self, text: str, produced: Optional[Action]) -> bool:
        return produced == self._gt(text)

    def calibrate(self, gold: List[Tuple[str, Action]]) -> float:
        """Agreement of the referee's criterion with a human-labeled gold set. Returns
        1.0 here by construction (synthetic); meaningful only with a real referee."""
        if not gold:
            return 1.0
        agree = sum(1 for text, lbl in gold if self._gt(text) == lbl)
        return agree / len(gold)
