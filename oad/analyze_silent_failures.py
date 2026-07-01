"""
Decompose OAD's silent failures under a real teacher into:

  (a) inherited error  -- the teacher itself is wrong on this input. OAD learned the
                          teacher's wrong-but-coherent label and replayed it. always-LLM
                          gets these wrong too (it asks and is misled); OAD's only
                          difference is character: it fails autonomously and confidently.
  (b) OAD-added error  -- the teacher is RIGHT on this input, but OAD's replay is wrong.
                          This is error the engine introduces on its own (bad slot
                          induction / wrong template / premature generalization).

If accuracy(OAD) == accuracy(always-LLM), we expect (b) ~ 0: OAD carries the teacher's
errors, it does not add new ones. If (b) ~ 0, the gate logic is sound and the entire
problem is that the production-confirmation channel is a tautology
(execute = produced == teacher(text), and produced was learned FROM the teacher), so it
carries zero independent information about correctness. This is LIMITATIONS.md #2,
empirically.

The teacher is cached to disk so the run is deterministic and free to re-run.

Run:
    export OPENAI_API_KEY=sk-...
    python -m oad.analyze_silent_failures --model gpt-5.5 --n 1500 --junk-frac 0.15 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Dict, List

from .engine import Engine
from .workload import Workload, default_families, Action
from .teacher import OpenAITeacher
from .experiments_real_teacher import _mixed_stream  # identical stream construction

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


class DiskCachingTeacher:
    """Caches teacher answers to a JSON file keyed by (model, text). Deterministic and
    free on re-run; only genuinely-new inputs hit the API."""

    def __init__(self, base, model: str, path: str):
        self.base = base
        self.model = model
        self.path = path
        self.api_calls = 0
        self.cache: Dict[str, Action] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.cache = json.load(f)

    def _key(self, text: str) -> str:
        return f"{self.model}\x1f{text}"

    def __call__(self, text: str) -> Action:
        k = self._key(text)
        if k not in self.cache:
            self.api_calls += 1
            self.cache[k] = self.base(text)
            with open(self.path, "w") as f:
                json.dump(self.cache, f)
        return self.cache[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    wl = Workload(seed=args.seed)
    gt = wl.ground_truth
    stream = _mixed_stream(wl, args.n, args.junk_frac, args.seed)

    cache_path = os.path.join(OUT, f"_teacher_cache_{args.model}.json")
    teacher = DiskCachingTeacher(OpenAITeacher(model=args.model, families=default_families()),
                                 args.model, cache_path)

    # warm the cache deterministically over unique inputs
    for t in sorted(set(stream)):
        teacher(t)
    print(f"unique inputs: {len(set(stream))}   new API calls this run: {teacher.api_calls}")

    eng = Engine()  # full design
    # per-step records
    inherited = []      # (a): act, wrong, teacher also wrong
    oad_added = []       # (b): act, wrong, teacher correct
    abstain_wrong = []   # asked the teacher and it was wrong (not silent -- it's an LLM call)
    acted = acted_correct = abstained = 0

    referee = lambda x, p: p == gt(x)
    for t in stream:
        r = eng.step(t, teacher, referee, execute=None)
        if r.decision == "act":
            acted += 1
            if r.harness_correct:
                acted_correct += 1
            else:
                teacher_wrong = (teacher(t) != gt(t))
                (inherited if teacher_wrong else oad_added).append(
                    {"input": t, "produced": r.action, "teacher": teacher(t), "gt": gt(t)})
        else:
            abstained += 1
            if not r.harness_correct:
                abstain_wrong.append({"input": t, "teacher": teacher(t), "gt": gt(t)})

    silent = len(inherited) + len(oad_added)

    def fam_breakdown(rows):
        c = Counter()
        for row in rows:
            # label by the tool the produced/teacher action used
            c[row.get("produced", row.get("teacher", {})).get("tool", "?")] += 1
        return dict(c)

    def uniq(rows):
        seen = {}
        for row in rows:
            seen[row["input"]] = row
        return list(seen.values())

    report = {
        "model": args.model, "n": args.n, "seed": args.seed,
        "acted": acted, "acted_correct": acted_correct,
        "abstained": abstained,
        "silent_failures_total": silent,
        "silent_inherited_a": len(inherited),     # teacher also wrong
        "silent_oad_added_b": len(oad_added),      # teacher right, OAD wrong
        "abstain_wrong_count": len(abstain_wrong),
        "inherited_by_tool": fam_breakdown(inherited),
        "oad_added_by_tool": fam_breakdown(oad_added),
        "inherited_unique_inputs": [r["input"] for r in uniq(inherited)],
        "oad_added_unique_examples": uniq(oad_added)[:25],
    }
    outpath = os.path.join(OUT, "silent_failure_breakdown.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n==== SILENT FAILURE DECOMPOSITION ====")
    print(f"acted {acted}  (correct {acted_correct})   abstained {abstained}")
    print(f"silent failures total:           {silent}")
    print(f"  (a) inherited  (teacher wrong): {len(inherited)}   "
          f"unique inputs: {len(uniq(inherited))}")
    print(f"  (b) OAD-added (teacher right):  {len(oad_added)}   "
          f"unique inputs: {len(uniq(oad_added))}")
    print(f"abstain-and-wrong (LLM asked, teacher erred): {len(abstain_wrong)}")
    print(f"\ninherited unique inputs: {[r['input'] for r in uniq(inherited)]}")
    if oad_added:
        print(f"\n(b) OAD-ADDED examples (the ones that would indict the gate):")
        for r in uniq(oad_added)[:25]:
            print(f"   {r['input']!r}  produced={r['produced']}  teacher={r['teacher']}  gt={r['gt']}")
    else:
        print("\n(b) = 0  -> OAD adds no error beyond the teacher; gate logic is sound, "
              "the defect is entirely the teacher-backed (tautological) confirmation channel.")
    print(f"\nSaved: {outpath}")
    return report


if __name__ == "__main__":
    main()
