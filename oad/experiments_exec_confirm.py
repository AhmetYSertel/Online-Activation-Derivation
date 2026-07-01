"""
Lever #1 test: does an execution-grounded confirmation channel -- one INDEPENDENT of
the teacher -- move the inherited silent failures into abstention?

Baseline confirmation is a tautology: execute = (produced == teacher(text)), and produced
was learned from the teacher, so it always "confirms". We swap in an execution-grounded
channel modeled by the ground-truth-backed make_execute(): a replayed action confirms iff
it is actually correct (eps=0 = a clean executor). This is INDEPENDENT of the teacher --
truth comes from "the world", not from re-asking the source of the label.

Architectural note: the in-loop confirmation channel (execute) and the out-of-loop referee
(harness.Referee) are SEPARATE instances. The engine uses execute (allowed: that is the
production channel); it never reads the referee (the out-of-loop property survives).

This isolates lever #1 only -- it does NOT change the learning channel, which still FOLDS
the teacher's wrong labels into structure. So it also exposes the coupling that motivates
levers #2/#3: a bad fill folded into an existing regularity shares that regularity's
posterior, so execution-failures on the bad fill can also suppress legitimate acting.

Run:
    export OPENAI_API_KEY=sk-...
    python -m oad.experiments_exec_confirm --model gpt-5.5 --n 1500 --junk-frac 0.15 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

from .engine import Engine
from .workload import Workload, default_families
from .teacher import OpenAITeacher, make_execute
from .analyze_silent_failures import DiskCachingTeacher
from .experiments_real_teacher import _mixed_stream

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")


def run_clean(engine_factory, stream, teacher, gt, execute_factory):
    """Drive the engine; score act AND abstain against ground truth, and split silent
    failures into inherited (a) vs OAD-added (b)."""
    eng = engine_factory()
    referee = lambda x, p: p == gt(x)
    execute = execute_factory()
    acted = acted_correct = abstained = abstain_correct = 0
    inh = added = 0
    weather_acts = weather_abstain = 0
    for t in stream:
        is_w = str(gt(t).get("tool", "")).startswith("get_weather")
        r = eng.step(t, teacher, referee, execute=execute)
        if r.decision == "act":
            acted += 1
            if r.harness_correct:
                acted_correct += 1
            elif teacher(t) != gt(t):
                inh += 1
            else:
                added += 1
            if is_w:
                weather_acts += 1
        else:
            abstained += 1
            abstain_correct += 1 if r.harness_correct else 0
            if is_w:
                weather_abstain += 1
    n = len(stream)
    return {
        "llm_rate": round(100.0 * abstained / n, 1),
        "accuracy": round(100.0 * (acted_correct + abstain_correct) / n, 1),
        "acted": acted, "acted_correct": acted_correct,
        "abstained": abstained,
        "silent_failures": acted - acted_correct,
        "silent_inherited_a": inh, "silent_oad_added_b": added,
        "weather_acts": weather_acts, "weather_abstain": weather_abstain,
    }


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
    for t in sorted(set(stream)):
        teacher(t)
    print(f"unique inputs {len(set(stream))}   new API calls {teacher.api_calls}")

    tautological = lambda: None  # execute=None -> engine default: produced == teacher(text)
    independent = lambda: make_execute(gt, eps=0.0, rng=random.Random(args.seed + 777))

    base = run_clean(lambda: Engine(), stream, teacher, gt, tautological)
    fixed = run_clean(lambda: Engine(), stream, teacher, gt, independent)
    # lever #1 + #2: independent confirmation ALSO gates template-induction.
    full = run_clean(lambda: Engine(confirm_learning=True), stream, teacher, gt, independent)

    report = {"model": args.model, "n": args.n, "seed": args.seed,
              "baseline_teacher_backed_confirm": base,
              "exec_grounded_confirm": fixed,
              "exec_grounded_confirm_plus_gated_induction": full}
    outpath = os.path.join(OUT, "exec_confirm_results.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    def line(tag, r):
        print(f"{tag:32} llm {r['llm_rate']:5}%  acc {r['accuracy']:5}%  "
              f"silent {r['silent_failures']:3} (a={r['silent_inherited_a']} b={r['silent_oad_added_b']})  "
              f"weather: act {r['weather_acts']} / abstain {r['weather_abstain']}")

    print("\n==== CONFIRMATION CHANNEL COMPARISON ====")
    line("baseline (teacher-backed)", base)
    line("exec-grounded confirm only", fixed)
    line("exec-grounded + gated induction", full)
    print(f"\nSaved: {outpath}")
    return report


if __name__ == "__main__":
    main()
