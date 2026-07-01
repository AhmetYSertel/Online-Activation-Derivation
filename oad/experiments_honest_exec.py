"""
experiments_honest_exec.py -- ROADMAP Phase A2 deliverable.

Re-runs the exec-confirmation progression with an HONEST execution signal
(`execution.make_honest_execute`) that has NO access to ground_truth, alongside the
original oracle-backed signal, in one run, so the contrast is in a single table.

Runs fully OFFLINE from the packaged GPT-5.5 cache (results/_teacher_cache_gpt-5.5.json):
no OPENAI_API_KEY, no network. The stream is regenerated identically to the packaged
real-teacher run (n=1500, junk=0.15, seed=1).

    python -m oad.experiments_honest_exec
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Callable, Dict, List, Optional

from .engine import Engine
from .workload import Workload, default_families, Action, FALLBACK
from .teacher import make_execute
from .execution import ToolEnvironment, make_honest_execute
from .experiments_real_teacher import _mixed_stream  # identical stream construction

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")


class FileCachedTeacher:
    """Offline teacher: reads the packaged (model,text)->action cache and never calls
    an API. Raises on a genuine miss so a stream/cache mismatch is loud, not silent."""

    def __init__(self, model: str, path: str):
        self.model = model
        with open(path) as f:
            self.cache: Dict[str, Action] = json.load(f)
        self.misses: List[str] = []

    def __call__(self, text: str) -> Action:
        k = f"{self.model}\x1f{text}"
        if k not in self.cache:
            self.misses.append(text)
            return dict(FALLBACK)
        return self.cache[k]


def run(engine_factory, stream, teacher, gt, execute_factory) -> dict:
    """Drive the engine; score act AND abstain vs ground truth; split silent failures
    into inherited (a) vs OAD-added (b). Also tracks which unique inputs OAD acted
    wrongly on (the residual silent-failure surface)."""
    eng = engine_factory()
    referee = lambda x, p: p == gt(x)
    execute = execute_factory()
    acted = acted_correct = abstained = abstain_correct = 0
    inh = added = 0
    weather_acts = weather_abstain = 0
    silent_inputs: Counter = Counter()
    for t in stream:
        is_w = str(gt(t).get("tool", "")).startswith("get_weather")
        r = eng.step(t, teacher, referee, execute=execute)
        if r.decision == "act":
            acted += 1
            if r.harness_correct:
                acted_correct += 1
            else:
                silent_inputs[t] += 1
                if teacher(t) != gt(t):
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
        "acted": acted, "acted_correct": acted_correct, "abstained": abstained,
        "silent_failures": acted - acted_correct,
        "silent_inherited_a": inh, "silent_oad_added_b": added,
        "weather_acts": weather_acts, "weather_abstain": weather_abstain,
        "silent_unique_inputs": dict(silent_inputs),
    }


def executor_calibration(unique_inputs, teacher, gt, env: ToolEnvironment) -> dict:
    """How honest/reliable is the execution signal itself? For each unique input, run
    the HONEST executor on the TEACHER's action and check whether 'succeeds' agrees
    with 'action is actually correct'. This is the < 1.0 reliability the safety claim
    now rests on (instead of a perfect oracle)."""
    tp = tn = fp = fn = 0
    leaks = []   # wrong action the executor wrongly confirms (false positive)
    for t in unique_inputs:
        a = teacher(t)
        correct = (a == gt(t))
        ok = env.succeeds(a)
        if correct and ok:
            tp += 1
        elif (not correct) and (not ok):
            tn += 1
        elif (not correct) and ok:
            fp += 1
            leaks.append({"input": t, "action": a})
        else:
            fn += 1
    total = max(1, tp + tn + fp + fn)
    return {
        "n_unique": len(unique_inputs),
        "agreement_pct": round(100.0 * (tp + tn) / total, 1),
        "confirms_wrong_action_fp": fp,    # the residual silent-failure source
        "rejects_correct_action_fn": fn,   # over-deferral source
        "fp_examples": leaks[:10],
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
    teacher = FileCachedTeacher(args.model, cache_path)

    uniq = sorted(set(stream))
    for t in uniq:  # warm/validate against cache
        teacher(t)
    if teacher.misses:
        print(f"WARNING: {len(teacher.misses)} stream inputs missing from cache "
              f"(stream/cache mismatch): {teacher.misses[:5]}")
    else:
        print(f"offline OK: all {len(uniq)} unique inputs served from cache")

    env = ToolEnvironment()
    tautological = lambda: None
    oracle_exec = lambda: make_execute(gt, eps=0.0, rng=random.Random(args.seed + 777))
    honest_exec = lambda: make_honest_execute(env, eps=0.0,
                                              rng=random.Random(args.seed + 777))

    rows = {
        "baseline_teacher_backed":              run(lambda: Engine(),
                                                    stream, teacher, gt, tautological),
        "oracle_exec_confirm":                  run(lambda: Engine(),
                                                    stream, teacher, gt, oracle_exec),
        "oracle_exec_confirm_gated_induction":  run(lambda: Engine(confirm_learning=True),
                                                    stream, teacher, gt, oracle_exec),
        "honest_exec_confirm":                  run(lambda: Engine(),
                                                    stream, teacher, gt, honest_exec),
        "honest_exec_confirm_gated_induction":  run(lambda: Engine(confirm_learning=True),
                                                    stream, teacher, gt, honest_exec),
    }
    cal = executor_calibration(uniq, teacher, gt, env)

    report = {"model": args.model, "n": args.n, "junk_frac": args.junk_frac,
              "seed": args.seed, "executor_calibration": cal, "rows": rows}
    outpath = os.path.join(OUT, "honest_exec_results.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    def line(tag, r):
        print(f"{tag:38} llm {r['llm_rate']:5}%  acc {r['accuracy']:5}%  "
              f"silent {r['silent_failures']:3} (a={r['silent_inherited_a']} "
              f"b={r['silent_oad_added_b']})  weather act/abstain {r['weather_acts']}/{r['weather_abstain']}")

    print("\n==== CONFIRMATION CHANNEL: ORACLE-BACKED vs HONEST (no ground truth) ====")
    line("baseline (teacher-backed tautology)", rows["baseline_teacher_backed"])
    line("oracle exec-confirm", rows["oracle_exec_confirm"])
    line("oracle exec + gated induction", rows["oracle_exec_confirm_gated_induction"])
    line("HONEST exec-confirm", rows["honest_exec_confirm"])
    line("HONEST exec + gated induction", rows["honest_exec_confirm_gated_induction"])

    print("\n---- honest executor's own reliability (the signal the fix now rests on) ----")
    print(f"agreement with correctness: {cal['agreement_pct']}%   "
          f"confirms-wrong (FP, residual silent source): {cal['confirms_wrong_action_fp']}   "
          f"rejects-correct (FN, over-defer): {cal['rejects_correct_action_fn']}")

    full = rows["honest_exec_confirm_gated_induction"]
    print("\n---- residual silent-failure inputs under HONEST + gated induction ----")
    if full["silent_unique_inputs"]:
        for t, c in sorted(full["silent_unique_inputs"].items(), key=lambda kv: -kv[1]):
            print(f"   x{c:<4} {t!r:45} teacher={teacher(t)['tool']:14} gt={gt(t)['tool']}")
    else:
        print("   (none)")
    print(f"\nSaved: {outpath}")
    return report


if __name__ == "__main__":
    main()
