"""
experiments_honest_exec_multiseed.py -- ROADMAP Phase A3 applied to the A2 table.

Runs the confirmation-channel comparison (teacher-backed tautology / oracle-backed /
honest ground-truth-free) across many seeds and reports mean +/- 95% CI, so the
single-seed numbers in honest_exec_results.json become defensible. Also reports the
PAIRED silent-failure reduction (baseline -> honest+gated) with a CI.

Fully offline: every seed's stream stays within the packaged 51-input GPT-5.5 cache
(verified -- the cache covers the entire input universe). No API key, no network.

    python -m oad.experiments_honest_exec_multiseed --seeds 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Callable, Dict, List

from .engine import Engine
from .workload import Workload
from .teacher import make_execute
from .execution import ToolEnvironment, make_honest_execute
from .experiments_real_teacher import _mixed_stream
from .experiments_honest_exec import FileCachedTeacher, run
from .multiseed import ci95, paired_ci95

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")

CONFIGS = ["baseline_teacher_backed", "oracle_exec_confirm",
           "oracle_exec_confirm_gated_induction", "honest_exec_confirm",
           "honest_exec_confirm_gated_induction"]
METRICS = ["silent_failures", "llm_rate", "accuracy", "silent_oad_added_b"]


def run_one_seed(seed: int, teacher, n: int, junk: float, env: ToolEnvironment
                 ) -> Dict[str, Dict[str, float]]:
    wl = Workload(seed=seed)
    gt = wl.ground_truth
    stream = _mixed_stream(wl, n, junk, seed)
    taut = lambda: None
    orc = lambda: make_execute(gt, eps=0.0, rng=random.Random(seed + 777))
    hon = lambda: make_honest_execute(env, eps=0.0, rng=random.Random(seed + 777))
    out = {}
    out["baseline_teacher_backed"] = run(lambda: Engine(), stream, teacher, gt, taut)
    out["oracle_exec_confirm"] = run(lambda: Engine(), stream, teacher, gt, orc)
    out["oracle_exec_confirm_gated_induction"] = run(
        lambda: Engine(confirm_learning=True), stream, teacher, gt, orc)
    out["honest_exec_confirm"] = run(lambda: Engine(), stream, teacher, gt, hon)
    out["honest_exec_confirm_gated_induction"] = run(
        lambda: Engine(confirm_learning=True), stream, teacher, gt, hon)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    args = ap.parse_args()

    cache_path = os.path.join(OUT, f"_teacher_cache_{args.model}.json")
    teacher = FileCachedTeacher(args.model, cache_path)
    env = ToolEnvironment()

    per_seed: Dict[str, List[Dict[str, float]]] = {c: [] for c in CONFIGS}
    misses = 0
    for s in range(1, args.seeds + 1):
        res = run_one_seed(s, teacher, args.n, args.junk_frac, env)
        misses += len(teacher.misses); teacher.misses.clear()
        for c in CONFIGS:
            per_seed[c].append(res[c])
    if misses:
        print(f"WARNING: {misses} cache misses across seeds (not fully offline)")

    summary = {}
    for c in CONFIGS:
        summary[c] = {m: vars(ci95([r[m] for r in per_seed[c]])) for m in METRICS}

    base_silent = [r["silent_failures"] for r in per_seed["baseline_teacher_backed"]]
    honest_silent = [r["silent_failures"]
                     for r in per_seed["honest_exec_confirm_gated_induction"]]
    oracle_silent = [r["silent_failures"]
                     for r in per_seed["oracle_exec_confirm_gated_induction"]]
    red_honest = paired_ci95(base_silent, honest_silent)
    red_oracle = paired_ci95(base_silent, oracle_silent)
    # residual that honest leaves but oracle removes (the schema-valid-wrong gap)
    gap = paired_ci95(honest_silent, oracle_silent)

    report = {
        "model": args.model, "seeds": args.seeds, "n": args.n,
        "junk_frac": args.junk_frac,
        "summary_mean_ci": summary,
        "paired_reductions": {
            "baseline_minus_honest_gated": vars(red_honest),
            "baseline_minus_oracle_gated": vars(red_oracle),
            "honest_gated_minus_oracle_gated_residual": vars(gap),
        },
    }
    outpath = os.path.join(OUT, "honest_exec_multiseed.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    def row(c):
        s = summary[c]
        return (f"{c:38} silent {s['silent_failures']['mean']:5.1f}+/-{s['silent_failures']['ci']:<4.1f} "
                f"llm {s['llm_rate']['mean']:5.1f}+/-{s['llm_rate']['ci']:<4.1f}% "
                f"acc {s['accuracy']['mean']:5.1f}% "
                f"b {s['silent_oad_added_b']['mean']:.2f}+/-{s['silent_oad_added_b']['ci']:.2f}")

    print(f"\n==== {args.seeds} seeds, mean +/- 95% CI ====")
    for c in CONFIGS:
        print(row(c))
    print("\n---- paired silent-failure reductions (95% CI) ----")
    print(f"  baseline -> honest+gated : {red_honest.fmt()} failures removed")
    print(f"  baseline -> oracle+gated : {red_oracle.fmt()} failures removed")
    print(f"  honest residual vs oracle: {gap.fmt()} (schema-valid-wrong, uncatchable by execution)")
    print(f"\nSaved: {outpath}")
    return report


if __name__ == "__main__":
    main()
