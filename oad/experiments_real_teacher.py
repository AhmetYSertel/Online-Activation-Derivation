"""
Real-teacher transfer run -- the experiment LIMITATIONS.md #1 calls for.

Every packaged result uses the synthetic OracleTeacher (perfect by construction).
This driver swaps in teacher.OpenAITeacher (a REAL GPT model, default gpt-5.5) and
re-runs the cost/coverage experiment, so the numbers reflect distilling an imperfect
frozen LLM rather than an oracle.

What changes versus the oracle run, and why it matters:
  * The teacher now makes real mistakes (tool-selection errors, accepting invalid
    slots, hallucinated parameters). So the abstain path is NO LONGER free of error:
    sending an input "to the LLM" can still produce a wrong action. We therefore score
    the teacher's action on every abstain too (honest accuracy), instead of assuming
    abstain -> correct the way the oracle-backed exp_cost did.
  * The referee is still backed by the workload ground truth (the workload is
    synthetic, so truth is known), but it is now INFORMATIONALLY independent of the
    teacher -- teacher!=referee. We report teacher<->referee agreement, which is < 1.0
    and meaningful (it is the real LLM's accuracy on this distribution).

Cost: the teacher is wrapped in a cache. The workload stream is Zipfian and the
families have small vocabularies, so only the unique inputs hit the API regardless of
stream length. The run reports how many real API calls were actually made.

Run:
    pip install openai
    export OPENAI_API_KEY=sk-...
    python -m oad.experiments_real_teacher                 # default gpt-5.5
    python -m oad.experiments_real_teacher --model gpt-5.4 # cheaper option
    python -m oad.experiments_real_teacher --n 1500 --junk-frac 0.15 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .confirm import Thresholds
from .engine import Engine
from .workload import Workload, default_families, Action
from .teacher import OpenAITeacher
from .harness import Referee

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.join(OUT, "figs", "real_teacher")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; BAD = "#a01b1b"; BLUE = "#2c5f8a"; GREY = "#8a8a8a"; GRID = "#d9d9d9"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight",
})

LOOSE = Thresholds(p_act=0.0, c_act=0.0)   # acts as soon as a match exists


class CachingTeacher:
    """Wraps any teacher with a per-input memo, so repeated inputs in the Zipfian
    stream cost nothing after the first call. Tracks how many real (uncached) calls
    were made -- that is the actual API spend."""

    def __init__(self, base):
        self.base = base
        self.cache: Dict[str, Action] = {}
        self.api_calls = 0

    def __call__(self, text: str) -> Action:
        if text not in self.cache:
            self.api_calls += 1
            self.cache[text] = self.base(text)
        return self.cache[text]


def _mixed_stream(wl: Workload, n: int, junk_frac: float, seed: int) -> List[str]:
    """Realistic traffic: valid family calls with a fraction of adversarial/OOD junk.
    Mirrors experiments_phase2._mixed_stream so the runs are comparable."""
    rng = random.Random(seed)
    base = wl.stream(n, zipf_s=0.6)
    junk = [t for t, _ in wl.adversarial_probes()] + wl.ood_probes()
    out = []
    for t in base:
        out.append(rng.choice(junk) if rng.random() < junk_frac else t)
    return out


def run_engine(engine: Engine, stream: List[str], teacher, referee: Callable) -> dict:
    """Drive the engine over the stream. Every produced action -- whether the engine
    acted (replay) or abstained (teacher) -- is scored by the referee against ground
    truth. This is the honest accounting for a fallible teacher."""
    acted = acted_correct = 0
    abstained = abstain_correct = 0
    llm_cumulative = []
    llm_calls = 0
    for i, t in enumerate(stream):
        r = engine.step(t, teacher, referee, execute=None)  # clean (teacher-confirmed) execution
        if r.decision == "act":
            acted += 1
            acted_correct += 1 if r.harness_correct else 0
        else:
            abstained += 1
            abstain_correct += 1 if r.harness_correct else 0
            llm_calls += 1
        llm_cumulative.append(100.0 * llm_calls / (i + 1))
    n = len(stream)
    return {
        "llm_rate": round(100.0 * abstained / n, 1),
        "accuracy": round(100.0 * (acted_correct + abstain_correct) / n, 1),
        "acted": acted,
        "acted_correct": acted_correct,
        "silent_failures": acted - acted_correct,   # engine acted and was wrong
        "abstained": abstained,
        "abstain_wrong": abstained - abstain_correct,  # teacher was asked and was wrong
        "llm_cumulative": llm_cumulative,
    }


def teacher_calibration(unique_inputs: List[str], teacher, gt: Callable) -> dict:
    """The real LLM's own accuracy on this distribution (teacher<->referee agreement),
    computed over the unique inputs. With the oracle this would be exactly 1.0; with a
    real model it is the headline transfer number."""
    agree = 0
    errors = []
    for t in unique_inputs:
        a = teacher(t)
        g = gt(t)
        if a == g:
            agree += 1
        elif len(errors) < 25:
            errors.append({"input": t, "teacher": a, "ground_truth": g})
    return {
        "n_unique": len(unique_inputs),
        "teacher_accuracy": round(100.0 * agree / max(1, len(unique_inputs)), 1),
        "errors_sample": errors,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    t0 = time.time()
    wl = Workload(seed=args.seed)
    gt = wl.ground_truth
    referee = Referee(gt).score
    stream = _mixed_stream(wl, args.n, args.junk_frac, args.seed)

    base = OpenAITeacher(model=args.model, families=default_families())
    teacher = CachingTeacher(base)

    print(f"Real-teacher transfer run: model={args.model} n={args.n} "
          f"junk_frac={args.junk_frac} seed={args.seed}")
    print("Warming teacher cache over unique inputs (this is the only API spend)...")
    unique_inputs = sorted(set(stream))
    cal = teacher_calibration(unique_inputs, teacher, gt)
    print(f"  unique inputs: {cal['n_unique']}   API calls so far: {teacher.api_calls}")
    print(f"  teacher raw accuracy vs ground truth: {cal['teacher_accuracy']}%")

    print("Running OAD (full design)...")
    oad = run_engine(Engine(), stream, teacher, referee)
    print("Running eager-no-gate baseline...")
    eager = run_engine(Engine(thr=LOOSE, use_gate=False, k_slot=1), stream, teacher, referee)

    # always-LLM: every input goes to the (real, fallible) teacher.
    always_correct = sum(1 for t in stream if teacher(t) == gt(t))
    always = {
        "llm_rate": 100.0,
        "accuracy": round(100.0 * always_correct / len(stream), 1),
        "silent_failures": 0,                 # never acts on its own; teacher errors are not "silent"
        "teacher_errors": len(stream) - always_correct,
        "llm_cumulative": [100.0] * len(stream),
    }

    # ---- figure: cost curve + cheap/safe bars (mirrors phase2 E_cost) ----
    n = len(stream)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    axA.plot(oad["llm_cumulative"], color=ACC, lw=1.8, label="OAD")
    axA.plot(always["llm_cumulative"], color=GREY, lw=1.3, ls="--", label="always-LLM")
    axA.plot(eager["llm_cumulative"], color=BAD, lw=1.3, ls=":", label="eager-no-gate")
    axA.set_ylim(-3, 103); axA.set_xlabel("call index")
    axA.set_ylabel("cumulative LLM-call rate (%)")
    axA.set_title(f"Real teacher ({args.model}): cost at held quality", loc="left")
    axA.legend(fontsize=8, loc="center right")

    names = ["always-LLM", "OAD", "eager-no-gate"]
    rows = [always, oad, eager]
    x = range(len(names)); w = 0.38
    axB.bar([i - w/2 for i in x], [r["llm_rate"] for r in rows], width=w,
            color=BLUE, label="LLM-call rate %")
    axB.bar([i + w/2 for i in x], [100.0 * r["silent_failures"] / n for r in rows], width=w,
            color=BAD, label="silent-failure rate %")
    axB.set_xticks(list(x)); axB.set_xticklabels(names, fontsize=8.5)
    axB.set_title("teacher is fallible now", loc="left")
    axB.legend(fontsize=8)
    for i, r in enumerate(rows):
        axB.text(i, max(r["llm_rate"], 100.0 * r["silent_failures"] / n) + 2,
                 f"acc {r['accuracy']:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    figpath = os.path.join(FIGDIR, "real_teacher_cost_quality.png")
    fig.savefig(figpath); plt.close(fig)

    for r in (oad, eager, always):
        r.pop("llm_cumulative", None)

    results = {
        "model": args.model,
        "n": args.n,
        "junk_frac": args.junk_frac,
        "seed": args.seed,
        "api_calls_made": teacher.api_calls,
        "teacher_calibration": cal,
        "oad": oad,
        "eager_no_gate": eager,
        "always_llm": always,
        "wall_clock_sec": round(time.time() - t0, 1),
        "figure": figpath,
    }
    outpath = os.path.join(OUT, "real_teacher_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n==== SUMMARY ====")
    print(f"API calls made (cached):     {teacher.api_calls}")
    print(f"teacher raw accuracy:        {cal['teacher_accuracy']}%  ({cal['n_unique']} unique inputs)")
    print(f"OAD:           llm_rate {oad['llm_rate']}%  acc {oad['accuracy']}%  "
          f"silent_failures {oad['silent_failures']}  abstain_wrong {oad['abstain_wrong']}")
    print(f"always-LLM:    llm_rate 100%   acc {always['accuracy']}%  teacher_errors {always['teacher_errors']}")
    print(f"eager-no-gate: llm_rate {eager['llm_rate']}%  acc {eager['accuracy']}%  "
          f"silent_failures {eager['silent_failures']}")
    print(f"\nFigure:  {figpath}")
    print(f"Metrics: {outpath}")
    return results


if __name__ == "__main__":
    main()
