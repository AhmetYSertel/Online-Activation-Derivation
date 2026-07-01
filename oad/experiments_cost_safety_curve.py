"""
Point 3: the cost-safety tradeoff is parameterized by the BOUNDARY RATE. Sweep the
junk fraction (surface-ambiguous / out-of-domain share of traffic) and trace, for the
teacher-backed baseline vs the independent-signal fix:

  * silent-failure rate  -- the safety axis
  * LLM-call rate         -- the cost axis

As boundary traffic grows, the baseline's silent failures climb (it acts confidently on
inputs the teacher mislabels); the fix holds silent failures at 0 by paying more
deferral. The gap between the two LLM-rate curves is the *price of safety*, and it grows
with the boundary rate.

Also decompose the fix's abstentions, to separate two very different costs:
  * genuine-boundary defer -- gt == fallback. Correct to defer; not a precision cost.
  * over-cautious defer     -- gt != fallback (a legitimate input the teacher would get
                              right) that the system nonetheless declined. THIS is the
                              precision cost of the fix. With a clean executor and a
                              teacher that is right on in-domain inputs it should be ~0
                              (only first-encounter learning abstentions); it grows under
                              executor noise / a teacher wrong on legitimate inputs.

Free to run: the unique-input set is fixed across junk fractions, so the disk-cached
teacher serves every point with 0 new API calls.

Run:
    export OPENAI_API_KEY=sk-...
    python -m oad.experiments_cost_safety_curve --model gpt-5.5 --n 1500 --seed 1
"""
from __future__ import annotations

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import Engine
from .workload import Workload, default_families
from .teacher import OpenAITeacher, make_execute
from .analyze_silent_failures import DiskCachingTeacher
from .experiments_real_teacher import _mixed_stream

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.join(OUT, "figs", "real_teacher")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; BAD = "#a01b1b"; BLUE = "#2c5f8a"; GREY = "#8a8a8a"; GRID = "#d9d9d9"; WARN = "#c25b3a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def run_point(confirm_learning, stream, teacher, gt, independent):
    eng = Engine(confirm_learning=confirm_learning)
    referee = lambda x, p: p == gt(x)
    execute = (make_execute(gt, 0.0, random.Random(99)) if confirm_learning else None)
    n = len(stream)
    acted = acted_correct = abstained = 0
    abstain_genuine = abstain_over = 0
    for t in stream:
        is_boundary = gt(t).get("tool", "") == "__llm_other__"   # gt == fallback
        r = eng.step(t, teacher, referee, execute=execute)
        if r.decision == "act":
            acted += 1
            acted_correct += 1 if r.harness_correct else 0
        else:
            abstained += 1
            if is_boundary:
                abstain_genuine += 1
            else:
                abstain_over += 1        # legitimate input we declined -> precision cost
    return {
        "llm_rate": round(100.0 * abstained / n, 2),
        "silent_rate": round(100.0 * (acted - acted_correct) / n, 2),
        "silent_failures": acted - acted_correct,
        "abstain_genuine_boundary": abstain_genuine,
        "abstain_over_cautious": abstain_over,
        "over_cautious_rate": round(100.0 * abstain_over / n, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    wl = Workload(seed=args.seed)
    gt = wl.ground_truth
    cache_path = os.path.join(OUT, f"_teacher_cache_{args.model}.json")
    teacher = DiskCachingTeacher(OpenAITeacher(model=args.model, families=default_families()),
                                 args.model, cache_path)
    # warm cache over the full junk + in-domain universe once
    warm = _mixed_stream(wl, args.n, 1.0, args.seed) + _mixed_stream(wl, args.n, 0.0, args.seed)
    for t in sorted(set(warm)):
        teacher(t)
    print(f"cache warm; new API calls {teacher.api_calls}")

    fracs = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
    rows = []
    for jf in fracs:
        stream = _mixed_stream(wl, args.n, jf, args.seed)
        base = run_point(False, stream, teacher, gt, None)
        fix = run_point(True, stream, teacher, gt, True)
        rows.append({"junk_frac": jf, "baseline": base, "fixed": fix})
        print(f"jf {jf:.2f}  baseline: silent {base['silent_rate']:5}% llm {base['llm_rate']:5}%  "
              f"| fixed: silent {fix['silent_rate']:4}% llm {fix['llm_rate']:5}% "
              f"over-cautious {fix['over_cautious_rate']:4}%")

    # ---- figure: safety (left) and cost (right) vs boundary rate ----
    xs = [r["junk_frac"] * 100 for r in rows]
    fig, (axS, axC) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    axS.plot(xs, [r["baseline"]["silent_rate"] for r in rows], color=BAD, marker="o",
             lw=1.8, label="baseline (teacher-backed)")
    axS.plot(xs, [r["fixed"]["silent_rate"] for r in rows], color=ACC, marker="s",
             lw=1.8, label="fixed (independent signal)")
    axS.set_xlabel("boundary / junk rate (%)"); axS.set_ylabel("silent-failure rate (%)")
    axS.set_title("Safety: silent failures vs boundary rate", loc="left")
    axS.legend(fontsize=8)

    axC.plot(xs, [r["baseline"]["llm_rate"] for r in rows], color=GREY, marker="o",
             lw=1.6, ls="--", label="baseline LLM-rate")
    axC.plot(xs, [r["fixed"]["llm_rate"] for r in rows], color=BLUE, marker="s",
             lw=1.8, label="fixed LLM-rate (cost)")
    axC.plot(xs, [r["fixed"]["over_cautious_rate"] for r in rows], color=WARN,
             marker="^", lw=1.4, ls=":", label="fixed over-cautious defer (precision cost)")
    axC.set_xlabel("boundary / junk rate (%)"); axC.set_ylabel("rate (%)")
    axC.set_title("Cost: deferral price of safety", loc="left")
    axC.legend(fontsize=8)
    fig.suptitle(f"Cost-safety tradeoff vs boundary rate ({args.model})", x=0.5, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    figpath = os.path.join(FIGDIR, "cost_safety_curve.png")
    fig.savefig(figpath); plt.close(fig)

    outpath = os.path.join(OUT, "cost_safety_curve.json")
    with open(outpath, "w") as f:
        json.dump({"model": args.model, "n": args.n, "seed": args.seed, "sweep": rows,
                   "figure": figpath}, f, indent=2, default=str)
    print(f"\nFigure:  {figpath}\nMetrics: {outpath}")


if __name__ == "__main__":
    main()
