"""
experiments_honest_exec_epssweep.py -- ROADMAP Phase D1.

The honest fix's 0-residual safety was shown under a CLEAN executor (eps=0). This
sweeps the execution channel's noise eps: with probability eps a schema-INVALID call
"succeeds" anyway (a tool that returns a plausible-but-wrong result instead of
erroring). It quantifies how much executor dishonesty the safety tolerates and how
fast it degrades back toward the no-protection baseline.

At eps=0 the executor is clean (-> ~27 residual silent failures, 20-seed). At eps=1
every invalid call is confirmed, so the signal carries no information and the system
regresses toward the teacher-backed tautology (~110). The curve between is the
"how honest must the executor be" answer.

Fully offline from the packaged 51-input GPT-5.5 cache. Reusable CIs via multiseed.

    python -m oad.experiments_honest_exec_epssweep --seeds 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import Engine
from .workload import Workload
from .execution import ToolEnvironment, make_honest_execute
from .experiments_real_teacher import _mixed_stream
from .experiments_honest_exec import FileCachedTeacher, run
from .multiseed import ci95

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.join(OUT, "figs", "honest_exec")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; BAD = "#a01b1b"; BLUE = "#2c5f8a"; GRID = "#d9d9d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    args = ap.parse_args()

    teacher = FileCachedTeacher(args.model,
                                os.path.join(OUT, f"_teacher_cache_{args.model}.json"))
    env = ToolEnvironment()

    rows = []
    for eps in EPS_GRID:
        sf, lr = [], []
        for s in range(1, args.seeds + 1):
            wl = Workload(seed=s); gt = wl.ground_truth
            stream = _mixed_stream(wl, args.n, args.junk_frac, s)
            hon = lambda: make_honest_execute(env, eps=eps,
                                              rng=random.Random(s + 777))
            r = run(lambda: Engine(confirm_learning=True), stream, teacher, gt, hon)
            sf.append(100.0 * r["silent_failures"] / args.n)  # rate %
            lr.append(r["llm_rate"])
        rows.append({"eps": eps, "silent_rate": vars(ci95(sf)), "call_rate": vars(ci95(lr))})
        st = ci95(sf)
        print(f"eps={eps:<4}  silent-rate {st.mean:5.2f} +/- {st.ci:4.2f}%   "
              f"call-rate {ci95(lr).mean:5.1f}%")

    # ---- figure: silent-failure rate vs eps, with CI band + call-rate line ----
    xs = [r["eps"] for r in rows]
    sm = [r["silent_rate"]["mean"] for r in rows]
    slo = [r["silent_rate"]["lo"] for r in rows]
    shi = [r["silent_rate"]["hi"] for r in rows]
    cm = [r["call_rate"]["mean"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, sm, color=BAD, lw=1.9, marker="o", ms=4, label="silent-failure rate")
    ax.fill_between(xs, slo, shi, color=BAD, alpha=0.15)
    ax.axhline(110.1 / args.n * 100, color="#999", ls=":", lw=1.2,
               label="teacher-backed baseline (~7.3%)")
    ax.set_xlabel("executor noise  eps  (P[invalid call confirmed])")
    ax.set_ylabel("silent-failure rate (%)", color=BAD)
    ax.tick_params(axis="y", labelcolor=BAD)
    ax.set_title("Phase D1: honest fix degrades as the executor becomes dishonest",
                 loc="left", fontsize=10.5)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(xs, cm, color=BLUE, lw=1.5, marker="s", ms=3, ls="--", label="call rate")
    ax2.set_ylabel("model-call rate (%)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="center right")
    fig.tight_layout()
    figpath = os.path.join(FIGDIR, "honest_exec_eps_sweep.png")
    fig.savefig(figpath); plt.close(fig)

    report = {"model": args.model, "seeds": args.seeds, "n": args.n,
              "junk_frac": args.junk_frac, "eps_grid": EPS_GRID, "rows": rows,
              "figure": figpath}
    outpath = os.path.join(OUT, "honest_exec_epssweep.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFigure: {figpath}\nSaved:  {outpath}")
    return report


if __name__ == "__main__":
    main()
