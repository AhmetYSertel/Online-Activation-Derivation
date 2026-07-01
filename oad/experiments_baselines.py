"""
experiments_baselines.py -- ROADMAP Phase C3: the head-to-head that turns OAD's
cost claim from "cheaper than always-call" (trivial) into a cost-SAFETY comparison
against its real peer, a semantic cache.

Axis: silent-failure rate vs model-call rate. The semantic cache traces a curve over
its similarity threshold; OAD is a single operating point. On slot-varying traffic +
real BFCL irrelevance, the cache serves stale parameters and mis-serves irrelevance
(silent failures) where OAD's slot induction + gate do not.

Runs offline with the oracle stand-in teacher, so the ONLY errors measured are the
serving mechanism's own (not teacher errors) -- the cleanest structural comparison.
Needs sentence-transformers (MiniLM); fully local after the model downloads.

    python -m oad.experiments_baselines --seeds 6
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import Engine
from .bfcl import (BFCLWorkload, build_families, load_irrelevance, bfcl_score,
                   OracleBFCLTeacher, DECLINE)
from .baselines import MiniLMEncoder, run_semantic_cache, run_exact_cache
from .multiseed import ci95

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.join(OUT, "figs", "baselines")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; BAD = "#a01b1b"; BLUE = "#2c5f8a"; GRID = "#d9d9d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

THRESHOLDS = [0.55, 0.65, 0.72, 0.78, 0.84, 0.90, 0.95]


def oad_point(stream, teacher, gt) -> Dict[str, float]:
    eng = Engine()
    ref = lambda x, p: bfcl_score(p, gt(x))
    acted = acted_ok = ab = 0
    for t in stream:
        r = eng.step(t, teacher, ref, execute=None)
        if r.decision == "act":
            acted += 1; acted_ok += bool(r.harness_correct)
        else:
            ab += 1
    n = len(stream)
    return {"call_rate": 100.0 * ab / n, "silent_rate": 100.0 * (acted - acted_ok) / n,
            "accuracy": 100.0 * (acted_ok + (ab - 0)) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    ap.add_argument("--fillers", type=int, default=40)
    args = ap.parse_args()

    fams = build_families(fillers_per_family=args.fillers); irr = load_irrelevance()
    wl0 = BFCLWorkload(seed=0, families=fams, irrelevance=irr)
    gt = wl0.ground_truth
    teacher = OracleBFCLTeacher(wl0)
    enc = MiniLMEncoder()
    print(f"families {len(fams)}  fillers/family {args.fillers}  "
          f"unique universe {len(wl0._by_text)}")

    # OAD point + exact-cache reference (multi-seed)
    oad_cr, oad_sr = [], []
    exact_cr, exact_sr = [], []
    cache_cr = {th: [] for th in THRESHOLDS}
    cache_sr = {th: [] for th in THRESHOLDS}
    for s in range(1, args.seeds + 1):
        wl = BFCLWorkload(seed=s, families=fams, irrelevance=irr)
        stream = wl.stream(args.n, junk_frac=args.junk_frac)
        p = oad_point(stream, teacher, gt)
        oad_cr.append(p["call_rate"]); oad_sr.append(p["silent_rate"])
        score = lambda x, a: bfcl_score(a, gt(x))
        e = run_exact_cache(stream, teacher, score)
        exact_cr.append(e["call_rate"]); exact_sr.append(e["silent_rate"])
        for th in THRESHOLDS:
            r = run_semantic_cache(stream, teacher, score, th, enc)
            cache_cr[th].append(r["call_rate"]); cache_sr[th].append(r["silent_rate"])

    oad = {"call_rate": vars(ci95(oad_cr)), "silent_rate": vars(ci95(oad_sr))}
    exact = {"call_rate": vars(ci95(exact_cr)), "silent_rate": vars(ci95(exact_sr))}
    curve = [{"threshold": th, "call_rate": vars(ci95(cache_cr[th])),
              "silent_rate": vars(ci95(cache_sr[th]))} for th in THRESHOLDS]

    # ---- Pareto figure ----
    cx = [c["call_rate"]["mean"] for c in curve]
    cy = [c["silent_rate"]["mean"] for c in curve]
    cyl = [c["silent_rate"]["lo"] for c in curve]
    cyh = [c["silent_rate"]["hi"] for c in curve]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(cx, cy, color=BLUE, lw=1.8, marker="o", ms=4, label="semantic cache (MiniLM), threshold swept")
    ax.fill_between(cx, cyl, cyh, color=BLUE, alpha=0.15)
    for c in curve:
        ax.annotate(f"{c['threshold']:.2f}", (c["call_rate"]["mean"], c["silent_rate"]["mean"]),
                    fontsize=7, color=BLUE, xytext=(3, 4), textcoords="offset points")
    ax.errorbar(oad["call_rate"]["mean"], oad["silent_rate"]["mean"],
                xerr=oad["call_rate"]["ci"], yerr=oad["silent_rate"]["ci"],
                color=ACC, marker="*", ms=15, lw=1.5, capsize=3, label="OAD (full design)")
    ax.errorbar(exact["call_rate"]["mean"], exact["silent_rate"]["mean"],
                xerr=exact["call_rate"]["ci"], yerr=exact["silent_rate"]["ci"],
                color=BAD, marker="D", ms=8, lw=1.5, capsize=3, label="exact-match cache")
    ax.set_xlabel("model-call rate (%)")
    ax.set_ylabel("silent-failure rate (%)")
    ax.set_title("Phase C3: cost-safety, OAD vs semantic cache (BFCL-grounded stream)",
                 loc="left", fontsize=10.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    figpath = os.path.join(FIGDIR, "pareto_cost_safety.png")
    fig.savefig(figpath); plt.close(fig)

    report = {"seeds": args.seeds, "n": args.n, "junk_frac": args.junk_frac,
              "fillers_per_family": args.fillers,
              "oad": oad, "exact_cache": exact, "semantic_cache_curve": curve,
              "figure": figpath}
    with open(os.path.join(OUT, "baselines_pareto.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n==== cost-safety Pareto ({args.seeds} seeds, {args.fillers} fillers/family) ====")
    print(f"OAD:         call {oad['call_rate']['mean']:5.1f}+/-{oad['call_rate']['ci']:.1f}%  "
          f"silent {oad['silent_rate']['mean']:.2f}+/-{oad['silent_rate']['ci']:.2f}%")
    print(f"exact cache: call {exact['call_rate']['mean']:5.1f}+/-{exact['call_rate']['ci']:.1f}%  "
          f"silent {exact['silent_rate']['mean']:.2f}%")
    print("semantic cache (threshold: call% / silent%):")
    for c in curve:
        print(f"  {c['threshold']:.2f}:  call {c['call_rate']['mean']:5.1f}%  "
              f"silent {c['silent_rate']['mean']:5.2f}%")
    # cache silent rate at the call rate closest to OAD's
    oc = oad["call_rate"]["mean"]
    nearest = min(curve, key=lambda c: abs(c["call_rate"]["mean"] - oc))
    print(f"\nAt ~OAD's call rate ({oc:.1f}%), nearest cache point (th={nearest['threshold']:.2f}, "
          f"call {nearest['call_rate']['mean']:.1f}%) has silent {nearest['silent_rate']['mean']:.2f}% "
          f"vs OAD {oad['silent_rate']['mean']:.2f}%")
    print(f"\nFigure: {figpath}")
    return report


if __name__ == "__main__":
    main()
