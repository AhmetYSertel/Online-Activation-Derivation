"""
Statistical rigor experiments -- the evidence a reviewer asks for beyond single-seed
point estimates. All run with the synthetic OracleTeacher (no API key needed); the
CAVEATS in workload.py / harness.py still apply (these quantify the MECHANISM's
variance and component contributions, not transfer to real traffic).

  R1  Multi-seed confidence intervals for the Phase 2 headline metrics.
  R2  Epsilon sweep of the self-poisoning scissors (gap vs execution-noise, safe vs
      ablation, with CIs) -- shows the asymmetry holds the gap ~0 across all noise.
  R3  Component ablations -- turn off each safeguard individually and measure the
      resulting silent failures / scissors gap, to attribute the safety.
  R4  Teacher-noise robustness -- silent-failure rate vs label noise delta of an
      imperfect (frozen-LLM-like) teacher, full design vs no-protection.

Run: python -m oad.experiments_rigor
"""
from __future__ import annotations

import json
import os
import random
import statistics
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .confirm import Thresholds
from .engine import Engine
from .harness import Referee
from .teacher import OracleTeacher, NoisyTeacher, make_execute
from .workload import Workload, default_families
from .experiments_phase2 import GT, drifted_workload, is_weather, _mixed_stream, _rolling_pair, LOOSE

OUT = "/mnt/user-data/outputs"
FIGDIR = os.path.join(OUT, "oad_release_figs/rigor")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; WARN = "#c25b3a"; BAD = "#a01b1b"; BLUE = "#2c5f8a"; GREY = "#8a8a8a"; GRID = "#d9d9d9"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.dpi": 130, "savefig.bbox": "tight"})

DRIFT_TOOLS = [f.tool for f in default_families()] + ["get_weather_v2"]


def ci95(xs: List[float]) -> Tuple[float, float, float]:
    """(mean, std, half-width of 95% CI)."""
    n = len(xs)
    m = statistics.fmean(xs)
    if n < 2:
        return m, 0.0, 0.0
    sd = statistics.stdev(xs)
    return m, sd, 1.96 * sd / (n ** 0.5)


# --------------------------------------------------------------- lean runners
def run_cost(seed, n, junk_frac, engine_factory,
             teacher_maker: Callable = OracleTeacher) -> Dict[str, float]:
    wl = Workload(seed=seed); gt = GT(wl.ground_truth)
    te = teacher_maker(gt); ref = Referee(gt)
    stream = _mixed_stream(wl, n, junk_frac, seed)
    eng = engine_factory()
    ac = aw = ab = 0
    for t in stream:
        r = eng.step(t, te, ref.score, execute=None)
        if r.decision == "act":
            ac += int(bool(r.harness_correct)); aw += int(not r.harness_correct)
        else:
            ab += 1
    return {"llm_rate": 100.0 * ab / n, "accuracy": 100.0 * (ac + ab) / n,
            "silent_rate": 100.0 * aw / n, "silent": aw}


def run_scissors_gap(seed, n, drift_at, eps, engine_factory, delta=0.0) -> float:
    wl = Workload(seed=seed); wl_d = drifted_workload(seed)
    gt = GT(wl.ground_truth)
    base = OracleTeacher(gt)
    te = base if delta == 0 else NoisyTeacher(base, delta, random.Random(seed + 5), tools=DRIFT_TOOLS)
    ref = Referee(gt)
    execute = make_execute(gt, eps, random.Random(seed + 777))
    eng = engine_factory()
    pts = []
    for i, t in enumerate(wl.stream(n, zipf_s=0.6)):
        if i == drift_at:
            gt.fn = wl_d.ground_truth
        r = eng.step(t, te, ref.score, execute=execute)
        if r.decision == "act" and is_weather(r.action) and r.acting_post_mean is not None:
            pts.append((i, r.acting_post_mean, 1 if r.harness_correct else 0))
    xs, bel, tru = _rolling_pair(pts)
    late = drift_at + (n - drift_at) // 2
    g = [b - t for (x, b, t) in zip(xs, bel, tru) if x > late]
    return sum(g) / len(g) if g else 0.0


def run_drift_recovery(seed, n, drift_at) -> float:
    wl = Workload(seed=seed); wl_d = drifted_workload(seed)
    gt = GT(wl.ground_truth)
    te = OracleTeacher(gt); ref = Referee(gt)
    eng = Engine(post_decay=0.99)
    w_correct = []
    recovered = None
    for i, t in enumerate(wl.stream(n, zipf_s=0.6)):
        if i == drift_at:
            gt.fn = wl_d.ground_truth
        r = eng.step(t, te, ref.score, execute=None)
        if is_weather(gt(t)):
            w_correct.append(1 if r.harness_correct else 0)
        if (i > drift_at + 30 and recovered is None and len(w_correct) >= 30
                and sum(w_correct[-30:]) / 30 >= 0.95):
            recovered = i - drift_at
    return float(recovered) if recovered is not None else float("nan")


SAFE = lambda: Engine(post_decay=0.99)
ABLATION = lambda: Engine(thr=LOOSE, self_train=True, use_gate=False, k_slot=1, post_decay=0.99)


# ===================================================================== R1
def r1_multiseed_ci(seeds=range(10)):
    cost_oad, cost_eager_silent, recov, gap_safe, gap_unsafe = [], [], [], [], []
    for s in seeds:
        oad = run_cost(s, 3000, 0.15, lambda: Engine())
        eager = run_cost(s, 3000, 0.15, lambda: Engine(thr=LOOSE, use_gate=False, k_slot=1))
        cost_oad.append(oad); cost_eager_silent.append(eager["silent_rate"])
        recov.append(run_drift_recovery(s, 5000, 2500))
        gap_safe.append(run_scissors_gap(s, 5000, 2500, 0.6, SAFE))
        gap_unsafe.append(run_scissors_gap(s, 5000, 2500, 0.6, ABLATION))

    def agg(key): return ci95([d[key] for d in cost_oad])
    res = {
        "n_seeds": len(list(seeds)),
        "oad_accuracy_pct": agg("accuracy"),
        "oad_llm_rate_pct": agg("llm_rate"),
        "oad_silent_rate_pct": agg("silent_rate"),
        "eager_silent_rate_pct": ci95(cost_eager_silent),
        "drift_recovery_calls": ci95([x for x in recov if x == x]),
        "scissors_gap_safe_pct": ci95(gap_safe),
        "scissors_gap_ablation_pct": ci95(gap_unsafe),
    }

    # figure: two panels with error bars
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    labels = ["OAD\naccuracy", "OAD\nsilent", "eager\nsilent"]
    means = [res["oad_accuracy_pct"][0], res["oad_silent_rate_pct"][0], res["eager_silent_rate_pct"][0]]
    errs = [res["oad_accuracy_pct"][2], res["oad_silent_rate_pct"][2], res["eager_silent_rate_pct"][2]]
    axA.bar(labels, means, yerr=errs, capsize=4, color=[ACC, BAD, BAD])
    axA.set_ylabel("%"); axA.set_title("R1  cost/quality (95% CI)", loc="left")
    for i, (m, e) in enumerate(zip(means, errs)):
        axA.text(i, m + max(errs) + 1, f"{m:.1f}±{e:.1f}", ha="center", fontsize=8)

    g_labels = ["scissors gap\nSAFE", "scissors gap\nABLATION"]
    g_means = [res["scissors_gap_safe_pct"][0], res["scissors_gap_ablation_pct"][0]]
    g_errs = [res["scissors_gap_safe_pct"][2], res["scissors_gap_ablation_pct"][2]]
    axB.bar(g_labels, g_means, yerr=g_errs, capsize=4, color=[ACC, BAD])
    axB.set_ylabel("believed - true accuracy (pts)")
    axB.set_title("R1  self-poisoning gap (95% CI)", loc="left")
    for i, (m, e) in enumerate(zip(g_means, g_errs)):
        axB.text(i, m + abs(max(g_means)) * 0.05 + 1, f"{m:.1f}±{e:.1f}", ha="center", fontsize=8)
    fig.tight_layout(); path = os.path.join(FIGDIR, "r1_multiseed_ci.png")
    fig.savefig(path); plt.close(fig)
    res["figure"] = path
    return res


# ===================================================================== R2
def r2_epsilon_sweep(seeds=range(6), epsilons=(0.0, 0.2, 0.4, 0.6, 0.8)):
    safe_m, safe_e, un_m, un_e = [], [], [], []
    for eps in epsilons:
        s_gaps = [run_scissors_gap(s, 3000, 1500, eps, SAFE) for s in seeds]
        u_gaps = [run_scissors_gap(s, 3000, 1500, eps, ABLATION) for s in seeds]
        m, _, h = ci95(s_gaps); safe_m.append(m); safe_e.append(h)
        m, _, h = ci95(u_gaps); un_m.append(m); un_e.append(h)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ex = list(epsilons)
    ax.plot(ex, un_m, color=BAD, marker="o", lw=1.8, label="ABLATION (eager+self-train+no-gate)")
    ax.fill_between(ex, [m - e for m, e in zip(un_m, un_e)], [m + e for m, e in zip(un_m, un_e)],
                    color=BAD, alpha=0.15)
    ax.plot(ex, safe_m, color=ACC, marker="o", lw=1.8, label="SAFE (full design)")
    ax.fill_between(ex, [m - e for m, e in zip(safe_m, safe_e)], [m + e for m, e in zip(safe_m, safe_e)],
                    color=ACC, alpha=0.15)
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xlabel("execution noise  eps  (wrong action confirmed w.p. eps)")
    ax.set_ylabel("late post-drift gap (believed - true, pts)")
    ax.set_title(f"R2  scissors gap vs execution noise ({len(list(seeds))} seeds, 95% CI)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); path = os.path.join(FIGDIR, "r2_epsilon_sweep.png")
    fig.savefig(path); plt.close(fig)
    return {"epsilons": list(epsilons), "safe_gap_mean": safe_m, "safe_gap_ci": safe_e,
            "ablation_gap_mean": un_m, "ablation_gap_ci": un_e, "figure": path}


# ===================================================================== R3
def r3_ablations(seeds=range(8)):
    # Panel A: silent failures on junk traffic -- gate & specificity contributions.
    cfgsA = {
        "full design": lambda: Engine(),
        "no gate": lambda: Engine(thr=LOOSE, use_gate=False, k_slot=1),
        "fall-through on": lambda: Engine(allow_fallthrough=True),
    }
    A = {name: [run_cost(s, 3000, 0.15, fac)["silent_rate"] for s in seeds]
         for name, fac in cfgsA.items()}

    # Panel B: scissors gap on drift+noise -- decay, acting bar, self-train, full ablation.
    cfgsB = {
        "full design": SAFE,
        "no decay": lambda: Engine(post_decay=1.0),
        "eager acting": lambda: Engine(thr=LOOSE, k_slot=1, post_decay=0.99),  # gate ON, self_train OFF
        "self-train on": lambda: Engine(post_decay=0.99, self_train=True),
        "full ablation": ABLATION,
    }
    B = {name: [run_scissors_gap(s, 4000, 2000, 0.6, fac) for s in seeds]
         for name, fac in cfgsB.items()}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    la = list(A.keys()); ma = [ci95(A[k])[0] for k in la]; ea = [ci95(A[k])[2] for k in la]
    axA.bar(la, ma, yerr=ea, capsize=4, color=[ACC, BAD, BAD])
    axA.set_ylabel("silent-failure rate on junk (%)")
    axA.set_title("R3a  gate & specificity", loc="left")
    for i, (m, e) in enumerate(zip(ma, ea)):
        axA.text(i, m + max(ea) + 0.3, f"{m:.1f}", ha="center", fontsize=8)

    lb = list(B.keys()); mb = [ci95(B[k])[0] for k in lb]; eb = [ci95(B[k])[2] for k in lb]
    axB.bar(lb, mb, yerr=eb, capsize=4, color=[ACC, WARN, BAD, WARN, BAD])
    axB.set_ylabel("self-poisoning gap (pts)")
    axB.set_title("R3b  decay & learning asymmetry", loc="left")
    for i, (m, e) in enumerate(zip(mb, eb)):
        axB.text(i, m + max(eb) + 0.3, f"{m:.1f}", ha="center", fontsize=8)
    plt.setp(axB.get_xticklabels(), fontsize=8)
    fig.tight_layout(); path = os.path.join(FIGDIR, "r3_ablations.png")
    fig.savefig(path); plt.close(fig)
    return {"silent_on_junk": {k: ci95(v) for k, v in A.items()},
            "scissors_gap": {k: ci95(v) for k, v in B.items()}, "figure": path}


# ===================================================================== R4
def r4_teacher_noise(seeds=range(6), deltas=(0.0, 0.1, 0.2, 0.3)):
    # silent-failure rate on clean (no-drift) junk traffic as the teacher's label noise
    # grows. Full design (maturity + specificity reject sparse bad labels) vs eager/no-gate.
    safe_m, safe_e, un_m, un_e = [], [], [], []
    for d in deltas:
        tmk = (lambda gt, _d=d: NoisyTeacher(OracleTeacher(gt), _d, random.Random(13), tools=DRIFT_TOOLS))
        s = [run_cost(seed, 3000, 0.15, lambda: Engine(), teacher_maker=tmk)["silent_rate"] for seed in seeds]
        u = [run_cost(seed, 3000, 0.15, lambda: Engine(thr=LOOSE, use_gate=False, k_slot=1),
                      teacher_maker=tmk)["silent_rate"] for seed in seeds]
        m, _, h = ci95(s); safe_m.append(m); safe_e.append(h)
        m, _, h = ci95(u); un_m.append(m); un_e.append(h)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    dx = list(deltas)
    ax.plot(dx, un_m, color=BAD, marker="o", lw=1.8, label="eager + no gate")
    ax.fill_between(dx, [m - e for m, e in zip(un_m, un_e)], [m + e for m, e in zip(un_m, un_e)], color=BAD, alpha=0.15)
    ax.plot(dx, safe_m, color=ACC, marker="o", lw=1.8, label="full design")
    ax.fill_between(dx, [m - e for m, e in zip(safe_m, safe_e)], [m + e for m, e in zip(safe_m, safe_e)], color=ACC, alpha=0.15)
    ax.set_xlabel("teacher label-noise  delta  (imperfect frozen LLM)")
    ax.set_ylabel("silent-failure rate (%)")
    ax.set_title(f"R4  robustness to teacher noise ({len(list(seeds))} seeds, 95% CI)", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); path = os.path.join(FIGDIR, "r4_teacher_noise.png")
    fig.savefig(path); plt.close(fig)
    return {"deltas": list(deltas), "safe_silent_mean": safe_m, "safe_silent_ci": safe_e,
            "noprotect_silent_mean": un_m, "noprotect_silent_ci": un_e, "figure": path}


def _fmt(t):  # (mean, std, ci) -> "mean ± ci"
    return f"{t[0]:.1f} ± {t[2]:.1f}" if isinstance(t, (list, tuple)) else str(t)


def main():
    print("R1  multi-seed confidence intervals...")
    r1 = r1_multiseed_ci()
    for k, v in r1.items():
        if k != "figure":
            print(f"  {k}: {_fmt(v) if isinstance(v,(list,tuple)) else v}")
    print()

    print("R2  epsilon sweep (scissors gap vs execution noise)...")
    r2 = r2_epsilon_sweep(seeds=range(5))
    print("  eps:", r2["epsilons"])
    print("  SAFE gap:    ", [f"{m:.1f}" for m in r2["safe_gap_mean"]])
    print("  ABLATION gap:", [f"{m:.1f}" for m in r2["ablation_gap_mean"]], "\n")

    print("R3  component ablations...")
    r3 = r3_ablations(seeds=range(6))
    print("  silent on junk:", {k: f"{v[0]:.1f}" for k, v in r3["silent_on_junk"].items()})
    print("  scissors gap:  ", {k: f"{v[0]:.1f}" for k, v in r3["scissors_gap"].items()}, "\n")

    print("R4  teacher-noise robustness...")
    r4 = r4_teacher_noise(seeds=range(5))
    print("  delta:", r4["deltas"])
    print("  full-design silent:", [f"{m:.1f}" for m in r4["safe_silent_mean"]])
    print("  no-protect  silent:", [f"{m:.1f}" for m in r4["noprotect_silent_mean"]], "\n")

    results = {"R1": r1, "R2": r2, "R3": r3, "R4": r4}
    with open(os.path.join(OUT, "oad_release_figs/rigor_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved figures to", FIGDIR)
    print("Saved metrics to", os.path.join(OUT, "oad_release_figs/rigor_results.json"))
    return results


if __name__ == "__main__":
    main()
