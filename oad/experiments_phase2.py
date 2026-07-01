"""
Phase 2 -- what is it good for? A long *sequential* call stream (the batch is a
delivery mechanism, NOT a frozen-snapshot parallel eval: the system updates online
as it consumes the stream, exactly as it would live).

  E_cost      cost/coverage at held quality vs baselines (always-LLM, eager-no-gate).
  E_drift     distribution drift injected at a known point -> graceful degradation
              and online recovery (the posterior drops, the system abstains and
              re-derives the new action).
  E_scissors  drift + noisy execution; the self-poisoning scissors. Self-confidence
              must not stay high while referee-accuracy falls. Shown closed under the
              production/learning asymmetry, and open in the ablation that removes it
              (eager acting + self-training + no gate).

Run: python -m oad.phase2
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import replace
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .confirm import Thresholds
from .engine import Engine
from .workload import Workload, default_families
from .teacher import OracleTeacher, make_execute as _make_execute
from .harness import Referee

OUT = "/mnt/user-data/outputs"
FIGDIR = os.path.join(OUT, "oad_release_figs/phase2")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"; ACC = "#2f6f4f"; WARN = "#c25b3a"; BAD = "#a01b1b"
BLUE = "#2c5f8a"; GREY = "#8a8a8a"; GRID = "#d9d9d9"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight",
})

LOOSE = Thresholds(p_act=0.0, c_act=0.0)   # acts as soon as a match exists


class GT:
    """Mutable ground-truth holder so drift can be applied mid-stream. The teacher,
    the executor, and the referee all read through this one reference -- but remain
    distinct channels (learning source / production confirmation / evaluation)."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, text): return self.fn(text)


def drifted_workload(seed: int) -> Workload:
    fams = default_families()
    fams[0] = replace(fams[0], tool="get_weather_v2")   # the weather action is renamed
    return Workload(seed=seed, families=fams)


def make_execute(gt: GT, eps: float, rng: random.Random) -> Callable:
    """Production-confirmation channel. A correct action always confirms; a WRONG
    action confirms with probability eps (noisy execution / wrong-but-non-crashing)."""
    def execute(text, produced):
        if produced == gt(text):
            return True
        return rng.random() < eps
    return execute


def is_weather(action) -> bool:
    return str(action.get("tool", "")).startswith("get_weather")


# ===================================================================== E_cost
def _mixed_stream(wl: Workload, n: int, junk_frac: float, seed: int) -> List[str]:
    """Realistic traffic: valid family calls with a fraction of adversarial/OOD junk."""
    rng = random.Random(seed)
    base = wl.stream(n, zipf_s=0.6)
    junk = [t for t, _ in wl.adversarial_probes()] + wl.ood_probes()
    out = []
    for t in base:
        if rng.random() < junk_frac:
            out.append(rng.choice(junk))
        else:
            out.append(t)
    return out


def exp_cost(seed=1, n=4000, junk_frac=0.15):
    wl = Workload(seed=seed)
    gt = GT(wl.ground_truth)
    teacher = OracleTeacher(gt)
    harness = Referee(gt).score
    stream = _mixed_stream(wl, n, junk_frac, seed)

    def run(engine: Engine):
        acted_correct = acted_wrong = abstained = 0
        llm_cumulative = []
        llm_calls = 0
        for i, t in enumerate(stream):
            r = engine.step(t, teacher, harness, execute=None)  # clean execution
            if r.decision == "act":
                if r.harness_correct:
                    acted_correct += 1
                else:
                    acted_wrong += 1
            else:
                abstained += 1
                llm_calls += 1
            llm_cumulative.append(100.0 * llm_calls / (i + 1))
        acc = 100.0 * (acted_correct + abstained) / n   # abstain -> teacher (correct)
        return {
            "llm_rate": round(100.0 * abstained / n, 1),
            "accuracy": round(acc, 1),
            "silent_failures": acted_wrong,
            "llm_cumulative": llm_cumulative,
        }

    oad = run(Engine())
    eager = run(Engine(thr=LOOSE, use_gate=False, k_slot=1))
    always = {"llm_rate": 100.0, "accuracy": 100.0, "silent_failures": 0, "llm_cumulative": [100.0] * n}

    # ---- figure
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    axA.plot(oad["llm_cumulative"], color=ACC, lw=1.8, label="OAD")
    axA.plot(always["llm_cumulative"], color=GREY, lw=1.3, ls="--", label="always-LLM")
    axA.plot(eager["llm_cumulative"], color=BAD, lw=1.3, ls=":", label="eager-no-gate")
    axA.set_ylim(-3, 103); axA.set_xlabel("call index")
    axA.set_ylabel("cumulative LLM-call rate (%)")
    axA.set_title("E_cost  cost drops at held quality", loc="left")
    axA.legend(fontsize=8, loc="center right")

    names = ["always-LLM", "OAD", "eager-no-gate"]
    rows = [always, oad, eager]
    x = range(len(names)); w = 0.38
    axB.bar([i - w/2 for i in x], [r["llm_rate"] for r in rows], width=w,
            color=BLUE, label="LLM-call rate %")
    axB.bar([i + w/2 for i in x], [100.0 * r["silent_failures"] / n for r in rows], width=w,
            color=BAD, label="silent-failure rate %")
    axB.set_xticks(list(x)); axB.set_xticklabels(names, fontsize=8.5)
    axB.set_title("cheap OR safe -- OAD is both", loc="left")
    axB.legend(fontsize=8)
    for i, r in enumerate(rows):
        axB.text(i, max(r["llm_rate"], 100.0 * r["silent_failures"]/n) + 2,
                 f"acc {r['accuracy']:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "e_cost_quality.png")
    fig.savefig(path); plt.close(fig)

    for r in (oad, eager, always):
        r.pop("llm_cumulative", None)
    return {"oad": oad, "eager_no_gate": eager, "always_llm": always,
            "junk_frac": junk_frac, "figure": path}


# ===================================================================== E_drift
def exp_drift(seed=2, n=6000, drift_at=3000):
    wl = Workload(seed=seed)
    wl_d = drifted_workload(seed)
    gt = GT(wl.ground_truth)
    teacher = OracleTeacher(gt)
    harness = Referee(gt).score
    stream = wl.stream(n, zipf_s=0.6)

    overall, weather, autonomy = [], [], []
    w_idx = []  # weather call indices for rolling weather accuracy
    w_correct = []
    recovered_at = None
    eng = Engine(post_decay=0.99)
    gt.fn = wl.ground_truth
    rolling_ov, rolling_au = [], []
    ov_window, au_window = [], []
    W = 150
    for i, t in enumerate(stream):
        if i == drift_at:
            gt.fn = wl_d.ground_truth
        r = eng.step(t, teacher, harness, execute=None)
        ov_window.append(1 if r.harness_correct else 0)
        au_window.append(1 if r.decision == "act" else 0)
        if len(ov_window) > W: ov_window.pop(0); au_window.pop(0)
        rolling_ov.append(100.0 * sum(ov_window) / len(ov_window))
        rolling_au.append(100.0 * sum(au_window) / len(au_window))
        if is_weather(gt(t)):
            w_idx.append(i)
            w_correct.append(1 if r.harness_correct else 0)
        if (i > drift_at and recovered_at is None and len(w_correct) >= 30
                and sum(w_correct[-30:]) / 30 >= 0.95
                and i - drift_at > 30):
            recovered_at = i - drift_at

    # rolling weather accuracy aligned to weather indices
    wr = []
    ww = []
    for c in w_correct:
        ww.append(c)
        if len(ww) > 40: ww.pop(0)
        wr.append(100.0 * sum(ww) / len(ww))

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(rolling_ov, color=ACC, lw=1.6, label="overall accuracy (rolling)")
    ax.plot(w_idx, wr, color=BLUE, lw=1.6, label="weather accuracy (drifted family)")
    ax.plot(rolling_au, color=GREY, lw=1.1, ls="--", label="autonomous coverage")
    ax.axvline(drift_at, color=BAD, lw=1.4, label="drift (weather action renamed)")
    ax.set_ylim(-3, 103); ax.set_xlabel("call index"); ax.set_ylabel("%")
    ttl = f"E_drift  graceful degradation + recovery"
    if recovered_at is not None:
        ttl += f"  (weather re-derived ~{recovered_at} calls after drift)"
    ax.set_title(ttl, loc="left")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "e_drift_recovery.png")
    fig.savefig(path); plt.close(fig)

    return {"drift_at": drift_at, "recovered_after_calls": recovered_at,
            "final_overall_acc": round(rolling_ov[-1], 1),
            "final_weather_acc": round(wr[-1], 1) if wr else None,
            "figure": path}


# ===================================================================== E_scissors
def _run_scissors(condition: str, stream, drift_at, eps, seed):
    wl = Workload(seed=seed)
    wl_d = drifted_workload(seed)
    gt = GT(wl.ground_truth)
    teacher = OracleTeacher(gt)
    harness = Referee(gt).score
    rng = random.Random(seed + 777)
    execute = make_execute(gt, eps, rng)

    if condition == "safe":
        eng = Engine(post_decay=0.99)                    # full design (asymmetry on)
    else:
        eng = Engine(thr=LOOSE, self_train=True, use_gate=False, k_slot=1,
                     post_decay=0.99)                     # ablation

    pts = []  # (call_index, believed=acting_post_mean, true=harness_correct) for weather acts
    for i, t in enumerate(stream):
        if i == drift_at:
            gt.fn = wl_d.ground_truth
        r = eng.step(t, teacher, harness, execute=execute)
        if r.decision == "act" and is_weather(r.action) and r.acting_post_mean is not None:
            pts.append((i, r.acting_post_mean, 1 if r.harness_correct else 0))
    return pts


def _rolling_pair(pts, win=40):
    xs, believed, true = [], [], []
    bb, tt = [], []
    for (i, b, c) in pts:
        bb.append(b); tt.append(c)
        if len(bb) > win: bb.pop(0); tt.pop(0)
        xs.append(i)
        believed.append(100.0 * sum(bb) / len(bb))
        true.append(100.0 * sum(tt) / len(tt))
    return xs, believed, true


def exp_scissors(seed=3, n=6000, drift_at=3000, eps=0.6):
    wl = Workload(seed=seed)
    stream = wl.stream(n, zipf_s=0.6)
    safe = _run_scissors("safe", stream, drift_at, eps, seed)
    unsafe = _run_scissors("unsafe", stream, drift_at, eps, seed)

    xs_s, bel_s, tru_s = _rolling_pair(safe)
    xs_u, bel_u, tru_u = _rolling_pair(unsafe)

    # gap measured in the LATE post-drift window: a safe system has recovered
    # (gap -> 0), an unsafe one is still self-poisoned (gap stays large).
    late = drift_at + (n - drift_at) // 2

    def late_gap(xs, bel, tru):
        g = [b - t for (x, b, t) in zip(xs, bel, tru) if x > late]
        return round(sum(g) / len(g), 1) if g else 0.0

    fig, (axS, axU) = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    for ax, xs, bel, tru, name, tag in [
        (axS, xs_s, bel_s, tru_s, "SAFE  (asymmetry on)", "closed"),
        (axU, xs_u, bel_u, tru_u, "ABLATION  (eager + self-train + no gate)", "OPEN"),
    ]:
        ax.plot(xs, bel, color=WARN, lw=1.7, label="believed accuracy (self-confidence)")
        ax.plot(xs, tru, color=ACC, lw=1.7, label="true accuracy (referee)")
        ax.axvline(drift_at, color=BAD, lw=1.3)
        ax.fill_between(xs, tru, bel, where=[b > t for b, t in zip(bel, tru)],
                        color=BAD, alpha=0.12)
        ax.set_ylim(-3, 103); ax.set_xlabel("call index")
        ax.set_title(f"{name}\nscissors {tag}", loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    axS.set_ylabel("weather-action accuracy (%)")
    fig.suptitle(f"E_scissors  self-poisoning under drift + noisy execution (eps={eps})",
                 x=0.5, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGDIR, "e_scissors.png")
    fig.savefig(path); plt.close(fig)

    return {
        "eps": eps, "drift_at": drift_at,
        "safe_late_gap_pct": late_gap(xs_s, bel_s, tru_s),
        "unsafe_late_gap_pct": late_gap(xs_u, bel_u, tru_u),
        "safe_weather_acts": len(safe), "unsafe_weather_acts": len(unsafe),
        "figure": path,
    }


# ===================================================================== run
def main():
    print("E_cost  (cost/coverage at held quality)...")
    c = exp_cost(n=6000)
    print(json.dumps({k: v for k, v in c.items() if k != "figure"}, indent=2), "\n")

    print("E_drift  (drift + recovery)...")
    d = exp_drift(n=10000, drift_at=5000)
    print(json.dumps({k: v for k, v in d.items() if k != "figure"}, indent=2), "\n")

    print("E_scissors  (self-poisoning)...")
    s = exp_scissors(n=10000, drift_at=5000)
    print(json.dumps({k: v for k, v in s.items() if k != "figure"}, indent=2), "\n")

    results = {"E_cost": c, "E_drift": d, "E_scissors": s}
    with open(os.path.join(OUT, "oad_release_figs/phase2_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved figures to", FIGDIR)
    print("Saved metrics to", os.path.join(OUT, "oad_release_figs/phase2_results.json"))
    return results


if __name__ == "__main__":
    main()
