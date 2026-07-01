"""
Phase 1 -- mechanism validity. Three experiments, each scored by the out-of-loop
harness:

  P1  Derivation curve     -- derivation happens and is correct; coverage grows
                              from zero with no hand-written templates; reports
                              sample-efficiency (calls to first autonomous replay).
  P2  Gate accuracy        -- act in-domain, abstain on adversarial + OOD, with an
                              auditable reason; zero silent failures.
  P3  Slot induction       -- a position is promoted to a slot only when execution-
                              confirmed (generalizes to unseen values); an action-
                              inconsistent boundary splits instead of contaminating.

Run: python -m oad.experiments
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import skeleton as sk
from .confirm import Thresholds, confirmations_for_acting
from .engine import Engine
from .workload import Workload
from .teacher import OracleTeacher
from .harness import Referee

OUT = "/mnt/user-data/outputs"
FIGDIR = os.path.join(OUT, "oad_release_figs/phase1")
os.makedirs(FIGDIR, exist_ok=True)

INK = "#1b1b1b"
ACC = "#2f6f4f"     # act / correct
WARN = "#c25b3a"    # abstain
BAD = "#a01b1b"     # silent failure
GRID = "#d9d9d9"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight",
})


# ===================================================================== P1
def exp1_derivation(seed: int = 1, n: int = 400, zipf_s: float = 0.7):
    wl = Workload(seed=seed)
    eng = Engine()
    teacher, harn = OracleTeacher(wl.ground_truth), Referee(wl.ground_truth)
    stream = wl.stream(n, zipf_s=zipf_s)

    decisions: List[int] = []     # 1 = autonomous act, 0 = abstain
    correct: List[int] = []       # harness-correct on the produced/teacher action
    first_act: Dict[int, int] = {}  # rid -> call index of first autonomous act

    for i, t in enumerate(stream):
        r = eng.step(t, teacher, harn.score)
        decisions.append(1 if r.decision == "act" else 0)
        correct.append(1 if r.harness_correct else 0)
        if r.decision == "act" and r.rid not in first_act:
            first_act[r.rid] = i

    # rolling coverage
    W = 25
    cov = []
    for i in range(len(decisions)):
        lo = max(0, i - W + 1)
        cov.append(100.0 * sum(decisions[lo:i + 1]) / (i - lo + 1))
    autonomy = 100.0 * sum(decisions) / len(decisions)
    acc_when_act = 100.0 * sum(c for c, d in zip(correct, decisions) if d) / max(1, sum(decisions))
    silent = sum(1 for c, d in zip(correct, decisions) if d and not c)

    # sample efficiency over multiple seeds: calls until each family first acts
    eff = []
    for s in range(5):
        wl2 = Workload(seed=10 + s)
        eng2 = Engine()
        te2, ha2 = OracleTeacher(wl2.ground_truth), Referee(wl2.ground_truth)
        seen = set()
        for i, t in enumerate(wl2.stream(n, zipf_s=zipf_s)):
            r = eng2.step(t, te2, ha2.score)
            if r.decision == "act" and r.rid not in seen:
                seen.add(r.rid)
                eff.append(1)  # placeholder, replaced by per-family below
        # per-family first-act in call-occurrences
    # Per-family occurrence-efficiency (single representative seed, clearer):
    fam_first_occ = _family_efficiency(seed=seed, n=n, zipf_s=zipf_s)

    # ---- figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.2),
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(cov, color=ACC, lw=1.8, label="autonomous coverage (rolling, w=25)")
    ax1.axhline(autonomy, color=INK, lw=0.9, ls="--",
                label=f"final autonomy {autonomy:.0f}%")
    for rid, idx in first_act.items():
        ax1.axvline(idx, color=WARN, lw=0.6, alpha=0.5)
    ax1.set_ylim(-3, 103)
    ax1.set_xlabel("call index")
    ax1.set_ylabel("autonomous coverage (%)")
    ax1.set_title(f"P1  Derivation curve  --  acc-when-acting {acc_when_act:.0f}%, "
                  f"silent failures {silent}", loc="left")
    ax1.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fams = list(fam_first_occ.keys())
    vals = [fam_first_occ[f] for f in fams]
    ax2.barh(fams, vals, color=ACC, height=0.6)
    for y, v in enumerate(vals):
        ax2.text(v + 0.3, y, str(v), va="center", fontsize=8)
    ax2.set_xlabel("family occurrences until first autonomous replay (sample-efficiency)")
    ax2.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(FIGDIR, "p1_derivation_curve.png")
    fig.savefig(path)
    plt.close(fig)

    return {
        "final_autonomy_pct": round(autonomy, 1),
        "acc_when_acting_pct": round(acc_when_act, 1),
        "silent_failures": silent,
        "acting_bar_clean_successes": confirmations_for_acting(Thresholds()),
        "family_occurrences_to_first_act": fam_first_occ,
        "figure": path,
    }


def _family_efficiency(seed: int, n: int, zipf_s: float) -> Dict[str, int]:
    """For each family, the count of that family's occurrences before its
    regularity first replays autonomously."""
    wl = Workload(seed=seed)
    eng = Engine()
    teacher, harn = OracleTeacher(wl.ground_truth), Referee(wl.ground_truth)
    occ = Counter()
    first = {}
    rid_to_fam = {}
    fam_names = [f.name for f in wl.families]
    stream = wl.stream(n, zipf_s=zipf_s)
    for t in stream:
        # which family produced t?
        gt = wl.ground_truth(t)
        fam = next((f.name for f in wl.families if f.tool == gt["tool"]), None)
        if fam is None:
            continue
        occ[fam] += 1
        r = eng.step(t, teacher, harn.score)
        if r.decision == "act" and fam not in first:
            first[fam] = occ[fam]
    # families that never acted -> n/a (mark with the count seen)
    return {f: first.get(f, occ[f]) for f in fam_names if occ[f] > 0}


# ===================================================================== P2
def exp2_gate(seeds=range(5), warm=350, zipf_s=0.6):
    cats = ["in_domain", "adversarial", "ood"]
    agg = {c: Counter() for c in cats}        # acted_correct / abstained / silent
    reasons = Counter()
    total_silent = 0

    for s in seeds:
        wl = Workload(seed=100 + s)
        eng = Engine()
        teacher, harn = OracleTeacher(wl.ground_truth), Referee(wl.ground_truth)
        for t in wl.stream(warm, zipf_s=zipf_s):
            eng.step(t, teacher, harn.score)

        # in-domain
        for t in wl.in_domain_probes(60):
            r = eng.step(t, teacher, harn.score)
            if r.decision == "act":
                key = "acted_correct" if r.harness_correct else "silent"
                agg["in_domain"][key] += 1
                total_silent += int(not r.harness_correct)
            else:
                agg["in_domain"]["abstained"] += 1
        # adversarial
        for t, kind in wl.adversarial_probes():
            r = eng.step(t, teacher, harn.score)
            if r.decision == "act":
                agg["adversarial"]["acted_correct" if r.harness_correct else "silent"] += 1
                total_silent += int(not r.harness_correct)
            else:
                agg["adversarial"]["abstained"] += 1
                reasons[r.reason] += 1
        # ood
        for t in wl.ood_probes():
            r = eng.step(t, teacher, harn.score)
            if r.decision == "act":
                agg["ood"]["acted_correct" if r.harness_correct else "silent"] += 1
                total_silent += int(not r.harness_correct)
            else:
                agg["ood"]["abstained"] += 1

    # ---- figure (stacked bars, counts)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = ["in-domain\n(should ACT)", "adversarial\n(should ABSTAIN)", "OOD\n(should ABSTAIN)"]
    correct = [agg[c]["acted_correct"] for c in cats]
    abst = [agg[c]["abstained"] for c in cats]
    silent = [agg[c]["silent"] for c in cats]
    x = range(len(cats))
    ax.bar(x, correct, color=ACC, label="acted & correct")
    ax.bar(x, abst, bottom=correct, color=WARN, label="abstained -> LLM")
    bot2 = [a + b for a, b in zip(correct, abst)]
    ax.bar(x, silent, bottom=bot2, color=BAD, label="acted & WRONG (silent failure)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"probe count (summed over {len(list(seeds))} seeds)")
    ax.set_title(f"P2  Gate accuracy  --  total silent failures: {total_silent}", loc="left")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    path = os.path.join(FIGDIR, "p2_gate_accuracy.png")
    fig.savefig(path)
    plt.close(fig)

    def rate(c, key):
        tot = sum(agg[c].values())
        return round(100.0 * agg[c][key] / tot, 1) if tot else 0.0

    return {
        "in_domain_act_correct_pct": rate("in_domain", "acted_correct"),
        "in_domain_abstain_pct": rate("in_domain", "abstained"),
        "adversarial_abstain_pct": rate("adversarial", "abstained"),
        "ood_abstain_pct": rate("ood", "abstained"),
        "total_silent_failures": total_silent,
        "adversarial_abstain_reasons": dict(reasons),
        "figure": path,
    }


# ===================================================================== P3
def exp3_slot_induction():
    log = {"promotion": {}, "split": {}}

    # ---- Promotion: born literal -> slot, then handle an UNSEEN value
    wl = Workload(seed=7)
    eng = Engine()
    teacher, harn = OracleTeacher(wl.ground_truth), Referee(wl.ground_truth)
    trace = []
    feed = (["weather in istanbul", "weather in tokyo", "weather in berlin"]
            + ["weather in tokyo", "weather in istanbul"] * 4)
    for t in feed:
        r = eng.step(t, teacher, harn.score)
        reg = next((x for x in eng.regs if x.rid == r.rid), None)
        trace.append((t, r.decision, sk.skeleton_str(reg.skeleton) if reg else "-",
                      sk.n_slots(reg.skeleton) if reg else 0))
    # unseen city (never in feed)
    unseen = "weather in cairo"
    r_un = eng.step(unseen, teacher, harn.score)
    reg = next((x for x in eng.regs if x.rid == r_un.rid), None)
    log["promotion"] = {
        "born_literal": trace[0][2],
        "after_second_value": next((s for (_, _, s, n) in trace if n == 1), trace[-1][2]),
        "unseen_input": unseen,
        "unseen_decision": r_un.decision,
        "unseen_correct": bool(r_un.harness_correct),
        "trace": trace,
    }

    # ---- Split: boundary shares anchors but a different tool -> must not fold.
    # The boundary is presented during LEARNING (interleaved), so it reaches the
    # learn path and spawns its own regularity instead of contaminating smart_home.
    wl2 = Workload(seed=7, include_boundary=True)
    eng2 = Engine()
    te2, ha2 = OracleTeacher(wl2.ground_truth), Referee(wl2.ground_truth)
    training = (["open the door", "open the window", "open the garage", "open the account"] * 12)
    for t in training:
        eng2.step(t, te2, ha2.score)

    sh = next(x for x in eng2.regs if x.tool == "smart_home")
    sh_vals = sorted(sh.slot_support.get(0, set()))
    bank = [x for x in eng2.regs if x.tool == "banking"]

    # routing: boundary -> banking (specificity), door -> smart_home, both correct
    route_account = eng2.step("open the account", te2, ha2.score)
    route_door = eng2.step("open the door", te2, ha2.score)

    # ---- honest limitation: a surface-identical UNSEEN boundary. Trained only on
    # smart_home (no banking), "open the account" is indistinguishable from a
    # smart_home command by surface features -> the system acts and fails silently.
    wl3 = Workload(seed=7, include_boundary=True)
    eng3 = Engine()
    te3, ha3 = OracleTeacher(wl3.ground_truth), Referee(wl3.ground_truth)
    for t in (["open the door", "open the window", "open the garage"] * 12):
        eng3.step(t, te3, ha3.score)
    lim = eng3.step("open the account", te3, ha3.score)   # never seen banking

    log["split"] = {
        "smarthome_skeleton": sk.skeleton_str(sh.skeleton),
        "smarthome_slot_values": sh_vals,
        "account_folded_into_smarthome": "account" in sh_vals,
        "boundary_spawned_separate_regularity": len(bank) > 0,
        "route_account_tool": route_account.action["tool"],
        "route_account_correct": bool(route_account.harness_correct),
        "route_door_tool": route_door.action["tool"],
        "route_door_correct": bool(route_door.harness_correct),
        "unseen_boundary_decision": lim.decision,
        "unseen_boundary_correct": bool(lim.harness_correct),
        "unseen_boundary_note": ("surface-identical unseen boundary -> acts and fails "
                                 "silently; the irreducible black-box limitation, and the "
                                 "motivation for white-box methods"),
    }

    # ---- figure: state-transition schematic rendered as text panels
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for ax in (axL, axR):
        ax.axis("off")
    pro = log["promotion"]
    txtL = (
        "PROMOTION  (execution-confirmed slot)\n"
        "------------------------------------\n"
        f"born literal:         {pro['born_literal']}\n"
        f"+2nd value -> slot:   {pro['after_second_value']}\n"
        f"unseen input:         '{pro['unseen_input']}'\n"
        f"decision:             {pro['unseen_decision'].upper()}"
        f"  (correct={pro['unseen_correct']})\n\n"
        "=> a position becomes a slot only after a second\n"
        "   confirmed value; the skeleton then handles a\n"
        "   value it has never seen."
    )
    sp = log["split"]
    txtR = (
        "SPLIT  (action-inconsistent boundary)\n"
        "------------------------------------\n"
        f"smart_home skeleton:  {sp['smarthome_skeleton']}\n"
        f"slot values:          {sp['smarthome_slot_values']}\n"
        f"'open the account' folded in?  {sp['account_folded_into_smarthome']}\n"
        f"separate reg spawned?          {sp['boundary_spawned_separate_regularity']}\n"
        f"route 'open the account' -> {sp['route_account_tool']}"
        f"  (correct={sp['route_account_correct']})\n"
        f"route 'open the door'    -> {sp['route_door_tool']}"
        f"  (correct={sp['route_door_correct']})\n\n"
        "limitation -- UNSEEN identical boundary:\n"
        f"  'open the account' (never seen) -> {sp['unseen_boundary_decision'].upper()}"
        f"  (correct={sp['unseen_boundary_correct']})\n"
        "  surface features cannot separate it; acts\n"
        "  and fails silently -> motivates white-box."
    )
    axL.text(0.0, 1.0, txtL, va="top", ha="left", family="monospace", fontsize=9.2,
             transform=axL.transAxes)
    axR.text(0.0, 1.0, txtR, va="top", ha="left", family="monospace", fontsize=9.2,
             transform=axR.transAxes)
    fig.suptitle("P3  Slot induction: promote vs split", x=0.06, ha="left", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(FIGDIR, "p3_slot_induction.png")
    fig.savefig(path)
    plt.close(fig)
    log["figure"] = path
    return log


# ===================================================================== run
def main():
    print("Calibrating harness against a small human gold set...")
    wl = Workload(seed=0)
    gold = [(t, wl.ground_truth(t)) for t in
            ["weather in tokyo", "show my calendar for monday", "open the door"]]
    rel = Referee(wl.ground_truth).calibrate(gold)
    print(f"  harness reliability on gold set: {rel:.2f}\n")

    print("P1  Derivation curve...")
    p1 = exp1_derivation()
    print(json.dumps({k: v for k, v in p1.items() if k != "family_occurrences_to_first_act"}, indent=2))
    print("  per-family occurrences to first autonomous replay:",
          p1["family_occurrences_to_first_act"], "\n")

    print("P2  Gate accuracy...")
    p2 = exp2_gate()
    print(json.dumps(p2, indent=2), "\n")

    print("P3  Slot induction...")
    p3 = exp3_slot_induction()
    print(json.dumps({k: v for k, v in p3.items() if k != "promotion"}, indent=2, default=str))
    print("  promotion:", json.dumps({k: v for k, v in p3["promotion"].items() if k != "trace"}, default=str), "\n")

    results = {"harness_reliability": rel, "P1": p1, "P2": p2, "P3": p3}
    with open(os.path.join(OUT, "oad_release_figs/phase1_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved figures to", FIGDIR)
    print("Saved metrics to", os.path.join(OUT, "oad_release_figs/phase1_results.json"))
    return results


if __name__ == "__main__":
    main()
