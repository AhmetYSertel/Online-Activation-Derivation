"""
experiments_bfcl.py -- ROADMAP Phase B transfer experiment.

OAD on a BFCL-grounded streaming workload (real tools / schemas / phrasings + real
BFCL irrelevance as the abstain set; see bfcl.py for the modeling decision). Reports
the cost/coverage transfer and the inherited-vs-engine-added (a/b) silent-failure
split, plus the abstention rate on real irrelevance, with multi-seed 95% CIs.

Offline validation (no key):
    python -m oad.experiments_bfcl --teacher oracle --seeds 8

Real transfer (your OpenAI key; one cached pass over the ~fixed unique-input universe):
    export OPENAI_API_KEY=sk-...
    python -m oad.experiments_bfcl --teacher openai --model gpt-5.5 --seeds 8
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List

from .engine import Engine
from .bfcl import (BFCLWorkload, build_families, load_irrelevance, bfcl_score,
                   OracleBFCLTeacher, OpenAIBFCLTeacher, DECLINE)
from .multiseed import ci95, paired_ci95

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results")


class DiskCachedBFCLTeacher:
    """Caches (model,text)->action to JSON. Only genuinely-new inputs hit the API, so
    multi-seed runs are free after the first warm pass."""
    def __init__(self, base, model: str, path: str):
        self.base, self.model, self.path = base, model, path
        self.api_calls = 0
        self.cache: Dict[str, Dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                self.cache = json.load(f)

    def __call__(self, text: str):
        k = f"{self.model}\x1f{text}"
        if k not in self.cache:
            self.api_calls += 1
            self.cache[k] = self.base(text)
            with open(self.path, "w") as f:
                json.dump(self.cache, f)
        return self.cache[k]


def run(engine_factory, stream, teacher, gt) -> dict:
    eng = engine_factory()
    ref = lambda x, p: bfcl_score(p, gt(x))
    acted = acted_ok = ab = ab_ok = inh = added = 0
    irrel_total = irrel_abst = 0
    for t in stream:
        is_irrel = gt(t)["tool"] == DECLINE["tool"]
        irrel_total += is_irrel
        r = eng.step(t, teacher, ref, execute=None)
        if r.decision == "act":
            acted += 1
            if r.harness_correct:
                acted_ok += 1
            elif not bfcl_score(teacher(t), gt(t)):
                inh += 1            # teacher also wrong -> inherited
            else:
                added += 1          # teacher right, replay wrong -> engine-added
        else:
            ab += 1
            ab_ok += bool(r.harness_correct)
            irrel_abst += is_irrel
    n = len(stream)
    return {
        "call_rate": round(100.0 * ab / n, 2),
        "accuracy": round(100.0 * (acted_ok + ab_ok) / n, 2),
        "autonomy": round(100.0 * acted / n, 2),
        "silent_failures": acted - acted_ok,
        "silent_inherited_a": inh, "silent_added_b": added,
        "irrelevance_abstain_pct": round(100.0 * irrel_abst / max(1, irrel_total), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["oracle", "openai"], default="oracle")
    ap.add_argument("--model", default="gpt-5.5")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--junk-frac", type=float, default=0.15)
    ap.add_argument("--max-families", type=int, default=30)
    args = ap.parse_args()

    fams = build_families(max_families=args.max_families)
    irr = load_irrelevance()
    wl0 = BFCLWorkload(seed=0, families=fams, irrelevance=irr)
    gt = wl0.ground_truth

    # fixed unique-input universe (filler sets are deterministic per family)
    universe = sorted(wl0._by_text.keys())
    print(f"families {len(fams)}  irrelevance {len(irr)}  unique-input universe {len(universe)}")

    if args.teacher == "oracle":
        teacher = OracleBFCLTeacher(wl0)
    else:
        cache_path = os.path.join(OUT, f"_bfcl_cache_{args.model}.json")
        teacher = DiskCachedBFCLTeacher(OpenAIBFCLTeacher(wl0, model=args.model),
                                        args.model, cache_path)
        for t in universe:           # one cached warm pass = the only API spend
            teacher(t)
        print(f"API calls made (cached): {teacher.api_calls}")
        # teacher raw accuracy on the universe
        acc = 100.0 * sum(bfcl_score(teacher(t), gt(t)) for t in universe) / len(universe)
        print(f"teacher raw accuracy on universe: {acc:.1f}%")

    oad_runs, always_runs = [], []
    for s in range(1, args.seeds + 1):
        wl = BFCLWorkload(seed=s, families=fams, irrelevance=irr)
        stream = wl.stream(args.n, junk_frac=args.junk_frac)
        oad_runs.append(run(lambda: Engine(), stream, teacher, gt))
        # always-call: teacher on every input
        ac = sum(bfcl_score(teacher(t), gt(t)) for t in stream)
        always_runs.append({"call_rate": 100.0, "accuracy": round(100.0 * ac / len(stream), 2),
                            "silent_failures": 0})

    keys = ["call_rate", "accuracy", "autonomy", "silent_failures",
            "silent_inherited_a", "silent_added_b", "irrelevance_abstain_pct"]
    oad_sum = {k: vars(ci95([r[k] for r in oad_runs])) for k in keys}
    always_acc = ci95([r["accuracy"] for r in always_runs])

    report = {"teacher": args.teacher, "model": args.model, "seeds": args.seeds,
              "n": args.n, "junk_frac": args.junk_frac, "n_families": len(fams),
              "universe": len(universe), "oad": oad_sum,
              "always_call_accuracy": vars(always_acc)}
    outpath = os.path.join(OUT, f"bfcl_results_{args.teacher}.json")
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    def g(k): return oad_sum[k]
    print(f"\n==== OAD on BFCL-grounded stream ({args.teacher} teacher, {args.seeds} seeds) ====")
    print(f"always-call accuracy : {always_acc.fmt()}%")
    print(f"OAD call rate        : {g('call_rate')['mean']:.1f} +/- {g('call_rate')['ci']:.1f}%")
    print(f"OAD accuracy         : {g('accuracy')['mean']:.1f} +/- {g('accuracy')['ci']:.1f}%")
    print(f"OAD autonomy         : {g('autonomy')['mean']:.1f} +/- {g('autonomy')['ci']:.1f}%")
    print(f"silent failures      : {g('silent_failures')['mean']:.1f} +/- {g('silent_failures')['ci']:.1f}  "
          f"(a={g('silent_inherited_a')['mean']:.1f}  b={g('silent_added_b')['mean']:.2f})")
    print(f"irrelevance abstain  : {g('irrelevance_abstain_pct')['mean']:.1f} +/- {g('irrelevance_abstain_pct')['ci']:.1f}%")
    print(f"\nSaved: {outpath}")
    return report


if __name__ == "__main__":
    main()
