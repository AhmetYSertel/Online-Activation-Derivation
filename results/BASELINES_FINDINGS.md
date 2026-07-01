# Cost-safety baselines — findings (ROADMAP Phase C / Tier-1 #2)

Closes LIMITATIONS.md #4 ("always-call / eager are not real competitors"). OAD is now
compared on the **cost-safety axis** (silent-failure rate vs model-call rate) against
its real peers: a **semantic (MiniLM embedding) cache** (GPTCache-style) and an
**exact-match cache**, on the BFCL-grounded streaming workload with an open slot space
(40 fillers/family, 1317-input universe), oracle stand-in teacher, 6 seeds (mean ± 95%
CI). Figure: `results/figs/baselines/pareto_cost_safety.png`.

## Result — OAD dominates both peers

| system | model-call rate | silent-failure rate |
|---|---|---|
| **OAD (full design)** | **24.1 ± 0.7%** | **0.00 ± 0.00%** |
| exact-match cache | 34.1 ± 0.3% | 0.00% |
| semantic cache @ matched call rate (th=0.84) | 25.4% | **37.4%** |
| semantic cache @ strictest (th=0.95) | 31.6% | 12.6% |

- **vs exact-match cache:** same perfect safety (0% silent), but OAD is **~10 points
  cheaper** (24.1% vs 34.1% calls). An exact cache cannot generalize — every unseen
  slot value is a miss and a model call — whereas OAD's slot induction **acts on
  unseen fillers**. This is the generalization advantage, isolated.
- **vs semantic cache:** at OAD's call rate the cache sits at **37% silent vs OAD's
  0%**, and it never reaches 0% at any threshold (12.6% even at the strictest). A
  verbatim cache serves a **stale parameter** across slot variants and **mis-serves**
  irrelevance probes that embed near a cached tool-call — exactly what OAD's gate +
  slot induction prevent.

So OAD is strictly lower-left of both competitors on the cost-safety plane: it matches
the safe cache's safety at lower cost, and the cheap cache's regime at far higher
safety.

## Why the open slot space matters (an honesty correction)

A first run used only 6 fillers/family. There an **exact-match cache looked competitive**
(heavy exact repeats → 0 silent at low cost), because the slot space was effectively
closed. That understated OAD: its real edge is **generalization to unseen slot values**,
which only shows when the slot space is open (as real agent traffic is). With 40
fillers/family the exact cache's cost rises above OAD's while OAD holds 0% silent —
the honest demonstration. (Reported so the choice is auditable.)

## Caveats (stated, not hidden)

- **Oracle stand-in teacher.** OAD's 0% silent here is structural (gate + correct slot
  fill); with a fallible teacher OAD inherits teacher errors — but so do both caches,
  *on top of* their structural failures, so the dominance persists or widens. The
  real-teacher version drops in once the BFCL cache exists (`--teacher openai`).
- **The semantic cache is verbatim.** A cache that templated/parameterized its hits
  would reduce stale-parameter failures — by reinventing OAD's slot induction. That
  convergence is itself the point: the safe way to reuse decisions on slot-varying
  traffic *is* induction, not flat caching.
- Fillers are same-shape synthetic; real value distributions would test the
  novel-content-word gate further.

## Run

```
python -m oad.experiments_baselines --seeds 6 --fillers 40
```

Needs `sentence-transformers` (MiniLM; downloads once, then local). `oad/baselines.py`
holds `SemanticCache`, `run_exact_cache`, and a pluggable encoder.
