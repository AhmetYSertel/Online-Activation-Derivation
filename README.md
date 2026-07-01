# Online Action Derivation (OAD)

A CPU-only, **no-trained-model** reference implementation of an online-derived,
auditable-gated, abstention-capable action policy. A frozen teacher (an LLM) is only the
*source* of (input → action) pairs; the system derives an action policy **online** from
them. Cost reduction, determinism, and coverage are *consequences* of the design, not
claims bolted on.

The live machinery is **classical**: induced symbolic structure (skeletons + slots via
anti-unification) plus online Beta-Binomial statistics. Action emission is deterministic
template instantiation, so identical input yields identical action by construction. There
is no transformer, no backprop, no GPU.

> **Before citing any result, read [`LIMITATIONS.md`](LIMITATIONS.md).** The packaged
> results use a *synthetic oracle teacher* and validate the mechanism, not transfer to
> real agent traffic. A real GPT teacher adapter is included and ready to run.

## Install
```
pip install -r requirements.txt          # numpy, scipy, matplotlib (CPU only)
```

## Reproduce the results
```
python -m oad.experiments_phase1          # mechanism validity (derivation, gate, slots)
python -m oad.experiments_phase2          # cost/coverage, drift recovery, scissors
python -m oad.experiments_rigor           # multi-seed CIs, eps sweep, ablations, teacher noise
```
Figures and metrics are written under `outputs/oad_release_figs/`.

## Use a real GPT model as the teacher
The teacher is pluggable. To distill a real model instead of the oracle:
```
pip install openai
export OPENAI_API_KEY=sk-...
```
```python
from oad.workload import Workload, default_families
from oad.teacher import OpenAITeacher
from oad.harness import Referee
from oad.engine import Engine

wl = Workload(seed=0)
teacher = OpenAITeacher(model="gpt-5.5", families=default_families())  # current GA flagship
ref     = Referee(wl.ground_truth)   # replace with an execution/human referee for real use
eng     = Engine(post_decay=0.99)
for text in wl.stream(2000):
    eng.step(text, teacher, ref.score)
```
`gpt-5.5` is the current GA flagship (2026-06); `gpt-5.4` is a cheaper option. Wrap the
teacher in a cache — the workload stream is Zipfian and repeats heavily, so caching cuts
API cost sharply.

## Package layout
```
oad/
  skeleton.py     tokenization, anchor-glob matching, cross-example slot induction
  confirm.py      Beta-Binomial posterior (+ recency decay) and the acting/learning bars
  regularity.py   a regularity: skeleton + action-template derivation + slot profiles + stats
  gate.py         the fixed auditable gate (coherence checks + maturity + hysteresis)
  engine.py       online loop: assignment, slot-promotion-vs-split, the safety flags
  workload.py     synthetic families + ground-truth oracle  (CAVEATS embedded)
  teacher.py      OracleTeacher / NoisyTeacher / OpenAITeacher  + the execute channel
  harness.py      out-of-loop referee  (CAVEATS embedded)
  experiments_phase1.py / _phase2.py / _rigor.py
LIMITATIONS.md    what these results do and do not establish (read this)
RESULTS.md        consolidated numbers with confidence intervals
```

## Headline results (see RESULTS.md for CIs)
- **Cost at held quality:** 100% accuracy at **~17% LLM-call rate**, **0 silent failures**
  (vs eager-no-gate: cheaper but 471 silent failures, 92% accuracy).
- **Drift:** automatic re-derivation, weather accuracy back to 100% ~90 calls after drift.
- **Self-poisoning scissors:** closed under the full design (gap ≈ 0) across execution
  noise ε ∈ {0…0.8}; opens (~60 pts) under ablation. The acting/maturity bar is the
  primary safety lever (R3).

## Safety/ablation flags on `Engine`
All default to the safe setting; flip them only for ablations:
`use_gate` (coherence checks), `k_slot` / `thr` (maturity & hysteresis), `post_decay`
(drift adaptation), `self_train` (fold own outputs — the asymmetry off-switch),
`allow_fallthrough` (act via a less-specific match).

## Relationship to WDR
OAD is the **black-box** sibling of White-Box Decision Replay (WDR). OAD's one
irreducible failure mode — an unseen, surface-identical boundary it cannot distinguish —
is exactly where white-box model internals are needed. The two are complementary.
