# OAD — full archive

Everything for the Online Action Derivation (OAD) project in one place.

## Start here
1. `README.md`        — what OAD is, install, how to run, GPT-5.5 teacher usage.
2. `LIMITATIONS.md`   — what the results do and do NOT establish. **Read before citing.**
3. `RESULTS.md`       — consolidated numbers with 95% confidence intervals.

## Code  (`oad/`, CPU-only, no trained model)
- `skeleton.py confirm.py regularity.py gate.py engine.py` — the system.
- `workload.py`  — synthetic families + ground-truth oracle (caveats embedded).
- `teacher.py`   — OracleTeacher / NoisyTeacher / **OpenAITeacher** (real GPT-5.5, pluggable).
- `harness.py`   — out-of-loop referee (caveats embedded).
- `experiments_phase1.py / _phase2.py / _rigor.py` — drivers.

Reproduce:
```
pip install -r requirements.txt
python -m oad.experiments_phase1     # mechanism validity
python -m oad.experiments_phase2     # cost/coverage, drift, scissors
python -m oad.experiments_rigor      # multi-seed CIs, eps sweep, ablations, teacher noise
```

## Pre-computed results  (`results/`)
- `figs/phase1|phase2|rigor/*.png` — all figures.
- `phase1_results.json phase2_results.json rigor_results.json` — all metrics.

## Documents  (`docs/`)
- `action_derivation_spec.md`      — full architecture spec + decision ledger (D1–D5, WP1–WP8).

## What's still needed for a paper
See LIMITATIONS.md: a real GPT-5.5 teacher run (adapter included), a real tool-use
benchmark as the input distribution, and one real cost-reduction baseline.
