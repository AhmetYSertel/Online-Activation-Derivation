# What's new in this revision (ROADMAP execution)

Additions/changes on top of the original OAD repo, by ROADMAP phase. All new
experiments run offline (no API key) except the BFCL real-teacher transfer.

## New code (oad/)
- `execution.py` — honest, ground-truth-free execution signal (Phase A2).
- `multiseed.py` — reusable mean/95%-CI + paired-CI helpers (Phase A3).
- `bfcl.py` — BFCL-grounded streaming workload + AST referee + teachers (Phase B).
- `baselines.py` — semantic (MiniLM) cache + exact-match cache (Phase C).
- `experiments_honest_exec.py` — oracle-vs-honest confirmation comparison (A2).
- `experiments_honest_exec_multiseed.py` — A2 table with CIs (A3).
- `experiments_honest_exec_epssweep.py` — executor-noise sweep + figure (D1).
- `experiments_bfcl.py` — BFCL transfer driver (oracle offline / OpenAI real).
- `experiments_baselines.py` — cost-safety Pareto: OAD vs caches (C3).

## Changed
- `engine.py` — `_learn` fold_strict now guarded by `_reproduces_all`
  (template-consistency). No-op under the oracle (verified: Phase-1 86.5%/0 silent
  and 109/97/0 unchanged); load-bearing under the honest executor. Surfaced by A2.

## New data
- `data/bfcl/` — real BFCL v3 `simple`, `possible_answer`, `irrelevance`.

## New results & figures (results/)
- `honest_exec_results.json`, `honest_exec_multiseed.json`, `honest_exec_epssweep.json`
- `bfcl_results_oracle.json`, `baselines_pareto.json`
- `figs/honest_exec/honest_exec_eps_sweep.png`, `figs/baselines/pareto_cost_safety.png`

## New writeups (read these first)
- `ROADMAP.md` — full phased plan to NeurIPS-workshop standard.
- `results/HONEST_EXEC_FINDINGS.md` — A2/A3/D1 (the "0 silent" reframe, CI-backed).
- `results/BFCL_FINDINGS.md` — Phase B + the `--teacher openai` run command.
- `results/BASELINES_FINDINGS.md` — Phase C cost-safety dominance.
- `RELATED_WORK.md` — expanded positioning + verified BibTeX (E2).

## What you run next (needs your OpenAI key)
    export OPENAI_API_KEY=sk-...
    python -m oad.experiments_bfcl --teacher openai --model gpt-5.5 --seeds 8
~297 cached API calls; produces the real-data transfer table.
