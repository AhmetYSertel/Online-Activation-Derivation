# Honest-executor findings — ROADMAP Phase A2

This is the deliverable for ROADMAP §0 finding **F-2**: the packaged "independent"
confirmation signal (`teacher.make_execute` = `produced == ground_truth(text)`) is
the referee's own oracle, so the headline **109 → 0** silent-failure "fix" rests on
giving the system access to the answer key. This run replaces it with an execution
signal that has **no access to `ground_truth`** and measures what survives.

Reproduce (fully offline from the packaged GPT-5.5 cache; no API key, no network):

```
python -m oad.experiments_honest_exec
```

## What was added / changed

- **`oad/execution.py`** (new) — `ToolEnvironment` + `make_honest_execute`. A call
  "succeeds" iff it is schema-valid for the named tool (a geocoder rejects `1945`, a
  calendar rejects `2026`, a contact book rejects `99problems`, a smart-home hub
  rejects `12`). Validators are authored from the **tool's** point of view; nothing
  here imports the workload ground truth or the referee. An `eps` knob models a tool
  that returns a plausible-but-wrong result instead of erroring (for the Phase-D
  sweep).
- **`oad/experiments_honest_exec.py`** (new) — runs the confirmation-channel
  progression for the oracle-backed signal and the honest signal in one offline run.
- **`oad/engine.py`** (1 change, `_learn`) — the `fold_strict` branch is now guarded
  by the same `_reproduces_all` template-consistency test the `generalize` branch
  already used (see below).

## Results (n=1500, seed 1, GPT-5.5 cache)

| confirmation channel | call rate | acc | silent (a / b) | weather act/abstain |
|---|---|---|---|---|
| baseline (teacher-backed tautology) | 11.4% | 91.7% | 109 (109 / 0) | 384 / 6 |
| oracle exec + gated induction *(packaged "fix")* | 18.9% | 91.7% | **0** (0 / 0) | 381 / 9 |
| **honest exec + gated induction** *(no ground truth)* | 17.5% | 91.7% | **22** (22 / 0) | 381 / 9 |

The honest, ground-truth-free signal removes **~80% of the silent failures (109 →
22) at the same call-rate cost** as the oracle fix (17.5% vs 18.9%), with engine-added
error **b = 0** preserved and legitimate acting intact (weather 381 acts, identical to
the oracle fix).

### The residual 22 are schema-valid-but-label-wrong (3 unique inputs)

| input | ×count | teacher | ground truth | why execution can't catch it |
|---|---|---|---|---|
| `remind me to do a b c d e` | 9 | `create_reminder` | `__llm_other__` | a reminder accepts free text — the call genuinely succeeds |
| `weather forecast for istanbul` | 8 | `get_weather(istanbul)` | `__llm_other__` | `get_weather(istanbul)` is schema-valid **and arguably correct**; the gt label is pedantic |
| `weather in istanbul please cancel` | 5 | `get_weather(istanbul)` | `__llm_other__` | same: the produced call executes cleanly |

So the genuinely-uncatchable-by-execution residue is essentially **one** case (the
long reminder); the other two are debatable ground-truth labels, not real harms. The
honest takeaway is sharper than the oracle's clean 0: *a pure execution signal cannot
distinguish a wrong-but-well-formed call from a right one, and on inspection most of
that residue is labelling artifact.*

### Executor self-reliability (the signal the fix now rests on)

On the 51 unique inputs: **82.4%** agreement with correctness; **3** false positives
(confirms a wrong action — the residual silent source) and **6** false negatives (all
OOD declines, which defer anyway). This is the `< 1.0` reliability the safety claim
must now be stated against, replacing the oracle's perfect-by-construction 1.0.

## The engine change, and why it is a contribution (not just a fix)

Turning on the honest executor first **broke** the system: silent failures fell but
legitimate weather acting collapsed to **40 / 350** and the call rate blew up to
**40.5%**, with `slot_extraction_method` abstentions firing all over weather — a gate
branch the paper states "does not fire on the present workload."

Root cause: the honest executor confirms `get_weather(istanbul)` for `weather in
istanbul please cancel` (istanbul *is* a valid city), but the captured slot is
`istanbul please cancel` while the param is `istanbul`. The old `fold_strict` folded
on tool-match alone, so this structurally-foreign example poisoned the weather
template into an unfillable/`semantic` state and silently killed the regularity.

Under a **perfect oracle** this never happens — only well-formed family instances are
ever gt-correct, so every folded example is automatically template-consistent. The
oracle's perfection was *masking* a latent coupling.

The fix: `fold_strict` now folds only if the augmented example set stays
template-consistent (`_reproduces_all`). This is a **no-op under the oracle** (verified:
oracle rows still 109 / 0; Phase-1 mechanism numbers unchanged at 86.5% autonomy / 0
silent) and **load-bearing under honest execution**.

The paper-level point: **an independent execution signal is necessary but not
sufficient.** Gating structural learning on "does it execute" must be paired with a
structural-consistency check at fold time, or the executor will confirm actions whose
parameters do not trace to the captured slots. This strengthens, rather than
weakens, the safety story — and it makes the dormant `slot_extraction_method` gate
branch genuinely active (partly addressing ROADMAP Tier-2 #6a).

## Multi-seed hardening (ROADMAP Phase A3)

The table above is seed 1. Across **20 seeds** (mean ± 95% CI, Student-t; fully
offline — every seed stays within the 51-input cache universe):

| confirmation channel | silent failures | call rate | b (engine-added) |
|---|---|---|---|
| baseline (teacher-backed) | 110.1 ± 3.6 | 11.2 ± 0.4% | 0.00 ± 0.00 |
| oracle exec + gated | **0.0 ± 0.0** | 18.9 ± 0.4% | 0.00 ± 0.00 |
| honest exec + gated | **26.9 ± 2.7** | 17.0 ± 0.4% | 0.00 ± 0.00 |

Paired reductions (same seeds):

- baseline → honest+gated: **83.2 ± 2.5 silent failures removed** (≈**75%**, not the
  ~80% seed-1 suggested — multi-seed corrected this).
- honest residual vs oracle: **26.9 ± 2.7** — the schema-valid-but-wrong gap a pure
  execution signal cannot close.
- **b = 0.00 ± 0.00 across all configs and seeds** — the engine-adds-no-error result
  is robust, not a single-seed coincidence.

Reproduce: `python -m oad.experiments_honest_exec_multiseed --seeds 20`. Helpers in
`oad/multiseed.py` (`ci95`, `paired_ci95`) are reusable by every other driver,
including the Phase-D ε-sweep.

## Executor-noise sweep (ROADMAP Phase D1)

The honest fix's residual was shown at a clean executor (eps=0). Sweeping eps = the
probability that a schema-**invalid** call is confirmed anyway (a tool that returns a
plausible-but-wrong result instead of erroring), 16 seeds, mean ± 95% CI:

| eps | silent-failure rate | call rate |
|---|---|---|
| 0.00 | 1.82 ± 0.21% | 16.9% |
| 0.05 | 3.88 ± 0.58% | 15.0% |
| 0.10 | 4.88 ± 0.44% | 14.0% |
| 0.20 | 6.05 ± 0.38% | 12.9% |
| 0.50 | 6.81 ± 0.35% | 11.9% |
| 1.00 | 7.33 ± 0.27% | 11.2% |

At eps=1 the signal carries no information and the system regresses exactly to the
teacher-backed baseline (7.33% ≈ 110/1500). The degradation is **steep early**: a
merely 10%-dishonest executor (eps=0.1) already gives back more than half the safety
gain (1.82% → 4.88%). So the guarantee tolerates only a near-honest executor; this
quantifies — rather than hand-waves — the "shown only at eps=0" caveat the paper
flags as future work. Figure: `results/figs/honest_exec/honest_exec_eps_sweep.png`.
Reproduce: `python -m oad.experiments_honest_exec_epssweep --seeds 16`.

## Implications for the paper

1. Reframe the §3.4 headline: not "binding to an execution signal drives silent
   failures to **zero**," but "to **zero under a ground-truth-backed executor**, and
   to a small schema-valid residue (**75% ± 2% reduction**, 26.9 ± 2.7 residual over
   20 seeds) under an honest, gt-free one." State the residual explicitly.
2. Keep `b = 0` — it survives the honest executor.
3. Add the structural-consistency requirement as a stated design condition of the fix.
4. Next: Phase **D1** ε-sweep is now trivial (`make_honest_execute(eps=…)`), and
   Phase **A3** multi-seed CIs apply directly to this table.
