# ROADMAP — bringing OAD to NeurIPS-workshop standard

Status as of 2026-06-30. This is the execution plan that turns the current preprint
(mechanism validated under an oracle; a thin real-teacher transfer run) into a
workshop-publishable systems paper. It is sequenced by dependency, not by tier, so
nothing in the plan blocks on something later in the list.

Every item from the initial standards discussion is folded in and traceable — see
**§7 Traceability** so nothing is silently dropped.

---

## 0. The two facts this plan is built around

These came out of reading the actual code/cache, and they reset the priorities.

- **F-1 — the real-teacher eval is informationally thin.** The entire GPT-5.5
  transfer reduces to **51 unique inputs, of which exactly 8 are errors, and all 8
  are author-written adversarial probes** from `workload.py`. The headline "109
  silent failures" is those 8 unique inputs Zipf-amplified; "84.3% teacher accuracy"
  is 43/51 (6 clean families right, 8 hand-written gotchas wrong, by construction).
  → A real input distribution is **mandatory**, not external-validity polish.
  → Multi-seed CIs are still required for honesty but **cannot** rescue this: the
    unique adversarial set stays the same 8 inputs across seeds.

- **F-2 — the "teacher-independent" execution signal is the ground-truth oracle.**
  `teacher.make_execute` returns `produced == ground_truth(text)` (± ε). The 109→0
  "fix" therefore rests on giving the system access to an oracle that already knows
  the answer — independence is *structural*, not *informational*. The honest version
  needs an execution signal genuinely decoupled from the referee (schema-valid /
  exception-based), then an ε-sweep on it.

## 1. What must NOT regress (preserve these)

The architecture already earns real credit; do not lose it during refactors.

- **Engine ⟂ Referee separation is real in code** (`engine.py` never imports the
  harness). Keep this property after every refactor — it is the basis of the
  auditability claim.
- **The two ablation levers** (`self_train`, `confirm_learning`) and the
  acting/learning threshold asymmetry (`p_act=0.85,c_act=0.80` vs
  `p_learn=0.95,c_learn=0.95`). These isolate the safety attribution; keep them
  switchable.
- **`_reproduces_all` promote-vs-split test** — the action-consistency check is the
  heart of the induction; do not weaken it for benchmark convenience.
- **The honest-LIMITATIONS culture.** Keep `LIMITATIONS.md` brutal; update items as
  they are resolved rather than deleting them.
- **The b=0 per-input decomposition method** (`silent_failure_breakdown.json`
  logic). Re-run it on the real distribution; do not change its definition.

---

## 2. Phased plan (dependency-ordered)

Effort is relative (engineer-days). "Done when" is the acceptance gate for each step.

### Phase A — Foundation refactor (unblocks everything; mostly no API)

- **A1. Generalize the teacher interface.** Introduce a `Teacher` protocol; keep
  `OpenAITeacher`, add `OpenWeightsTeacher` (Qwen2.5-Coder via local/HF function
  calling) so a fully reproducible, closed-API-free run exists.
  - *Deliverable:* `teacher.py` with two interchangeable backends + a cache keyed by
    `(model, text)`.
  - *Done when:* the same experiment runs unchanged against GPT and against Qwen.
  - *Effort:* 1–2 d. *Addresses:* Tier-1 #4 (open-weights reproducibility).

- **A2. Honest execution signal (fixes F-2).** Split `ExecutionSignal` from
  `Referee`. Implement a schema/exception-based executor: a call "succeeds" iff it is
  schema-valid and does not raise — **never** by comparison to ground truth. Rewrite
  `make_execute` so `confirm_learning` binds to *this*, not to the oracle.
  - *Deliverable:* `execution.py` (or extend `teacher.py`) + an `eps`-parameterized
    honest executor.
  - *Done when:* the 109→0 result can be reproduced with an executor that has no
    access to `ground_truth`.
  - *Effort:* 1–2 d. *Addresses:* Turn-2 finding F-2; prerequisite for D1.

- **A3. Multi-seed + bootstrap CIs in every driver.** Wrap the experiment runners to
  sweep ≥10 seeds and report mean ± 95% CI (paired bootstrap where comparing
  systems). Cache makes API cost ~0.
  - *Deliverable:* a `run_multiseed(...)` helper used by all `experiments_*`.
  - *Done when:* every headline table cell carries a CI, including the real-teacher
    ones.
  - *Effort:* 1 d. *Addresses:* Tier-1 #3.

### Phase B — Real input distribution (the core credibility fix)

- **B1. Benchmark workload adapter.** `BFCLWorkload` loading a BFCL-v4 subset:
  *simple*, *multiple*, *parallel*, and especially **relevance-detection** (this
  category *is* OAD's abstention contract, measured by a standard metric). Optional
  stretch: τ²-bench for multi-turn / user-side tools.
  - *Deliverable:* `bfcl_workload.py` exposing `stream()` + `ground_truth()` in the
    existing `Action = {tool, params}` shape.
  - *Done when:* the engine consumes BFCL prompts with no engine-side changes.
  - *Effort:* 3–5 d. *Addresses:* Tier-1 #1; also exercises the dormant
    `slot_extraction_method` gate branch (derived, non-verbatim params) → Tier-2 #6a.

- **B2. AST-match referee.** Replace exact-dict comparison with BFCL-style AST
  matching (function name + arg structure), so paraphrase-equivalent calls score
  correctly.
  - *Deliverable:* `Referee` variant with AST scoring; `calibrate()` now returns a
    meaningful <1.0 against a gold slice.
  - *Done when:* referee agrees with BFCL's own scorer on a held sample.
  - *Effort:* 2–3 d. *Addresses:* Tier-1 #1 (honest scoring), Turn-2 F-2 (referee no
    longer trivially 1.0).

- **B3. Run OAD on the real distribution.** Cost/coverage/silent-failure with CIs,
  for both teachers (GPT + Qwen). Re-run the **b=0 decomposition** here.
  - *Deliverable:* `results/bfcl_*` JSON + figs; an updated transfer table replacing
    the 51-unique run as the primary result.
  - *Done when:* the engine-added-error count `b` is reported with a CI on a
    distribution OAD's author did not write.
  - *Effort:* 2–3 d. *Addresses:* Tier-1 #1/#3, Turn-2 F-1; Tier-2 #6 (b under
    real conditions).

### Phase C — Real baselines (competitiveness)

- **C1. Semantic-cache baseline.** `SemanticCache`: MiniLM embedding + cosine
  threshold serve, else defer to teacher. Self-contained, no API key (prototype
  against oracle/noisy first).
  - *Deliverable:* `baselines.py::SemanticCache`.
  - *Done when:* it runs on the same stream + referee as OAD.
  - *Effort:* 1–2 d. *Addresses:* Tier-1 #2.

- **C2. Cascade baseline (stretch).** FrugalGPT-style: cheap model → escalate on
  low confidence. Strengthens the "true peers" comparison.
  - *Deliverable:* `baselines.py::Cascade`.
  - *Effort:* 2–3 d. *Addresses:* Tier-1 #2 (second peer). *MVP-optional.*

- **C3. Head-to-head Pareto.** The figure that makes the contribution land:
  **silent-failure rate vs model-call rate**, OAD vs semantic cache (vs cascade), on
  BFCL. Frame on the *safety* axis — the cache mis-serves boundaries silently where
  OAD abstains.
  - *Deliverable:* `results/figs/pareto_cost_safety.*` + table.
  - *Done when:* the plot shows OAD on/under the cache's silent-failure curve at
    matched call rate.
  - *Effort:* 1–2 d. *Addresses:* Tier-1 #2; directly upgrades paper Fig. 3.

### Phase D — Safety hardening (make the headline robust)

- **D1. Noisy-executor ε-sweep of the fix.** Now meaningful because A2 made the
  executor honest. Quantify how much execution-signal noise the 0-silent guarantee
  tolerates and how fast precision cost rises.
  - *Deliverable:* `results/figs/fix_eps_sweep.*`.
  - *Effort:* 1–2 d. *Addresses:* Tier-3 #8; closes the "simulated guarantee"
    critique on F-2.

- **D2. Teacher non-determinism (temp>0).** Run the real teacher at temperature>0,
  measure (i) OAD's determinism value vs an inconsistent teacher, (ii) whether b=0
  survives teacher self-inconsistency.
  - *Deliverable:* a determinism/consistency table.
  - *Effort:* 1–2 d. *Addresses:* Tier-2 #6b; supports the determinism axis of the
    abstract. *MVP-optional.*

- **D3. Re-establish b=0 (engine-added=0) on BFCL with CIs.** The strong result,
  now on a real distribution and multi-seed. (Output of B3 + A3; listed separately
  because it is the load-bearing claim.)
  - *Done when:* `b` mean ± CI is reported and discussed honestly if it is no longer
    exactly 0.

### Phase E — Positioning & theory depth

- **E1. Conformal / selective-prediction calibration of the acting bar.** Replace
  the fixed `p_act/c_act` with a calibrated abstention criterion giving a
  distribution-free, finite-sample silent-failure target. Cite selective
  classification (Geifman & El-Yaniv), learning-to-defer (Mozannar & Sontag),
  conformal prediction (Angelopoulos & Bates).
  - *Deliverable:* `calibration.py` + a short "calibrated abstention" subsection.
  - *Done when:* the user can set a target silent-failure rate and the system meets
    it with a stated guarantee on held data.
  - *Effort:* 3–5 d. *Addresses:* Tier-2 #5 + the paper's own "principled,
    calibrated abstention" future-work item. *MVP-optional but high-value.*

- **E2. Related-work expansion (writing only, do anytime).** Add: program-by-example
  / version-space (FlashFill; Lau & Weld) for the induction; cascades/routing beyond
  FrugalGPT; semantic-cache lineage; the selective-prediction line from E1.
  - *Effort:* 1–2 d. *Addresses:* Tier-2 #7.

### Phase F — Paper polish

- **F1. Auditability figure.** A reason-distribution chart and/or a sample audit log
  making the abstention-with-reason concrete. *Addresses:* Tier-3 #9.
- **F2. Claim→evidence map.** Tie each of the three abstract axes
  (cost / determinism / auditability) to a specific figure or table. *Addresses:*
  Tier-3 #10.
- **F3. Restructure results.** Promote BFCL to the primary result; demote the
  synthetic oracle run to "mechanism validation" (appendix or a short early section).
- **F4. Limitations pass.** Move resolved items out of "open"; keep the honest tone.
  - *Effort (F1–F4):* 3–4 d total.

### Phase G — Submission

- **G1. Venue choice.** Target a **trustworthy / reliable-agents**-angled workshop
  over a pure-efficiency one — OAD's most novel axis is abstention + auditability,
  not cost (which the paper itself calls "trivial" vs always-call).
- **G2. Format + OpenReview submission.** NeurIPS 2026 workshops: notification is
  **mandatory before Sept 29, 2026**; contributed deadlines land ~late-Aug to
  mid-Sept; workshops are Dec 11–13. Aim for "under review" status to support the
  Yonsei Spring-2027 application.

---

## 3. Dependency graph (critical path in **bold**)

```
A1 ─┐
A2 ─┼─► (independent, parallel)
A3 ─┘
**A1 ──► B1 ──► B2 ──► B3 ──► C3 ──► F ──► G**
                 C1 ──────────► C3
                 C2 ──────────► C3   (optional)
A2 ──► D1
A1+B1 ──► D2            (optional)
B3+A3 ──► D3
(B3) ──► E1             (E1 optional; can prototype on synthetic earlier)
E2, F1, F2 ── writing, schedulable anytime
```

Critical path: **A1 → B1 → B2 → B3 → C3 → F → G**. Everything else hangs off it in
parallel.

## 4. MVP cut line (if the calendar collapses)

Non-negotiable core that clears the four Tier-1 blockers and the honest-executor fix:

> **A1, A2, A3, B1, B2, B3, C1, C3, F3, F4, G.**

Defer if needed: **C2** (cascade), **D2** (temp), **E1** (conformal — high-value but
heavy), **D1** can ship as a single ε point rather than a full sweep. Even the MVP
gives: a real distribution, a real baseline on the cost–safety axis, an honest
execution signal, and CIs — which is the difference between "preliminary" and
"accepted."

## 5. Indicative timeline (≈9 weeks: July → early Sept)

| Weeks | Focus | Output |
|---|---|---|
| 1–2 | Phase A (A1, A2, A3) | reproducible dual-teacher harness, honest executor, CI machinery |
| 3–4 | Phase B (B1, B2) | BFCL adapter + AST referee |
| 4–5 | B3 + C1 | OAD-on-BFCL numbers + semantic cache |
| 5–6 | C3 + D1 (+ D3 reporting) | cost–safety Pareto, ε point, b on real data |
| 6–7 | E2 + E1 (if MVP allows) + D2 (optional) | positioning, calibrated abstention |
| 7–8 | Phase F | restructured paper, figures, limitations |
| 8–9 | Phase G | format, OpenReview submit before workshop deadline |

## 6. Risks & mitigations

- **BFCL params aren't verbatim copies** → induction may degrade. *Mitigation:* this
  is the point (it finally exercises `slot_extraction_method`); report the
  defer-rate honestly rather than tuning it away.
- **b=0 may not hold on BFCL.** *Mitigation:* that's a *finding*, not a failure —
  report engine-added error with a CI and analyze where it enters. A non-zero,
  characterized b is more credible than a suspiciously perfect 0.
- **Open-weights teacher much weaker than GPT** → low accuracy ceiling.
  *Mitigation:* report both; the transfer claim is about the *mechanism* tracking
  whatever ceiling the teacher sets, not about a high ceiling.
- **Conformal calibration (E1) overruns.** *Mitigation:* it is MVP-optional; ship the
  fixed-threshold version and frame conformal as the principled successor.

## 7. Traceability — every prior item is here

| Source item | Covered by |
|---|---|
| Tier-1 #1 real benchmark | B1, B2, B3 |
| Tier-1 #2 real baseline (semantic cache / cascade) | C1, C2, C3 |
| Tier-1 #3 multi-seed + CIs | A3, D3 |
| Tier-1 #4 open-weights teacher | A1 |
| Tier-2 #5 selective-prediction / conformal abstention | E1 |
| Tier-2 #6 harder b=0 (slot-extraction fires, temp>0) | B1 (derived params), B3, D2, D3 |
| Tier-2 #7 program-by-example / version-space positioning | E2 |
| Tier-3 #8 noisy-executor sweep of the fix | D1 |
| Tier-3 #9 auditability figure | F1 |
| Tier-3 #10 claim→evidence map | F2 |
| Turn-2 F-1 thin real-teacher eval | B (whole phase) — reframed as mandatory |
| Turn-2 F-2 oracle-backed "independent" executor | A2, D1 |
| Preserve list | §1 |
| Venue / timing / Yonsei link | G1, G2, §5 |
