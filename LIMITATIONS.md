# LIMITATIONS — read before citing any number

This file is deliberately prominent. The OAD results are a clean validation of a
*mechanism* and its safety properties under controlled stress. They are **not** yet
evidence of transfer to real agent traffic. Every limitation below is also surfaced
in the relevant module's docstring (`workload.py`, `harness.py`, `teacher.py`,
`engine.py`).

## 1. The teacher is a synthetic oracle, not a real LLM
`teacher.OracleTeacher` returns the workload's exact correct action. It never makes the
mistakes a real frozen LLM makes (label noise, paraphrase confusion, tool-selection
errors, hallucinated parameters). **What this means:** the cost/coverage, drift, and
scissors results show how the *system* behaves given perfect supervision — they do not
show how it behaves distilling a real model.
- *Partial mitigation already in the repo:* `teacher.NoisyTeacher` injects label noise
  δ, and experiment R4 reports silent-failure rate vs δ (full design stays ≤ ~2% up to
  δ = 0.3; no-protection degrades to ~22%). This stresses the channel but still uses a
  synthetic base.
- *What is needed:* run with `teacher.OpenAITeacher` (a real GPT model — default
  `gpt-5.5`) as the teacher. The adapter is implemented and ready; it is **not** run in
  the packaged results because the build environment has no API key.
- *DONE (2026-06-30, gpt-5.5, see [`results/REAL_TEACHER_FINDINGS.md`](results/REAL_TEACHER_FINDINGS.md)):*
  the cost reduction transfers (OAD answers at the teacher's 91.7% accuracy on only 11.4%
  of calls), but the **"0 silent failures" headline does not** — OAD commits 109 silent
  failures because the real teacher emits surface-coherent boundary errors (e.g.
  `weather in 1945` → `get_weather(1945)`), which OAD faithfully learns and replays and
  which the gate cannot flag. This makes #2 (independent referee) a prerequisite for any
  safety claim, not just a nicety: the teacher-backed confirmer used here cannot detect a
  teacher error by construction.

## 2. Referee independence is structural, not informational
`harness.Referee` is backed by the same ground-truth oracle as the teacher, so its
reliability on a gold set is 1.0 *by construction*. The independence is that the engine
has no import path to it (it never reads the referee) — **not** that the referee is a
separately-validated judge. With a real teacher, the referee must become execution-based
or human-validated, and `Referee.calibrate()` would then report a meaningful (< 1.0)
agreement that could catch criterion errors.
- *CONFIRMED + FIXED (2026-06-30, see [`results/REAL_TEACHER_FINDINGS.md`](results/REAL_TEACHER_FINDINGS.md)):*
  the real-teacher run showed the in-loop production-confirmation channel
  (`execute = produced == teacher(text)`) is a **tautology** under a learned policy — it only
  measures self-consistency with the teacher and seals the teacher's confident errors into the
  posterior (all 109 silent failures were teacher errors; OAD added 0 of its own). Replacing it
  with a teacher-independent execution signal AND gating template-induction on that same signal
  (`Engine(confirm_learning=True)`) drives silent failures 109 → 0 while preserving legitimate
  coverage; the inherited errors move into visible LLM-call abstention. Independent confirmation
  is therefore a prerequisite for the safety claim, not an enhancement.

## 3. The workload and adversarial probes are author-written
Six families, single-token-ish slots, and eight adversarial probes — all authored by the
same hand as the system. The adversarial set is **illustrative, not adversarially
optimized**, and the families are simpler than real tool-call distributions. A held-out,
third-party, or red-teamed probe set, and a real tool-use benchmark (e.g. a BFCL /
API-Bank / ToolBench subset) as the input distribution, are future work.

## 4. Baselines are reference points, not strong competitors
`always-LLM` and `eager-no-gate` are sanity baselines. They establish that OAD is both
cheap and safe, but they are not the real cost-reduction methods OAD should be measured
against: semantic/embedding action caches (e.g. GPTCache-style), LLM cascades/routing
(e.g. FrugalGPT-style), or the white-box sibling (WDR). A publishable cost-quality claim
needs at least one of these on the same axis.

## 5. One gate check is latent in this regime
The `slot_extraction_method` check (refuse replay when no parameter can be derived
deterministically from slots) never fires on this workload, because every family's
parameter is a verbatim copy of its slot. It is implemented and kept for completeness,
but a reviewer will note it is inactive here; a workload with `transform`/`semantic`
parameter derivations is needed to exercise it.

## 6. The characterized failure mode: unseen surface-identical boundary
When an input shares a learned skeleton and a plausible filler but maps to a different
tool, and it has **never been seen**, the system acts and fails silently
(`open the account` → `smart_home(account)`). This is irreducible for a black-box
surface system and is the explicit motivation for the white-box sibling (WDR). It is
characterized qualitatively (Phase 1, P3); it is **not** yet quantified as "what fraction
of a realistic input space is surface-ambiguous."

## 7. Causal attribution of the safety (what the ablations actually show)
The R3 ablations show the **acting/maturity bar is the primary safety lever**: making
acting eager (low bar) *alone* opens the self-poisoning scissors (~59 pts) even with the
gate and decay still on. Self-training *alone* is harmless (gap ~ -1) because the bar
still gates acting; it only compounds the failure once acting is already unrestrained.
Recency decay contributes to *recovery* (without it a small residual gap ~8 pts
persists). Earlier framing that called the self-training asymmetry "the" lever is
corrected by this data — the safety is a bundle, with the acting bar load-bearing.

## 8. Not yet a paper
This repository is the experimental substrate and an honest results record. Abstract,
introduction, related work, formal method description, and the WDR positioning are not
written. A workshop submission (~4–8 pages) is still mostly writing work on top of this.

---

### What would move each claim from "mechanism" to "evidence"
1. Real GPT teacher (`OpenAITeacher`) + execution/human referee → transfer of the
   cost/coverage and drift results.
2. A real tool-use benchmark as the input distribution → external validity.
3. One real cost-reduction baseline on the same cost-quality axis → competitiveness.
4. A workload that activates `slot_extraction_method` and a quantified surface-ambiguity
   rate → completeness of the gate story and of the limitation.
5. Held-out / red-teamed adversarial probes → robustness of the gate-accuracy result.
