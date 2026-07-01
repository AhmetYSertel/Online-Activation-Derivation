# Online Action Derivation with Auditable Abstention
### Architecture Specification and Implementation Plan

**Author:** Ahmet Yiğit Sertel
**Status:** Working specification — supersedes the *Hybrid Governance Architecture (HGA)* framing
**Target:** NeurIPS 2026 workshop submission (under-review milestone for Yonsei Spring 2027)

> **Naming is open.** This document uses *Online Action Derivation (OAD)* as a working placeholder.
> Alternatives to consider: *Confirmation-Gated Action Replay*, *Self-Derived Action Policy*,
> *Auditable Action Distillation*. The thesis below does not depend on the name.

---

## 0. What changed, and why this is a different paper

The original HGA document was a three-layer **cost-reduction memory substrate**. This specification keeps
only the L3 idea (decision replay) and rebuilds the system around a single, narrower thesis. The shift is
not a subtraction; it is a change of identity. Cost savings, deterministic output, and coverage are no
longer *claims* — they are *consequences* of the thesis.

Crucially, the headline numbers from HGA (78.1% / 89.5% token savings) **cannot be carried over**. Those
figures were co-produced with the L1 routing/context-filtering layer; without it, the cold-start savings
floor drops to zero and only the mature-phase replay cost survives. The L3-only configuration must be
re-measured from scratch. This is expected and accepted.

---

## 1. Decision ledger (locked)

These five decisions are settled and frame everything below.

**D1 — Identity.** The paper is not a memory or cost architecture. It is an **online-derived,
auditable-gated, abstention-capable action policy**: a system that derives an action policy online from a
frozen teacher LLM's input/output behaviour, decides *whether to act or abstain* through a fixed,
auditable gate, and withdraws to the full LLM whenever it is not sufficiently certain. The "Hybrid
Governance" and "memory substrate" framing is dropped.

**D2 — Validity backbone.** Three separate confirmation layers (production / learning / referee) and a
single mathematical threshold located only at the learning gate. Numbers are re-measured for the
L3-only configuration; no figure is inherited.

**D3 — Positioning.** Black-box. This is the **black-box sibling** of WDR (which is white-box). The two
answer one question — *when can an agent safely replay a previously made decision?* — from two
surfaces: OAD from auditable surface features, WDR from model-internal signals. Cross-referenced, not
overlapping.

**D4 — Evaluation.** Two phases, two questions. Phase 1: *how does it work* (mechanism validity). Phase
2: *what is it good for* (deployment value), via a **sequential** batch of ~10,000 synthetic agent calls
(~30 min wall-clock), with drift injected at a known point and recovery shown.

**D5 — Referee.** Verification harness as primary referee, calibrated once against a small human gold
set, with real executable tools wherever reachable. The referee never touches the learning loop or the
gate decision.

---

## 2. Architecture

### 2.1 What the system is

A frozen teacher LLM produces (input → action) pairs in the course of normal operation. The system
learns the regularities in these pairs **online** — i.e. it keeps updating from its own
execution-confirmed outputs as it runs. The live, adapting part is **not a trained neural model**: it is
the online-updated symbolic structure (induced skeletons and slots) and the online statistics (success
posteriors, frequencies). Action emission itself is deterministic — once an input is matched to a
skeleton and its slots are filled, the action is produced by template instantiation, not by a learned
generator, which gives the deterministic-output property for free (identical input → identical action by
construction). The decision of *whether to act* is made by a fixed, auditable gate. This is "machine
learning" in the classical sense — a system that improves from experience — with **no backpropagation, no
GPU, and no transformer at its core**. (The teacher is a transformer, but it is frozen and is only the
*source* of I/O; the derived system that learns is classical.) This is what keeps the system black-box
and its safety auditable.

Two properties are derived, not assumed:

- **Selective correctness.** The system acts only when an auditable gate clears it; otherwise it abstains
  and routes to the full LLM. The claim is not "cheaper" but "correct when it acts, silent when unsure."
- **Coverage that grows by derivation, not by a fixed schema.** Templates are not hand-written. They are
  *induced* from confirmed I/O, so the set of situations the system can handle expands as it encounters
  more confirmed regularities. A locked schema would cap coverage; an induced one does not.

### 2.2 The three confirmation layers (validity backbone)

This is the spine of the paper's validity. Conflating any two of these reintroduces the circularity the
original HGA methodology note already admitted to fixing ("circular validation, gate-dependent ground
truth").

1. **Production confirmation (loose).** The gate permits an action; the action executes; a result
   returns. This is the everyday signal the running system uses.
2. **Learning confirmation (strict).** A pair enters the *training* of the output organ only when it
   clears a much higher bar than production. This asymmetry — production threshold ≪ learning threshold —
   is the primary mechanism that suppresses online drift / self-poisoning. The system does **not** learn
   from everything it acts on; it learns only from the cleanest subset of what it acts on.
3. **Evaluation referee (out-of-loop).** A separate verifier (the harness) that touches neither the
   learning loop nor the gate. Every accuracy number in the paper comes from it.

The hard rule for layer 2: learning confirmation must come from a **model-external** signal — strongest
is real execution (the world confirmed it), supported by the gate's auditable checks. The output organ's
*own* confidence never counts as a vote for its own training. Otherwise the model feeds its own
conviction back to itself — circular by construction.

### 2.3 Derivation mechanism (how the policy is induced)

The system starts with **zero** templates. Regularities are learned implicitly in the model's parameters;
the word "cluster" below is an explanatory metaphor — **no explicit clustering is performed**, and the
paper must say this in one sentence to forestall the obvious reviewer question.

**Cluster identity = skeleton.** A "cluster" is the set of inputs requiring the same action. Its identity
is defined by the input's **skeleton** (its surface structure with variable positions abstracted), not by
semantic distance. Semantic neighbourhood (embedding similarity) is used only as a *candidate-finding
filter* to narrow which existing regularities a new input might match. Identity is skeleton; embedding is
a pre-filter. Using embedding alone is dangerous — semantically close but action-different inputs ("Berlin
weather" vs "Berlin weather history 1945") would collapse into one cluster.

**Assignment at runtime.** New input → embedding finds nearest candidate regularities → the skeleton must
match → slots must be fillable. Three outcomes: skeleton matches and slots fill → belongs; near candidate
exists but skeleton fails → does not belong, new-cluster candidate; no near candidate → genuinely new.
Assignment is graded, not binary: skeleton similarity + slot fillability + semantic proximity together
produce an **assignment confidence**. Low-confidence assignment = no assignment, fall back to full LLM.
This is the abstention boundary applied at the *assignment* stage, not only at the action stage.

**Slot induction is a cross-example operation.** A single observation cannot tell you which token is the
variable — "weather in Berlin" gives no way to know whether *Berlin*, *weather*, or *in* is the slot. The
variable position only emerges once a second/third example arrives: the position that changes
(Berlin→Tokyo) becomes the slot, the positions that stay constant become the skeleton. Consequence: a
newly born cluster is necessarily **over-specific** (treats the whole input as fixed) and **loosens** as
examples accumulate. The generalization direction is one-way — narrow → wide — which is the safe
direction: while narrow you under-cover, you do not mis-act.

**The slot-promotion rule (this is the heart of the mechanism).** A position is promoted to a slot only
if, when it varies, **action-correctness is preserved** (execution-confirmed). Berlin→Tokyo varied and
`get_weather` still produced a correct result → that position is genuinely a slot. If varying a position
*breaks* action-correctness → that position is not a slot but a **cluster boundary**; it starts a new
cluster. This is what turns "the cluster changes as similar inputs arrive" from blind expansion into
**confirmed expansion**: slot candidates become permanent only when execution-confirmed, otherwise the
cluster splits. Automatic derivation is preserved, but it never generalizes blindly.

A useful side effect: slot *names* are irrelevant. A slot's identity is the functional definition "a
position whose variation does not break action-correctness." No naming dependency.

**Two distinct derivation events, two thresholds:**

- **Slot promotion** — does this position vary freely without breaking correctness? (a structure question)
- **Cluster acting-authorization** — has this cluster accumulated enough confirmation and is it internally
  action-consistent (no split pending)? (a confidence question)

These must stay separate, because one learns the *schema* and the other learns *when to act*. Merging them
would put a low-example-but-consistent cluster and a high-example-but-unsettled cluster in the same bucket.

### 2.4 The fixed auditable gate

Simplified from the original five-path governance gate. With the deterministic vault (L2) gone, the
"exact key → DeterministicPath" and "sensitivity=HIGH → force exact recall" branches lose their meaning.
The sensitivity classifier may stay, but high-sensitivity now means **fail-safe to full LLM**, not "go to
vault."

What survives, and what makes the system auditable, are the three coherence checks (the source of the
original "zero silent failures over 300 adversarial probes" result):

- **Slot-extraction-method check** — fails the replay if all slots were derived by embedding-difference
  alone (no exact or template match). Purely-semantic slot derivation is exactly where the system could
  act confidently-but-wrong, so it is barred from autonomous action.
- **Template-skeleton check** — compares the input skeleton against known templates; falls back to full
  LLM if similarity drops below threshold.
- **Novel-content-word check** — triggers fallback when too large a fraction of content words are unseen.

The value of these checks is precisely that they are **explicit and inspectable**: when the system
abstains, it can say *why* (skeleton failed, content-word novel). That is what preserves the
selective-correctness argument and what keeps OAD distinct from WDR's opaque, model-internal confidence.

### 2.5 The single mathematical threshold

All thresholds collapse to one place: the **learning-confirmation gate**. Everywhere else, the fixed rules
of §2.3–2.4 handle "when to act." The only quantity to derive mathematically is "is this regularity
certain enough to be worth learning from?"

Mechanism: a **Beta–Binomial posterior** over each regularity's success probability *p*, updated from
**independent (execution) confirmations**. A regularity enters the output organ's training only when

```
P(p > p_target) ≥ confidence_target        e.g.  P(p > 0.99) ≥ 0.95
```

This replaces the arbitrary "15 successful hits" constant of HGA: the number of observations becomes an
*output* of the chosen error budget, not a hand-set input. As a one-line intuition for the text, the
rule-of-three bound (0 failures in *n* trials ⇒ true error ≤ ~3/*n* at 95%) makes "why this many and not
15" answerable in a sentence.

**Conformal / risk–coverage guarantees are deferred** to future work for the first paper. They would give
a distribution-free coverage guarantee but require a held-out calibration set and exchangeability
bookkeeping. The Beta–Binomial layer plus the Phase-2 durability curve is defensible on its own for a
workshop submission.

### 2.6 Online drift control (a consequence, not a separate claim)

Because the system adapts online from its own execution-confirmed outputs, the gate's false positives
become the next adaptation signal. There is no weight drift to worry about (there are no weights), but the
symbolic/statistical layer can still poison itself in two ways: an over-broad **slot promotion** (a
position promoted to slot when it should have been a cluster boundary, making the cluster
action-inconsistent) and an **inflated success posterior** (a regularity's success statistic updated by
false-positive confirmations, authorizing it to act when it should not). Either way a wrong regularity
gets reinforced and acted on more readily next time — **self-poisoning** in the classical-ML sense. The
job of the learning-confirmation threshold is therefore not "pick good examples" but **keep the
feedback-loop gain below 1**.

Three levers do this together: the **asymmetry** (§2.2 layer 2), the **independent-confirmation
requirement** (model's own confidence cannot vote for its own training), and **hysteresis/cooldown**
(confirmed pairs accumulate before becoming small, delayed weight updates; no single-example update).

The observable signature of failure is a **scissors**: the model's *self-confidence* rising while the
*referee's accuracy* falls. Phase 2's durability result is simply showing this scissors **does not open**.
This is a reading of the accuracy-over-call-index curve, not a separate experiment.

### 2.7 Relationship to WDR

OAD and WDR are siblings, deliberately cross-referenced. Both answer: *when can an agent safely replay a
previously made decision?* OAD answers from **black-box, auditable surface features** (template skeleton,
content-word novelty, slot-extraction method, plus the externally-confirmed success posterior). WDR
answers from **white-box, model-internal signals** (activation signatures, logit margins, interventional
slot validation). Stated this way the two read as coverage, not repetition — and the distinction must be
drawn explicitly in both papers so the Yonsei portfolio does not look redundant.

---

## 3. Evaluation design

### 3.1 Phase 1 — How it works (mechanism validity)

Goal: show, against the out-of-loop referee, that derivation actually happens and happens correctly.

- **Derivation curve.** Start with zero templates; let confirmed pairs accumulate; plot, against the
  harness, how many independent confirmations a regularity takes before it produces correct output. This
  is the direct evidence for sample-efficiency and for "automatic derivation works." It contains no
  hand-written component, so it protects the novelty.
- **Gate accuracy.** Show the fixed gate acts at the right moments and abstains at the right moments —
  act in-domain, withdraw out-of-domain (adversarial + OOD). This is where the old E3 lives: abstentions
  with an auditable reason, zero silent failures.
- **Slot induction.** Show that a position's promotion to slot is execution-confirmed, and that an
  unconfirmed one splits as a cluster boundary — i.e. the schema is induced from data, not written by hand.

### 3.2 Phase 2 — What it's good for (deployment value)

Goal: show what the live system gains in realistic use, and that it does not degrade.

Setup: a **sequential** batch of ~10,000 synthetic agent calls, ~30 min wall-clock, packaged. The batch
is a *delivery mechanism*, not a parallel evaluation of a frozen snapshot. **The paper must state this
explicitly:** calls are processed sequentially; the model updates online after each confirmed
interaction; the 30-minute figure is wall-clock batch execution, not parallel evaluation of a frozen
model. Otherwise a reviewer reads "10,000 calls in 30 minutes" as a frozen snapshot and the online claim
collapses.

- **Coverage × accuracy.** How much the system takes over (coverage) over the call sequence, at what
  harness-accuracy. The risk–coverage trade-off in one curve.
- **Online durability.** Harness-accuracy tracked along the call axis. Drift would show here; its absence
  is the evidence that the learning-confirmation asymmetry works. A reading of the curve, not a new
  experiment.
- **Drift recovery.** Distribution shift injected at a known point: the system first abstains (precision
  held), then — once a new regularity is derived — coverage returns without sacrificing precision. This is
  the part that demonstrates beating locked-schema brittleness by derivation.

**Synthetic generation constraints (these matter for validity):**

- The source generating the 10K calls must be **independent of the teacher** the system distills from
  (otherwise circular).
- The repetition structure (how many distinct regularities, at what frequency) must be **deliberately
  designed and reported** — the coverage number is a direct output of it, so it is an experiment
  parameter, not a hidden assumption. A Zipfian intent distribution is the realistic and defensible choice.

### 3.3 The referee

Harness primary; a small human gold set calibrates the harness once (so "how reliable is the referee" is
answered up front); real executable tools serpentined in wherever reachable so Phase 2 has at least one
real-world hook. Explicitly **not used**: LLM-as-judge alone (judge is itself noisy/biased and not
independent of the teacher family), and execution-success-as-truth (in-loop, and accepts
wrong-but-non-crashing actions — the exact failure mode being guarded against).

### 3.4 Two open operational gaps (writing, not design)

1. **Harness correct-action criterion.** Per tool, what counts as a correct action (expected call
   structure / result). Must be defined independently of the system.
2. **10K-call regularity-distribution design.** The deliberate repetition structure described in §3.2.

Neither is an architectural risk; both are content to fill before experiments run.

---

## 4. Implementation plan

The plan is organized as work packages. Sequencing and the critical path follow. Existing assets to reuse
are flagged — the goal is to lean on the `rm_bench` harness pattern and the distillation-track
verification harness rather than rebuild infrastructure.

### The output organ is not a trained model (resolved)

The "live ML mechanism that produces output" is classical and CPU-only — **no trained neural model, no
transformer being fine-tuned, no GPU.** Action emission is deterministic template instantiation: once an
input is matched to a skeleton and its slots are filled, the action is produced by filling the action
template. The *learning* is entirely in the online-adapting symbolic structure (skeletons, slots) and
online statistics (Beta–Binomial posteriors, frequencies). This is machine learning in the classical
sense — improvement from experience — not deep learning.

This resolution improves three things and relocates one risk:

- **Determinism is free** — identical input → identical action by construction; the old E14 claim becomes
  trivially true rather than something to demonstrate against a stochastic LLM.
- **Auditability sharpens** — there is no learned black-box generator anywhere; every output traces to a
  matched skeleton + extracted slots + a fixed action template.
- **Separation from WDR widens** — WDR reasons about model internals; OAD has no trained model at its core
  at all.
- **Self-poisoning relocates** from (non-existent) weight drift to the symbolic/statistical layer (§2.6) —
  still real, still the thing Phase 2 must show does not happen, but mechanically different.

The one remaining neural component is the **frozen sentence encoder** used only as the candidate
pre-filter for assignment (MiniLM-class, runs fine on CPU). It is not trained by us and is not the "ML" of
the system — it is a retrieval utility. If you want the system fully non-neural, it can be replaced by
lexical skeleton matching (token-overlap / edit-distance over the skeleton), at some cost to paraphrase
robustness. **Open choice:** frozen-encoder pre-filter (better paraphrase handling, one CPU-only neural
utility) vs purely lexical matching (fully non-neural, weaker on paraphrase).

### Work packages

**WP1 — Core derivation engine.**
Skeleton extraction; embedding pre-filter for candidate regularities; slot induction as a cross-example
diff; the implicit/parametric "cluster" store; the slot-promotion rule (execution-confirmed) and
cluster-split on boundary. *Deliverable:* an engine that, given a stream of confirmed (input → action)
pairs, induces skeletons and slots and assigns new inputs with a graded assignment confidence.
*Decision gate:* the encoder-vs-lexical pre-filter choice above (small; does not block the rest).

**WP2 — Fixed auditable gate.**
The three coherence checks (slot-extraction-method, template-skeleton, novel-content-word); the
sensitivity classifier as fail-safe-to-LLM; the graded abstention boundary at both assignment and action
stages. *Deliverable:* a gate that, for any input, returns {act via replay | abstain to LLM} with an
inspectable reason string. *Depends on:* WP1.

**WP3 — Learning-confirmation threshold.**
Beta–Binomial posterior per regularity, updated from independent execution confirmations; the
`P(p > p_target) ≥ confidence_target` admission rule; the production-vs-learning threshold asymmetry made
explicit and configurable. *Deliverable:* a single function that decides whether a confirmed pair is
clean enough to enter the system's adaptation (slot promotion / posterior update), with the threshold
exposed as the one tunable knob. *Depends on:* WP1.

**WP4 — Online update loop.**
Sequential processing of an incoming call stream; hysteresis/cooldown so confirmed pairs accumulate into
small delayed updates of the induced structure and statistics; the three confirmation layers wired
together (production feeds running behaviour, learning feeds the adaptation, referee stays out).
*Deliverable:* a loop that runs a call stream end-to-end, updating the induced structure and statistics
online without single-example jumps. *Depends on:* WP1, WP2, WP3.

**WP5 — Referee harness.**
Per-tool correct-action criterion (gap §3.4.1); small human gold set to calibrate the harness and report
its own accuracy; hooks for real executable tools where reachable. Reuse the distillation-track
verification harness pattern (Linux-verified). *Deliverable:* an out-of-loop verifier that scores any
(input → action) pair as correct/incorrect, with a reported reliability figure from the gold set.
*Independent of:* WP1–WP4 (must stay out of the loop).

**WP6 — Synthetic workload generator.**
Teacher-independent generation of ~10K calls; deliberate, reported regularity distribution (Zipfian);
a known drift-injection point for the recovery experiment. Reuse `rm_bench` harness scaffolding for
multi-seed aggregation and manifest/reproducibility. *Deliverable:* a generator that emits a sequential,
parameterized call stream plus its documented distribution. *Independent of:* WP1–WP4.

**WP7 — Phase 1 experiments.**
Derivation curve (sample-efficiency vs harness); gate accuracy (act in-domain / abstain on adversarial +
OOD, zero silent failures with reasons); slot-induction validation (confirmed promotion vs boundary
split). *Deliverable:* the three Phase-1 results with figures. *Depends on:* WP1–WP6.

**WP8 — Phase 2 experiment.**
The 10K sequential batch; coverage × accuracy curve; online-durability curve (the scissors does not open);
drift-injection recovery. Explicit sequential-not-parallel framing in the writeup. *Deliverable:* the
Phase-2 results with figures, plus the methodological statement on batch semantics. *Depends on:*
WP1–WP6, and ideally WP7 first (Phase 1 surfaces whether slot induction and the slot-filler handle the
workload's slot types before Phase 2 runs the full stream).

### Sequencing / critical path

WP1 is the root; WP2 and WP3 depend on it and can proceed in parallel; WP4 integrates the three. WP5 and
WP6 are independent of the engine and can be built in parallel from day one (and *should* be, since the
correct-action criterion and the workload distribution are the two open gaps that block experiments).
WP7 runs once WP1–WP6 exist; WP8 follows WP7.

Critical path: **WP1 → WP4 → WP7 → WP8**, with WP5/WP6 as a parallel track that must finish before WP7.
The two gaps in §3.4 sit inside WP5 and WP6 — close them early.

### Compute and environment

The entire system runs on CPU on your existing setup (Ryzen 9800X3D, 32 GB, Python 3.12). There is **no
GPU requirement at any point** — the only neural component is a frozen CPU sentence encoder used for the
assignment pre-filter, and even that is optional (see WP1). The teacher LLM is queried via API for I/O
generation only; it is never fine-tuned. The whole pipeline — derivation, online adaptation, both phases —
is reproducible on a single workstation, which is also a clean reproducibility story for the paper.

### Risk register

- **Self-poisoning (highest).** Now a symbolic/statistical risk, not a weight-drift risk (§2.6):
  over-broad slot promotion and inflated success posteriors. Mitigated by WP3's asymmetry + WP4's
  hysteresis; *proven* by WP8's non-opening scissors. If the scissors opens, deliberately loosen then
  re-tighten the threshold to show it as the control variable — that turns a risk into an ablation.
- **Circularity (validity-critical).** Mitigated by the three-layer separation and by teacher-independent
  generation (WP6); the referee (WP5) must never enter the loop. This is the single most likely
  desk-reject cause if violated.
- **Correlated coherence checks.** The three checks derive from the same surface representation, so they
  are *not* independent. Do not claim independence; use the most conservative agreement rule (all must
  pass) and say so. Framing correlated checks as "independent votes" is a reviewer trap.
- **Phase-2 misread as frozen.** Mitigated entirely by the explicit sequential-batch statement in WP8.
- **Scope creep.** Two questions, two phases, small. Resist promoting drift-control, the posterior, or
  conformal into separate headline claims — they live *inside* the two phases.

---

## 5. Deliberately deferred (future work)

- Distribution-free coverage guarantees (conformal / online-adaptive conformal) over the admission score.
- A small online-finetuned student as the output organ (escalation path from the lightweight slot-filler).
- Multi-teacher / cross-model confirmation to genuinely raise independence beyond surface-derived checks.
- Long-horizon, real (non-batch) deployment durability.

These are honest extensions, not hidden dependencies. The first paper stands on the two-phase result with
the lightweight organ and the harness referee.
