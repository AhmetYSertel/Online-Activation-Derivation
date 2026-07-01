# RESULTS — Online Action Derivation (OAD)

Numbers below are reproduced by the drivers in `oad/`. **Read `LIMITATIONS.md` first** —
these validate a mechanism under the synthetic oracle teacher, not transfer to real
agent traffic.

## Phase 1 — mechanism validity (`oad.experiments_phase1`)
Single representative seed (P1, P3) / 5 seeds (P2). CPU-only, no trained model.

- **P1 derivation.** Autonomous coverage 0% → **86.5%**, accuracy when acting **100%**,
  **0 silent failures**, sample-efficiency **10 family-occurrences** to first autonomous
  replay. No templates hand-written.
- **P2 gate.** In-domain act-correct **100%**; adversarial abstain **100%**; OOD abstain
  **100%**; **0 silent failures**. Every abstention is auditable: adversarial reasons
  decompose into `novel_content_word` (35) and `no_match` (5).
- **P3 slot induction.** Promotion: literal `weather in istanbul` → `weather in <0>` →
  unseen `weather in cairo` acted correctly. Split: `open the account` (different tool)
  not folded into the smart-home slot; separate regularity spawned; specificity routes
  both correctly. Characterized limitation: an *unseen* surface-identical boundary acts
  and fails silently.

## Phase 2 — what it is good for (`oad.experiments_phase2`)
~10K sequential calls; the batch is a delivery mechanism, not a parallel snapshot.

| | LLM-call rate | accuracy | silent failures |
|---|---|---|---|
| always-LLM | 100% | 100% | 0 |
| **OAD** | **16.4%** | **100%** | **0** |
| eager-no-gate | 7.9% | 92.2% | 471 |

- **Drift** (weather action renamed at call 5000): weather accuracy dips, the decayed
  posterior falls below the acting bar, the system abstains and re-derives; weather
  accuracy back to 100% **~90 calls** after drift.
- **Scissors** (drift + noisy execution, ε = 0.6): believed vs true accuracy on the
  drifted family. SAFE late-window gap **≈ 0** (recovers); ABLATION (eager + self-train +
  no-gate) gap **≈ 60 pts** and never recovers.

## Statistical rigor (`oad.experiments_rigor`)

**R1 — multi-seed 95% CIs (10 seeds).** OAD accuracy **100.0 ± 0.0**, LLM-rate
**17.3 ± 0.2**, silent-rate **0.0 ± 0.0**; eager silent-rate **7.6 ± 0.2**; drift
recovery **141.8 ± 12.5 calls**; scissors gap SAFE **−1.0 ± 0.0** vs ABLATION
**59.2 ± 1.7**. The headline results are not single-seed flukes.

**R2 — execution-noise sweep (ε ∈ {0…0.8}, 5 seeds).** SAFE gap stays at **≈ −1 across
all ε**; ABLATION gap grows linearly **5.7 → 23.0 → 40.7 → 59.4 → 80.3**. The safety
property holds across the entire noise spectrum, not just at one ε. (The ablation gap
tracks ≈ f·ε, the blast-radius × false-confirm-rate product.)

**R3 — component ablations (6–8 seeds).**
- *Gate is load-bearing for junk:* silent-failure rate on junk traffic is **7.6%**
  without the gate, **0.0%** with it.
- *The acting/maturity bar is the primary safety lever:* making acting eager *alone*
  opens the scissors to **59.0 pts** (gate and decay still on). **Self-training alone is
  harmless** (gap **−1.0**) — it only bites once acting is unrestrained (full ablation
  **59.0**). Recency **decay aids recovery** (without it, residual gap **7.7**).

**R4 — teacher-noise robustness (δ ∈ {0…0.3}, 5 seeds).** With an imperfect (frozen-LLM-
like) teacher, the full design holds silent-failure rate at **0.0 → 1.7%** as δ grows to
0.3; eager-no-gate degrades to **22.5%**. The maturity bar + specificity reject sparse
bad labels.

## Honest bottom line
Mechanism validated; payoff and safety margin demonstrated with tight CIs and across a
noise sweep; safety attributed to the acting/maturity bar (primary) plus the gate, with
decay aiding recovery and self-training harmless in isolation. The path from here to a
defensible *transfer* claim is in `LIMITATIONS.md` — chiefly: a real GPT teacher (adapter
included), a real tool-use benchmark, and one real cost-reduction baseline.

Reproduce:
```
pip install -r requirements.txt
python -m oad.experiments_phase1
python -m oad.experiments_phase2
python -m oad.experiments_rigor
# real GPT teacher (your environment): set OPENAI_API_KEY, pip install openai,
# then swap teacher.OracleTeacher -> teacher.OpenAITeacher(model="gpt-5.5")
```
