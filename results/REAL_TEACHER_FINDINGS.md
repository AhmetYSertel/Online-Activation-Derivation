# Real-teacher transfer run — findings

This is the experiment [`LIMITATIONS.md`](../LIMITATIONS.md) #1 called for: the packaged
results all use the synthetic `OracleTeacher` (perfect by construction). This run swaps in
`teacher.OpenAITeacher` with a **real GPT model** and re-runs the cost/coverage experiment,
so the numbers reflect distilling a *fallible* frozen LLM.

Reproduce:
```
pip install openai
export OPENAI_API_KEY=sk-...
python -m oad.experiments_real_teacher --model gpt-5.5 --n 1500 --junk-frac 0.15 --seed 1
```
Outputs: [`real_teacher_results.json`](real_teacher_results.json),
[`figs/real_teacher/real_teacher_cost_quality.png`](figs/real_teacher/real_teacher_cost_quality.png).

## Setup
- **Teacher:** `gpt-5.5` (current GA flagship, 2026-06), via OpenAI function calling, cached.
- **Run:** n = 1500 sequential calls, 15% adversarial/OOD junk, seed 1.
- **API cost:** **51 real calls** — the stream is Zipfian over small-vocab families, so only the
  51 unique inputs hit the API regardless of stream length. The cache makes the run ~cents.

## Headline numbers (seed 1)
| system | LLM-call rate | accuracy | silent failures |
|---|---|---|---|
| always-LLM (real teacher) | 100% | 91.7% | 0 (125 teacher errors, but never replayed) |
| **OAD (full design)** | **11.4%** | **91.7%** | **109** |
| eager-no-gate | 7.9% | 91.7% | 122 |

Teacher raw accuracy on this distribution: **84.3%** (43/51 unique inputs).

## What transfers, and what does not

**Cost reduction transfers cleanly.** OAD answers at the teacher's own accuracy (91.7%) while
calling the LLM on only **11.4%** of inputs — an ~9× call reduction, holding quality at the
ceiling the teacher sets. This is the result the oracle run promised, and it survives a real
teacher.

**The "0 silent failures" headline does NOT transfer.** The oracle run reported 0 silent
failures; here OAD commits **109**. The reason is structural, not a bug:

1. **Every teacher error is an over-eager boundary miss.** All 8 unique errors are on the
   adversarial probes — `weather in 1945` → `get_weather(1945)`, `open the 12` →
   `smart_home(12)`, `weather in istanbul please cancel` → `get_weather(istanbul)`. The teacher
   gets **100% of clean in-domain inputs right** and **0% of the adversarial probes**, by
   confidently routing invalid slots to a real tool instead of abstaining. These are the
   real-LLM mistakes the oracle never makes.
2. **OAD faithfully learns the wrong-but-coherent template and replays it.** A teacher error
   like `weather in <digits>` → `get_weather` is surface-coherent: real tool, plausible
   filler. The gate's coherence + maturity checks guard against *premature / incoherent*
   acting — they do **not** flag a confidently-wrong label the teacher endorsed. So OAD
   internalizes the error and silently fails on every repeat.
3. **The production-confirmation channel can't catch it, because it is teacher-backed.** In
   this harness `execute(text, produced) == (produced == teacher(text))`. A faithfully-replayed
   teacher error *confirms* (the engine reproduced exactly what the teacher would have said), so
   the posterior stays high. This is a concrete demonstration of [`LIMITATIONS.md`](../LIMITATIONS.md)
   #2: with a real teacher you **must** replace the referee/execution channel with an
   independent execution-based or human-validated judge. A teacher-backed confirmer is blind to
   teacher errors by construction.

**Accuracy is teacher-bounded; the gate buys cost, not safety, under a fallible teacher.** All
three systems land at 91.7% because accuracy here is capped by the teacher. Under the oracle the
gate's value showed up as *safety* (0 vs 471 silent failures); under a real teacher whose errors
are surface-coherent, the gate's value collapses to *cost* (11.4% vs 100% calls) and the safety
gap to eager-no-gate nearly vanishes (109 vs 122 silent failures).

## Decomposition of the 109 silent failures (seed 1)
Splitting each silent failure by whether the teacher was *also* wrong on that input
(`python -m oad.analyze_silent_failures`, → [`silent_failure_breakdown.json`](silent_failure_breakdown.json)):

| class | count | unique inputs |
|---|---|---|
| (a) inherited — teacher also wrong | **109** | 8 |
| (b) OAD-added — teacher right, replay wrong | **0** | 0 |

And the error budget closes exactly: **109 silent (acted) + 16 abstain-and-wrong = 125 =
always-LLM's teacher-error count.** OAD and always-LLM commit the *identical* error set; the
only difference is character — always-LLM "asked and was misled," OAD "didn't ask and was
confidently wrong." **(b) = 0 means the engine adds no error of its own: the gate, slot
induction, and maturity logic are sound.** The entire defect is that the in-loop
production-confirmation channel is a **tautology**: `execute = (produced == teacher(text))`,
but `produced` was *learned from* the teacher, so re-asking the teacher trivially "confirms"
it, the posterior rises, and the gate permits acting. The channel carries **zero independent
information about correctness** — it only measures "am I consistent with the teacher." Under a
perfect teacher consistency = correctness, so it worked; under a fallible teacher, at exactly
the inputs where the teacher is confidently wrong, consistency ≠ correctness, and the posterior
*seals in* the teacher's errors. This is [`LIMITATIONS.md`](../LIMITATIONS.md) #2, proven
empirically.

## The fix has one real principle
Bring in a correctness signal **independent of the teacher**, and bind both
*action-promotion* and *template-induction* to that signal — not to teacher-agreement. You
cannot validate a cached teacher-action by asking the same teacher again; with no independent
signal, no gate/threshold tuning helps — you only re-derive the teacher's error surface.
Concrete channels, by leverage:
1. **Execution-based confirmation (strongest).** Actually run the action; let the world judge.
   `get_weather(1945)` errors/empties against a real weather API → confirmation negative →
   posterior does not rise → gate defers. Truth comes from the world, not the teacher. Build it
   as **two separate instances** — one in-loop confirmation channel and one out-of-loop referee
   — so the engine never reads the referee (the out-of-loop property must survive).
2. **Slot-domain induced from *confirmed* data.** Define the acceptable slot domain from
   independently-confirmed examples, not "everything the teacher labeled." (The oracle-era
   `novel_content` check broke under a real teacher precisely because the teacher's wrong labels
   polluted the "seen fills" set and drowned the OOD guard.)
3. **Region-wise teacher-reliability + defer-when-unconfirmed.** Track, per regularity, how often
   its teacher labels were independently confirmed; gate promotion on that, not self-consistency.
   Regions with no independent confirmation (adversarial/boundary) never promote to autonomous
   action → a permanent defer zone → no silent failures originate there.
4. **Independent verifier / human signal (weak, complementary).** A second model that did not
   produce the action, or user corrections. Shares many biases, so weaker than execution.

Honest target after the fix: not "0 silent failures" but **"0 silent failures added beyond the
teacher's independently-confirmed-correct rate"** — i.e. (b) stays 0 and the (a) errors move
into abstention instead of autonomous action.

## The fix, demonstrated (`python -m oad.experiments_exec_confirm`)
Binding the engine to a teacher-INDEPENDENT signal, at two binding points
(→ [`exec_confirm_results.json`](exec_confirm_results.json)). The independent signal is the
execution channel `make_execute(gt, eps=0)` — truth from the world, modeled here by the
ground-truth-backed executor; it is a *separate instance* from the out-of-loop referee, so the
engine still never reads the referee.

| confirmation channel | LLM-rate | accuracy | silent failures (a / b) | weather act/abstain |
|---|---|---|---|---|
| baseline (teacher-backed, tautological) | 11.4% | 91.7% | **109** (109 / 0) | 384 / 6 |
| + exec-grounded confirm *only* (lever #1) | 14.3% | 91.7% | 97 (97 / 0) | 384 / 6 |
| + exec-grounded **and gated induction** (#1+#2) | 18.9% | 91.7% | **0** (0 / 0) | 381 / 9 |

**Lever #1 alone is insufficient — and the reason is precise.** Execution-grounded
confirmation feeds the *per-regularity* posterior. A wrong fill (`weather in 1945`) gets
*folded* into a healthy regularity (rid 0: 9 legitimate weather examples, posterior 0.91), so a
handful of execution-failures on `1945` are averaged away by dozens of real weather
confirmations in the same posterior — `may_act` stays True and the silent failure persists. It
only suppressed the one error (`weather forecast for istanbul`) that had formed its *own*
regularity. Confirmation at regularity granularity cannot isolate a per-fill error.

**Levers #1+#2 together drive silent failures to 0.** Gating *induction* on the same
independent signal (`Engine(confirm_learning=True)`: a teacher label is folded into structure
only if executing it actually succeeds) keeps the wrong fill out of the regularity entirely.
The weather regularity's slot domain never admits `1945`; the input never strict-matches a
mature regularity; it stays a **permanent defer zone** that routes to the LLM. Legitimate
weather acting is preserved (381 vs 384 acts — no collateral coupling), and accuracy holds at
the teacher ceiling (91.7%). The 109 inherited errors did not vanish — they **moved from silent
autonomous failure into visible LLM-call abstention** (LLM-rate 11.4% → 18.9%, the honest price
of safety). The target is met: **0 silent failures added beyond the teacher's
independently-confirmed-correct rate** (b stayed 0 throughout; a went to 0).

Backward-compatible: `confirm_learning` defaults False, and with the oracle teacher both modes
are identical (every oracle label is correct, so it is confirmed and learned either way) — all
packaged results are unchanged.

## Cost–safety tradeoff vs boundary rate (`python -m oad.experiments_cost_safety_curve`)
The tradeoff is parameterized by the **boundary rate** (the surface-ambiguous / OOD share of
traffic). Sweeping junk fraction 0 → 60% (→ [`cost_safety_curve.json`](cost_safety_curve.json),
[`figs/real_teacher/cost_safety_curve.png`](figs/real_teacher/cost_safety_curve.png)):

| boundary rate | baseline silent | fixed silent | fixed LLM-rate | fixed over-cautious defer |
|---|---|---|---|---|
| 0% | 0.0% | 0.0% | 3.6% | 3.6% |
| 5% | 1.8% | **0.0%** | 8.9% | 3.6% |
| 10% | 3.7% | **0.0%** | 12.5% | 3.6% |
| 15% | 7.3% | **0.0%** | 18.9% | 3.6% |
| 25% | 12.2% | **0.0%** | 26.7% | 3.6% |
| 40% | 14.2% | **0.0%** | 42.3% | 3.6% |
| 60% | 20.2% | **0.0%** | 62.7% | 3.6% |

- **Safety:** the baseline's silent-failure rate climbs ~linearly with the boundary rate (to
  20% at 60% junk); the fix holds it at **0% across the whole sweep**. The price of safety is
  the deferral-rate gap, which widens as boundary traffic grows (at 60% the fix pays 62.7% vs
  the baseline's 52.5% calls — more surface-ambiguous input means a larger permanent defer zone).
- **Precision cost is ~0 here, and the table shows why.** The "over-cautious defer" column
  (legitimate inputs, `gt != fallback`, that the fix nonetheless declined) is **flat at the
  3.6% cold-start floor** — it does *not* rise with the boundary rate. That 3.6% is the
  unavoidable first-encounter learning abstention present even at 0% junk, not a penalty the
  fix introduces. So the fix sacrifices essentially no legitimate coverage; it converts
  boundary silent-failures into boundary deferrals without over-deferring real work. **Caveat:**
  this holds because the executor is clean (eps=0) and the teacher is right on every in-domain
  input — under a noisy executor or a teacher that errs on legitimate inputs, the over-cautious
  rate would rise, and that precision cost is exactly what the next sweep must quantify.

## Takeaway for the writeup — read these four framings before citing
1. **Accuracy is still teacher-bounded (91.7%); the fix does not raise it, and cannot.**
   "0 silent failures" is **not** "OAD is correct." The ceiling is the teacher's accuracy; the
   claim is "at the teacher's quality, ~9× cheaper, **with no silent-failure risk added** beyond
   the teacher's own confirmed-correct rate." Never let the 0 be read as an accuracy number.
2. **The fix moved the load-bearing caveat up one level — it did not remove it.** Confirmation
   is now teacher-independent, but the result now rests on the *execution signal* being genuinely
   independent and reliable. Here that signal is ground-truth-backed, i.e. structurally clean —
   so the fix is proven *in simulation*. The open question is real: in a live tool environment,
   does `get_weather(1945)` actually error, or does it return 200 OK + garbage? Only a real tool
   env settles that. The safety is exactly as good as the execution channel is honest.
3. **The cost–safety curve is the reviewer-facing artifact.** Safety vs boundary-rate and
   cost vs boundary-rate, baseline vs fix, on one axis (above). A reviewer will push on the
   boundary rate (more surface-ambiguous traffic → bigger defer zone → steeper cost) and on the
   abstention split (genuine boundary vs over-cautious). Both are now decomposed.
4. **WDR's role is sharpened, not erased.** The induction gate catches **lexically-separable**
   boundaries (`open the account` — "account" is not in the confirmed slot domain → defer).
   What remains irreducible is the **same-surface, same-domain-filler, different-meaning**
   boundary (real `open the door` vs a deceptive one) — surface methods cannot tell these apart;
   internal representation (WDR) is needed. So the fix *narrows* the silent-failure surface to
   exactly WDR's motivating case. The two-paper story (black-box OAD + white-box WDR) is
   **stronger** for it: OAD handles lexical-novelty boundaries, WDR handles surface-identical
   semantic ambiguity.

**Next experiment (queued):** rerun `experiments_cost_safety_curve` / `experiments_exec_confirm`
under a **noisy executor** (`make_execute(gt, eps>0)`) to quantify how much independent-signal
noise the 0-silent-failure guarantee tolerates, and how fast the over-cautious (precision) cost
rises — the real-world version of the R4 noise sweep, now on the *confirmation* channel.
