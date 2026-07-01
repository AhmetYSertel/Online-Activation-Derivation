# BFCL transfer — findings & run handoff (ROADMAP Phase B)

Replaces the author-written 6-family workload (the F-1 / LIMITATIONS #3 critique) with
a **BFCL-grounded streaming workload**: real BFCL `simple` tools/schemas/phrasings +
real BFCL `irrelevance` items as the abstain set. The repetition OAD needs is the only
modeled element (see the header of `oad/bfcl.py` for the full rationale and why vanilla
BFCL — 400 items, 370 unique functions — is the wrong *shape* for OAD).

## What is built and validated (offline, no key)

- `oad/bfcl.py` — loader, 30 single-slot families derived from real BFCL functions,
  AST-match referee (`bfcl_score`: name + required-arg + value-in-accepted-set),
  Zipfian stream + real irrelevance probes, an `OracleBFCLTeacher` stand-in, and the
  real `OpenAIBFCLTeacher`.
- `oad/experiments_bfcl.py` — transfer driver, multi-seed 95% CIs, disk-cached teacher.
- `data/bfcl/` — the real BFCL v3 `simple`, `possible_answer`, and `irrelevance` files.

**Pipeline validation (oracle stand-in, 8 seeds):** OAD forms ~37 regularities and on
the BFCL-grounded stream reaches **74.8 ± 1.0% autonomy, 100% accuracy, 0 silent
failures, and abstains on 100% of real BFCL irrelevance probes**. So the real-tool
templates induce and mature cleanly, OAD acts on recurring real-tool calls, and it
defers on a third-party should-not-call set. This is the mechanism-validity result,
now on real BFCL material instead of author-written families.

| metric (oracle teacher, 8 seeds) | value |
|---|---|
| always-call accuracy | 100.0% |
| OAD call rate | 25.2 ± 1.0% |
| OAD autonomy | 74.8 ± 1.0% |
| OAD accuracy | 100.0% |
| silent failures (a / b) | 0.0 (0 / 0) |
| irrelevance abstain | 100.0% |

## What you run (the real transfer — needs your OpenAI key)

The only missing column is a **fallible** teacher. One cached pass over the fixed
**297-input universe** is the entire API spend (filler sets are deterministic, so the
cache then covers every seed for free).

```
export OPENAI_API_KEY=sk-...
python -m oad.experiments_bfcl --teacher openai --model gpt-5.5 --seeds 8
```

This prints, with CIs: the real teacher's raw accuracy on the universe, OAD's call
rate / accuracy / autonomy, the silent-failure split **a (inherited) vs b
(engine-added)**, and the irrelevance abstain rate — i.e. the real-data analogue of
the paper's Table 2, replacing the thin 51-input / 8-error run. Output:
`results/bfcl_results_openai.json` + cache `results/_bfcl_cache_gpt-5.5.json`.

What to look for: (1) does cost reduction transfer (call rate well under 100% at the
teacher's accuracy)? (2) is **b ≈ 0** on real tools (the load-bearing claim)? (3) what
is the real teacher's irrelevance-decline rate, and does OAD inherit its
relevance/irrelevance errors rather than add new ones?

## Immediate follow-up once the cache exists

With `results/_bfcl_cache_gpt-5.5.json` in place, the **honest executor (Phase A2/D1)
re-runs on real teacher errors**: a BFCL `ToolEnvironment` (tool-known + arg-shape
plausible) gives the ground-truth-free confirmation signal, and we measure how much of
the real teacher's silent failures it removes — the real-data version of the
109 → 22 (~75%) result. That is the natural next deliverable after you produce the
cache.

## Notes / honest caveats

- Surface and values are lowercased to match OAD's tokenizer; documented in `bfcl.py`.
- The stream's repetition is modeled (real agents repeat; BFCL-simple as shipped does
  not encode that). Tools, schemas, phrasings, and the abstain set are all third-party.
- `simple` is single-arg/single-tool; multi-arg and parallel-call families (BFCL
  `multiple` / `parallel`) are a later extension that would also exercise multi-slot
  induction.
