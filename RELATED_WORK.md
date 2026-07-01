# Related Work & Positioning (expanded) — ROADMAP Phase E2

Paper-ready prose to replace/expand the current thin related work (§4.3 + the 6
references). Each paragraph situates OAD in an established literature and states the
difference, so a reviewer sees command of the field rather than a narrow framing.
Citations below are verified; BibTeX at the end. Weave the **Phase-C dominance result**
(OAD lower-left of both caches on the cost-safety plane) into the first paragraph.

---

**Decision re-use: caches, cascades, and routing.** OAD's cost-reduction goal places
it among methods that avoid calling a large model for every decision. Semantic caches
serve a stored response on a sufficiently similar query [gptcache]; cascades and
routers send easy inputs to cheap models and hard ones to expensive models
[frugalgpt, hybridllm, routellm]. OAD shares this regime but differs in mechanism and
contract: it induces a *generalizing symbolic structure* online — templates with
variable slots — rather than keying a flat cache on surface strings, and it makes
abstention a first-class, auditable outcome rather than always returning a (possibly
stale) hit. This difference is measurable on the cost–safety axis: a verbatim semantic
cache serves a stale parameter across slot variants and mis-serves should-decline
inputs, whereas OAD's slot induction fills the correct argument on unseen fillers and
its gate abstains; on a BFCL-grounded stream OAD attains the safe regime at lower call
rate than an exact cache and at far lower silent-failure rate than a semantic cache at
matched cost. Routers, in turn, learn a continuous dispatch policy over models; OAD
instead *removes* the model from the loop on the inputs it has provably learned, and
defers the rest — a discrete, auditable act/abstain decision rather than a learned
soft route.

**Abstention as selective prediction (the reject option).** OAD's acting/maturity bar
is, in the language of selective prediction, a risk–coverage operating point: it trades
coverage (autonomous actions) for risk (silent failures), exactly the trade studied
since the classical reject option [chow1970] and formalized for modern models as
selective classification [geifman2017, geifman2019] and surveyed broadly as learning
with a reject option [hendrickx2024]. OAD's Beta–Binomial gate is a selective predictor
over induced templates. Two differences matter. First, OAD's abstention is *symbolic
and auditable*: every refusal carries a machine-readable reason
(`novel_content_word`, `slot_extraction_method`, `immature`, …), not a scalar
confidence below a threshold over an opaque network. Second, the predictor being
gated is *induced online* from a teacher rather than a pre-trained classifier. The
fixed acting threshold is the natural place to import the calibration machinery of
this literature (next paragraph).

**Distribution-free calibration of the bar.** The acting bar is currently a fixed
`P(p > p_act) ≥ c_act`. Conformal prediction [vovk2005, angelopoulos2021] offers a
distribution-free, finite-sample route to set it to a *target* silent-failure rate
with a coverage guarantee on held data, replacing the hand-tuned threshold with a
calibrated one — turning "we picked 0.85" into "we guarantee silent-failure ≤ α with
probability 1−δ." This is the principled-abstention direction the paper flags as
future work, now with a concrete tool.

**Learning to defer to an expert.** OAD's abstain-and-call-the-teacher is precisely a
defer-to-expert decision, the subject of the learning-to-defer literature
[madras2018, mozannar2020], which learns a joint predictor–rejector that either acts
or routes to a downstream expert. OAD differs in two consequential ways. (i) The
"expert" is the *same frozen LLM that taught the policy*, so naïvely confirming an
action by re-asking the teacher is a tautology that seals the teacher's errors into
autonomous behavior — which is why OAD binds confirmation to a teacher-independent
execution signal. Our experiments show this is load-bearing and only partial: a
ground-truth-free execution signal removes ~75% of the inherited silent failures
(the rest are schema-valid-but-wrong calls no execution check can catch), where the
oracle-backed signal removed 100% — a distinction the learning-to-defer framing,
which usually assumes an independent expert, does not surface. (ii) OAD's rejector is
a symbolic gate over induced structure, not a learned rejector trained jointly with
the predictor.

**Online template induction is inductive program synthesis.** The engine's core —
born-literal regularities, conservative generalization of a constant to a slot only
when it provably preserves action-correctness, and splitting at a boundary rather than
over-generalizing — is anti-unification / least-general generalization [plotkin1970]
operating as *online version-space narrowing*. This connects OAD to a deep line in
programming-by-example and version-space learning: version spaces as search
[mitchell1982], the version-space *algebra* that composes simple spaces into complex
ones [lau2003], and practical example-driven synthesis such as FlashFill
[gulwani2011]; recent work even revisits programming-by-example with LLMs
[li2024]. OAD's contribution against this backdrop is the *setting*, not a new
synthesis algorithm: it runs single-pass and online over a Zipfian stream of
(input → action) pairs supplied by a frozen LLM, gated by a maturity/acting criterion
so that a partially-induced program never acts prematurely, and targeting tool-call
templates rather than string transformers. Framing the induction this way both
strengthens the novelty claim and inherits the vocabulary (generalization, boundary,
consistency) of a mature field.

**Not network distillation.** Finally, OAD is not knowledge distillation
[hinton2015]: it does not compress the teacher into a smaller network, which would
inherit the teacher's opacity and a training cost. It derives a transparent symbolic
policy with no gradients, no parameters, and no GPU at serving time. "Distillation"
here is online and into symbols, and the resulting policy's decisions are inspectable
by construction.

---

## Suggested placement

- Fold paragraphs 1–4 into an expanded **Related Work** (promote from the current
  inline §4.3 "Positioning"), and paragraph 5 into the **slot-induction** discussion
  (or Related Work) — it is the strongest novelty anchor for the induction.
- Paragraph 3 (calibration) and the conformal cite double as the lead-in to the
  Future Work item on a principled abstention criterion.
- The cost–safety sentence in paragraph 1 should cite the new Pareto figure.

## BibTeX (verified)

```bibtex
@inproceedings{gptcache,
  author = {Bang, Fu},
  title = {{GPTCache}: An Open-Source Semantic Cache for {LLM} Applications Enabling Faster Answers and Cost Savings},
  booktitle = {Proc. 3rd Workshop for Natural Language Processing Open Source Software (NLP-OSS)},
  pages = {212--218}, publisher = {ACL}, year = {2023}}

@article{frugalgpt,
  author = {Chen, Lingjiao and Zaharia, Matei and Zou, James},
  title = {{FrugalGPT}: How to Use Large Language Models While Reducing Cost and Improving Performance},
  journal = {Transactions on Machine Learning Research (TMLR)}, year = {2024}, note = {arXiv:2305.05176}}

@inproceedings{hybridllm,
  author = {Ding, Dujian and Mallick, Ankur and Wang, Chi and Sim, Robert and Mukherjee, Subhabrata and R{\"u}hle, Victor and Lakshmanan, Laks V. S. and Awadallah, Ahmed Hassan},
  title = {Hybrid {LLM}: Cost-Efficient and Quality-Aware Query Routing},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2024}}

@article{routellm,
  author = {Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M. Waleed and Stoica, Ion},
  title = {{RouteLLM}: Learning to Route {LLMs} with Preference Data},
  journal = {arXiv preprint arXiv:2406.18665}, year = {2024}}

@article{chow1970,
  author = {Chow, C. K.},
  title = {On Optimum Recognition Error and Reject Tradeoff},
  journal = {IEEE Transactions on Information Theory}, volume = {16}, number = {1}, pages = {41--46}, year = {1970}}

@inproceedings{geifman2017,
  author = {Geifman, Yonatan and El-Yaniv, Ran},
  title = {Selective Classification for Deep Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {30}, year = {2017}}

@inproceedings{geifman2019,
  author = {Geifman, Yonatan and El-Yaniv, Ran},
  title = {{SelectiveNet}: A Deep Neural Network with an Integrated Reject Option},
  booktitle = {Proc. 36th International Conference on Machine Learning (ICML)},
  series = {PMLR}, volume = {97}, pages = {2151--2159}, year = {2019}}

@article{hendrickx2024,
  author = {Hendrickx, Kilian and Perini, Lorenzo and Van der Plas, Dries and Meert, Wannes and Davis, Jesse},
  title = {Machine Learning with a Reject Option: A Survey},
  journal = {Machine Learning}, year = {2024}}

@inproceedings{madras2018,
  author = {Madras, David and Pitassi, Toniann and Zemel, Richard},
  title = {Predict Responsibly: Improving Fairness and Accuracy by Learning to Defer},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2018}}

@inproceedings{mozannar2020,
  author = {Mozannar, Hussein and Sontag, David},
  title = {Consistent Estimators for Learning to Defer to an Expert},
  booktitle = {Proc. 37th International Conference on Machine Learning (ICML)},
  series = {PMLR}, volume = {119}, pages = {7076--7087}, year = {2020}}

@book{vovk2005,
  author = {Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn},
  title = {Algorithmic Learning in a Random World}, publisher = {Springer}, year = {2005}}

@article{angelopoulos2021,
  author = {Angelopoulos, Anastasios N. and Bates, Stephen},
  title = {A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  journal = {arXiv preprint arXiv:2107.07511}, year = {2021}}

@article{mitchell1982,
  author = {Mitchell, Tom M.},
  title = {Generalization as Search},
  journal = {Artificial Intelligence}, volume = {18}, number = {2}, pages = {203--226}, year = {1982}}

@article{lau2003,
  author = {Lau, Tessa and Wolfman, Steven A. and Domingos, Pedro and Weld, Daniel S.},
  title = {Programming by Demonstration Using Version Space Algebra},
  journal = {Machine Learning}, volume = {53}, number = {1--2}, pages = {111--156}, year = {2003}}

@inproceedings{gulwani2011,
  author = {Gulwani, Sumit},
  title = {Automating String Processing in Spreadsheets Using Input-Output Examples},
  booktitle = {Proc. 38th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL)},
  pages = {317--330}, year = {2011}}

@inproceedings{li2024,
  author = {Li, Wen-Ding and Ellis, Kevin},
  title = {Is Programming by Example Solved by {LLMs}?},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, volume = {37}, year = {2024}}
```

(Already in the paper, reuse existing keys: Plotkin 1970 anti-unification; Hinton et
al. 2015 distillation; ReAct; Toolformer.)
