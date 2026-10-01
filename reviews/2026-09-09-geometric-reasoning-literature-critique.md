# Geometric Reasoning Literature Critique

Date: 2026-09-09

## Short Verdict

The current approach is scientifically sane as a first robustness harness, but
it is not yet a geometric-reasoning program in the stronger senses used by the
literature.

GRR-001 measures whether a language model tracks symbolic relational facts
through a controlled answer-flipping perturbation. That is useful, especially
because the current measured SmolLM2 failure is clean: answer-choice scoring
shows `0/15` both-correct pairs and `15/15` flip failures for both the base and
one-step distilled smoke variants. The next pair-contrastive slice directly
targets that defect.

The critique is that the project currently has only one narrow symbolic rail,
no spatial/object rail, no formal verifier, no theorem/proof state, and no
representation-level measurement. So the work is best described as
perturbation-based relational reasoning until additional rails or diagnostics
earn the word "geometric".

## Papers And What They Imply

### 1. ARC and ConceptARC: Generalization must be concept-level, not item-level

Chollet's ARC paper frames intelligence as skill-acquisition efficiency under
limited priors and experience, warning that task skill can be bought with
training data and does not by itself prove generalization. ConceptARC extends
that criticism by organizing tasks around explicit concept groups and testing
whether systems generalize a concept across varied instantiations.

Implication for this project:

- GRR-001 is aligned with the controlled-generalization instinct, but its
  concept space is still tiny: implication chains, polarity flips, distractors,
  and nonsense tokens.
- We need concept families, not just difficulty levels. For example:
  inheritance, exclusion, transitivity, containment, ordering, symmetry,
  intersection, betweenness, parallel/perpendicular, and diagram/program
  spatial relations.
- Held-out vocabularies are good but insufficient. A model can still learn the
  generator template rather than the abstract operation.

### 2. RAVEN, Relation Networks, OCRA, and sparse object-centric reasoning:
relations need architectural or data-shaping bias

Relation Networks showed that plain neural models may struggle with relational
questions, while an explicit relation module can help. RAVEN added structured
representations to visual analogy tasks. OCRA combines object-centric
representations with relational abstraction to improve systematic visual
generalization. Sparse relational reasoning work adds a useful caution:
object-centric slots help only if the relevant objects are actually captured;
otherwise the abstraction can fail more badly than a less interpretable model.

Implication for this project:

- Pair-contrastive training is a good data/objective move, but it does not add
  a relational inductive bias to SmolLM2. It asks a general causal LM to infer
  that bias from a tiny synthetic slice.
- The next training slice should be treated as a mechanics test, not as likely
  to produce durable reasoning. That matches the current project posture.
- If the project wants small specialist models, it should consider explicit
  structured intermediate representations: parsed facts, proof paths, graph
  traces, or a small verifier/head over relation graphs.

### 3. Bongard problems and pragmatic concept induction: contrast pairs are
valid, but the pair selection matters

The Bongard literature treats examples as deliberately chosen communicative
evidence, not random samples. Depeweg et al. use visual feature extraction,
symbolic vocabulary, formal concept language, Bayesian inference, and pragmatic
reasoning to infer concepts from small contrastive sets.

Implication for this project:

- The canonical/perturbed pair idea is defensible. Contrast is exactly the
  right measurement unit for some reasoning claims.
- But one positive/negative pair per record is weak evidence of concept
  understanding. Stronger evaluation should use small contrast sets where
  several minimally varied examples isolate one rule while varying surface
  form.
- Diagnostics should identify which relation changed and whether the model's
  preference moved in the right direction, not only whether the final answer
  flipped.

### 4. RUPBench: perturbation robustness is broader than load-bearing flips

RUPBench evaluates LLM reasoning robustness under lexical, syntactic, and
semantic perturbations across multiple reasoning datasets. Its main lesson for
us is not that GRR-001 is wrong; it is that a single perturbation type can
overfit the evaluation story.

Implication for this project:

- GRR-001's load-bearing flips are valuable, but the harness should add
  non-load-bearing paraphrases and distractor perturbations.
- We need invariance tests as well as flip tests:
  - load-bearing fact changes should flip the answer;
  - irrelevant fact changes should not flip the answer;
  - paraphrases should not flip the answer;
  - fact-order shuffles should not flip the answer.
- That distinction is crucial for avoiding a dumb "always flip under
  perturbation" student.

### 5. GeoQA, Geometry3K/Inter-GPS, AlphaGeometry, and AlphaGeometry2:
serious geometry systems separate perception/parsing, search, and verification

GeoQA and Geometry3K emphasize multimodal geometry: text, diagrams, theorem
knowledge, and interpretable programs or formal language. Inter-GPS parses text
and diagrams into formal language, then uses symbolic reasoning. AlphaGeometry
and AlphaGeometry2 go further: they combine a language model for auxiliary
construction with symbolic deduction/search, trained heavily on synthetic
proofs and evaluated against formalized olympiad geometry problems.

Implication for this project:

- Current GRR-001 is not comparable to geometry theorem proving. It has no
  diagram, theorem language, auxiliary construction, proof search, or formal
  proof checker.
- The project should avoid claiming a route to AlphaGeometry-like results
  unless it adds a verifier/search component or explicitly stays in abstract
  relational reasoning.
- The strongest evidence from geometry AI says: let neural models propose,
  let symbolic systems check. Pure answer-only distillation is the weaker
  path if the task has crisp logical structure.

### 6. GeoGramBench and SimplifiedRPM: "geometric" can mean representation
geometry, spatial/program geometry, or visual relation geometry

GeoGramBench tests program-to-geometry reasoning: whether LLMs can infer
spatial/geometric relations from drawing code, with high-level abstraction
remaining difficult even for frontier models. SimplifiedRPM investigates
representation geometry for visual relational reasoning and proposes objectives
that explicitly shape signal-to-dimensionality trade-offs.

Implication for this project:

- The word "geometric" is overloaded. The repo currently uses it mostly as
  "structural relation tracking under perturbation."
- If we mean spatial geometry, add a spatial/program rail.
- If we mean representation geometry, add activation/logit diagnostics and
  geometry-aware objectives.
- If we mean formal Euclidean geometry, add theorem-language examples and a
  checker.

## Critique Of The Current Approach

### What is strong

- The project correctly rejects unsupported hype and records measured claims
  separately from speculation.
- Pair-level scoring is the right unit for this first hypothesis.
- Held-out splits, deterministic generation, raw outputs, and answer-choice
  scoring are good evidence hygiene.
- The planned pair-contrastive exporter targets the actual observed failure:
  same-answer behavior across answer-flip pairs.

### What is weak

- The current rail is symbolic logic with nonsense nouns, not spatial or
  formal geometry.
- One rail is too easy to generator-overfit.
- Answer-only distillation does not teach or expose proof structure.
- Pair-contrastive loss may teach answer anti-correlation rather than
  reasoning, unless paired with invariance controls.
- There is no verifier, proof trace, or external symbolic engine, despite the
  literature repeatedly showing that crisp geometry benefits from symbolic
  checking.
- The current validation set is only 15 pairs, useful for smoke tests but too
  small for claims about model capability.

## Recommended Adjustments

1. Keep GR-017, but make the exporter preserve pair diagnostics needed for
   both flip and invariance tests.
2. Add GRR-002 as an invariance/control rail:
   same answer under paraphrase, fact order shuffles, renamed entities,
   irrelevant distractor edits, and non-load-bearing polarity changes.
3. Add GRR-003 as a spatial/program rail inspired by GeoGramBench:
   textual or code-described points, lines, containment, distance comparisons,
   left/right/inside/between relations, with deterministic labels.
4. Add proof-path supervision before longer distillation:
   require the teacher or generator to emit the load-bearing path, then train
   or evaluate whether the student can select the changed premise.
5. Add a symbolic checker layer for rails with crisp logic:
   generated examples should include a machine-checkable proof or derivation,
   even if the student is still evaluated as a language model.
6. Increase validation/test sizes before any positive performance claim:
   15 pairs is fine for smoke, not for "small model learned geometric
   reasoning."
7. Add representation diagnostics only after behavior improves:
   track logit margins across canonical/perturbed/invariant sets first; then
   inspect hidden states if there is a real behavioral signal to explain.

## Bottom Line

The current approach is a good first experimental scaffold. It is not yet a
strong geometry system. The next move should not be a larger answer-only
distillation run; it should be a contrast-and-invariance data slice with richer
diagnostics, followed by a second rail that introduces actual spatial or formal
geometric structure.

That keeps the project honest: first prove robust relational behavior, then
earn the stronger geometric claims one rail at a time.

## Sources

- Francois Chollet, "On the Measure of Intelligence", arXiv:1911.01547.
- Melanie Mitchell et al., "The ConceptARC Benchmark: Evaluating Understanding
  and Generalization in the ARC Domain", arXiv:2305.07141.
- Chi Zhang et al., "RAVEN: A Dataset for Relational and Analogical Visual
  rEasoNing", arXiv:1903.02741.
- Adam Santoro et al., "A simple neural network module for relational
  reasoning", arXiv:1706.01427.
- Taylor Webb et al., "Systematic Visual Reasoning through Object-Centric
  Relational Abstraction", arXiv:2306.02500.
- Alexander Fabian Spies and Stefan Roth, "Sparse Relational Reasoning with
  Object-Centric Representations", arXiv:2207.07512.
- Stefan Depeweg et al., "Solving Bongard Problems with a Visual Language and
  Pragmatic Reasoning", arXiv:1804.04452.
- Yuqing Wang et al., "Benchmarking Reasoning Under Perturbations for
  Robustness Evaluation in Large Language Models", arXiv:2406.11020.
- Jiaqi Chen et al., "GeoQA: A Geometric Question Answering Benchmark Towards
  Multimodal Numerical Reasoning", arXiv:2105.14517.
- Pan Lu et al., "Interpretable Geometry Problem Solving with Formal Language
  and Symbolic Reasoning", arXiv:2105.04165.
- Trieu H. Trinh et al., "Solving olympiad geometry without human
  demonstrations", Nature 2024.
- Yuri Chervonyi et al., "Gold-medalist Performance in Solving Olympiad
  Geometry with AlphaGeometry2", arXiv:2502.03544.
- Lianlei Shan et al., "Benchmarking the Geometric Program Reasoning in Modern
  LLMs", arXiv:2505.17653.
- Jiaqi Shang et al., "Unraveling the geometry of visual relational reasoning",
  arXiv:2502.17382.
