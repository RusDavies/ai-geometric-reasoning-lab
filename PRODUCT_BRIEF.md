# Product Brief

## Mission

Create, optimize, and test a geometric reasoning model.

This project treats "geometric reasoning" as an empirical machine-learning claim: a model should learn and preserve the structural relationships needed to solve reasoning tasks under controlled perturbation, rather than merely matching familiar surface patterns.

## Ultimate Goal

The ultimate goal is to distill an existing large LLM, such as Gemma 4 - 26B A4B, down to a much smaller geometric reasoning model or set of cooperating models.

## Product Thesis

Reasoning quality should be measurable by how well a model tracks load-bearing structural changes through a problem. If a model answers the original item correctly but fails when a minimal fact changes, the model has not demonstrated robust reasoning on that pair.

The first product should therefore be a reproducible research system:

- a perturbation-based reasoning evaluation harness;
- baseline measurements for existing small and accessible models;
- one or more trained or fine-tuned prototype models;
- optimization loops that improve paired reasoning performance;
- reports that separate measured results from speculation.

## Audience

- Project collaborators evaluating whether the idea is real.
- ML engineers who need reproducible experiments, not vibes in a lab coat.
- Researchers interested in small-model reasoning, benchmark robustness, and representation-level diagnostics.

## Scope

In scope:

- Synthetic and semi-synthetic reasoning tasks with controlled perturbations.
- Pair-level scoring, where canonical and perturbed items are evaluated together.
- Baseline runs against open or locally accessible models.
- Small-model training or fine-tuning experiments.
- Representation and activation analysis where it helps explain improvements.
- Clear result reporting with negative findings preserved.

Out of scope for now:

- Claims about consciousness, spirituality, quantum foundations, or cosmic order.
- Claims that data centers are obsolete.
- Claims of general AGI or frontier-model replacement.
- Irreproducible demos that cannot be rerun from tracked code and data recipes.

## Success Criteria

The project is useful when it can:

- generate or load paired reasoning tasks with explicit load-bearing perturbations;
- run the same eval against multiple baseline models;
- report pair-level accuracy, flip failure rate, and confidence intervals;
- train or tune a small model and compare it against baselines on held-out perturbation rails;
- preserve exact prompts, model settings, dataset versions, and scoring code;
- publish a claim ledger that says what was demonstrated, what failed, and what remains unknown.

## First Useful Outcome

A local evaluation harness that can run a small paired reasoning suite against at least one accessible baseline model and produce a reproducible report.

## Evidence Standard

No performance claim counts unless the repo contains enough information to rerun it:

- model identity and version;
- prompts and decoding settings;
- dataset generation or source provenance;
- scoring code;
- raw outputs or auditable summaries;
- confidence intervals or equivalent uncertainty notes;
- contamination and overfitting caveats.

## Open Questions

- Which model family should be used for the first baseline and prototype?
- Should the first task rail be purely symbolic logic, spatial/geometric reasoning, or a mix?
- Should "geometric" initially mean latent-space/representation structure, geometric problem content, or both?
- What local compute budget is acceptable for the first training run?
