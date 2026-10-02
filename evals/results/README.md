# Eval Results

Tracked result files preserve auditable baseline outputs that support project
claims.

## Results

- `grr001-mini-smollm2-135m.json`: first local Ollama baseline run for GR-003.
  Updated in GR-004 with pair-level confidence intervals and required metric
  breakdowns. This is a harness smoke baseline, not a project performance
  claim.
- `grr001-validation-smollm2-135m.json`: GR-011 validation run for the first
  student/local baseline candidate.
- `grr001-validation-llama3-latest.json`: GR-011 validation run for the
  selected local teacher candidate.

GRR-002 invariance-control datasets can be evaluated with the same harnesses.
For GRR-002 records, summaries include `invariance_failure_rate` and
`by_invariance_family` breakdowns in addition to the existing pair-level fields.

- `grr003-validation-hf-smollm2-135m-base-choice.json`: first GRR-003
  closed-label answer-choice baseline for SmolLM2-135M.
- `grr003-validation-hf-smollm2-135m-base-choice-diagnosis.md`: GR-031
  same-answer failure diagnosis for that GRR-003 baseline.
