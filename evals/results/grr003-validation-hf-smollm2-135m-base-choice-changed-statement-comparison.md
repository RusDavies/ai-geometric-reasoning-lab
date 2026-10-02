# GRR-003 Prompt-Intervention Margin Comparison

Backlog item: GR-033
Status: comparison report

## Inputs

- Baseline result: `evals/results/grr003-validation-hf-smollm2-135m-base-choice-margins.json`
- Candidate result: `evals/results/grr003-validation-hf-smollm2-135m-base-choice-changed-statement-margins.json`
- Baseline prompt mode: `direct`
- Candidate prompt mode: `changed_statement`
- Model: `HuggingFaceTB/SmolLM2-135M`
- Dataset: `data/grr003/grr003-validation.jsonl`

## Summary

- Compared items: `48`
- Prediction changes: `0`
- Correctness changes: `0`
- Baseline average margin: `1.184`
- Candidate average margin: `1.265`

| Relation family | Items | Prediction changes | Correctness changes | Baseline avg margin | Candidate avg margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| between | 6 | 0 | 0 | 0.474 | 0.618 |
| containment | 6 | 0 | 0 | 0.130 | 0.457 |
| distance_comparison | 6 | 0 | 0 | 3.205 | 2.383 |
| horizontal_order | 6 | 0 | 0 | 1.267 | 0.907 |
| inside | 6 | 0 | 0 | 1.271 | 1.542 |
| line_membership | 6 | 0 | 0 | 0.789 | 1.582 |
| point_identity | 6 | 0 | 0 | 2.209 | 2.279 |
| vertical_order | 6 | 0 | 0 | 0.129 | 0.353 |

## Interpretation

This comparison measures whether the prompt intervention changes the closed-label
choice surface, not whether it teaches the model geometry. Prediction changes show
that the prompt moved the selected label; margin changes show whether it made the
selection more or less decisive.

Verdict: this prompt intervention did not change any selected labels or item
correctness on the validation split. It mostly changed confidence margins.
