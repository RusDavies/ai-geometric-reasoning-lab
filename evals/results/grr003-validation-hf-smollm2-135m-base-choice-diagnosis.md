# GRR-003 SmolLM2-135M Same-Answer Diagnosis

Backlog item: GR-031
Status: diagnostic report

## Inputs

- Result file: `evals/results/grr003-validation-hf-smollm2-135m-base-choice.json`
- Dataset: `data/grr003/grr003-validation.jsonl`
- Dataset SHA-256: `4a22bcee662f7f95a4af0222acf670779ad9b974e7e373c7580e22cc94c46ad8`
- Model: `HuggingFaceTB/SmolLM2-135M`
- Provider: `huggingface_causal_lm`
- Answer mode: `choice`
- Prompt template: `dynamic: facts prompt or scene-program allowed-label prompt`

## Summary

- Pair count: `24`
- Both-correct pairs: `0`
- Flip failures: `24` same-answer pairs out of `24`
- Same predicted label across canonical and perturbed side: `24` / `24` (100.0%)
- Invalid-output pairs: `0`
- Canonical accuracy: `12` / `24`
- Perturbed accuracy: `7` / `24`

## Relation-Family Pattern

| Relation family | Pairs | Canonical correct | Perturbed correct | Same prediction | Predicted transitions | Gold transitions |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| between | 3 | 3 | 0 | 3 | `between -> between` (3) | `between -> not_between` (3) |
| containment | 3 | 0 | 3 | 3 | `does_not_contain -> does_not_contain` (3) | `contains -> does_not_contain` (3) |
| distance_comparison | 3 | 1 | 0 | 3 | `same_distance -> same_distance` (3) | `closer_to_first -> closer_to_second` (1), `closer_to_second -> closer_to_first` (1), `same_distance -> closer_to_first` (1) |
| horizontal_order | 3 | 1 | 1 | 3 | `right_of -> right_of` (3) | `left_of -> right_of` (1), `right_of -> left_of` (1), `same_x -> left_of` (1) |
| inside | 3 | 1 | 1 | 3 | `inside -> inside` (3) | `inside -> outside` (1), `boundary -> inside` (1), `outside -> boundary` (1) |
| line_membership | 3 | 2 | 0 | 3 | `on_line -> on_line` (3) | `on_line -> off_line` (2), `on_segment -> off_segment` (1) |
| point_identity | 3 | 3 | 0 | 3 | `same_point -> same_point` (3) | `same_point -> different_point` (3) |
| vertical_order | 3 | 1 | 2 | 3 | `below -> below` (3) | `below -> above` (1), `above -> below` (1), `same_y -> below` (1) |

## Diagnosis

The dominant failure is not output-format compliance: all outputs were parseable labels.
It is also not random label noise. The model selected one stable answer per relation family
and reused it on both sides of every label-changing pair.

That pattern means the closed-label scorer is measuring a real pair-level weakness:
the model is sensitive enough to prefer a plausible family-default label, but not sensitive
enough to the load-bearing coordinate or boundary edits that should flip the label.

The family defaults are especially visible in the predicted transitions:
`same_point -> same_point`, `on_line -> on_line`, `does_not_contain -> does_not_contain`,
`same_distance -> same_distance`, `right_of -> right_of`, `inside -> inside`,
`below -> below`, and `between -> between`.

## Follow-Up Options

- GR-032 should add a diagnostic scoring/export pass that records per-label candidate scores or margins, so we can distinguish strong default bias from near-ties.
- A prompt variant should ask for the changed statement and relation before the final label, then compare choice-only scoring against generate-and-parse scoring.
- A larger or stronger baseline should be run on the same GRR-003 split before treating this as a dataset-level difficulty claim.
- The generator should preserve these small validation pairs, but future validation should add more examples per relation family before broad claims.

## Bounded Claim

This diagnosis supports only the bounded claim that SmolLM2-135M, under this closed-label
answer-choice setup, collapses each GRR-003 relation family to a same-answer prediction
on the current 24-pair validation split. It does not show that GRR-003 is unsolved by
larger models or better prompted systems.
