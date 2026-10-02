# GRR-003 SmolLM2-135M Choice-Margin Diagnosis

Backlog item: GR-032
Status: diagnostic report

## Inputs

- Result file: `evals/results/grr003-validation-hf-smollm2-135m-base-choice-margins.json`
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
- Choice-margin evidence: avg `1.184`, min `0.032`

## Relation-Family Pattern

| Relation family | Pairs | Canonical correct | Perturbed correct | Same prediction | Choice margins | Predicted transitions | Gold transitions |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| between | 3 | 3 | 0 | 3 | avg `0.474`, min `0.324` | `between -> between` (3) | `between -> not_between` (3) |
| containment | 3 | 0 | 3 | 3 | avg `0.130`, min `0.032` | `does_not_contain -> does_not_contain` (3) | `contains -> does_not_contain` (3) |
| distance_comparison | 3 | 1 | 0 | 3 | avg `3.205`, min `2.988` | `same_distance -> same_distance` (3) | `closer_to_first -> closer_to_second` (1), `closer_to_second -> closer_to_first` (1), `same_distance -> closer_to_first` (1) |
| horizontal_order | 3 | 1 | 1 | 3 | avg `1.267`, min `1.175` | `right_of -> right_of` (3) | `left_of -> right_of` (1), `right_of -> left_of` (1), `same_x -> left_of` (1) |
| inside | 3 | 1 | 1 | 3 | avg `1.271`, min `1.219` | `inside -> inside` (3) | `inside -> outside` (1), `boundary -> inside` (1), `outside -> boundary` (1) |
| line_membership | 3 | 2 | 0 | 3 | avg `0.789`, min `0.746` | `on_line -> on_line` (3) | `on_line -> off_line` (2), `on_segment -> off_segment` (1) |
| point_identity | 3 | 3 | 0 | 3 | avg `2.209`, min `2.177` | `same_point -> same_point` (3) | `same_point -> different_point` (3) |
| vertical_order | 3 | 1 | 2 | 3 | avg `0.129`, min `0.085` | `below -> below` (3) | `below -> above` (1), `above -> below` (1), `same_y -> below` (1) |

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

The margin evidence is mixed rather than one-note. Distance comparison and point
identity show strong default-label preferences, while containment and vertical order
are much closer to runner-up ties. That means the same-answer collapse is partly
strong family-default bias and partly fragile near-tie scoring, depending on the
relation family.

## Follow-Up Options

- GR-033 should run a GRR-003 prompt-intervention diagnostic that asks for the changed statement and relation before the final label, then compares choice margins against this baseline.
- A stronger baseline should be run on the same split before treating this as dataset-level difficulty rather than small-model brittleness.
- Future GRR-003 validation should add more examples per relation family so margin summaries are less hostage to three examples at a time.

## Bounded Claim

This diagnosis supports only the bounded claim that SmolLM2-135M, under this closed-label
answer-choice setup, collapses each GRR-003 relation family to a same-answer prediction
on the current 24-pair validation split. It does not show that GRR-003 is unsolved by
larger models or better prompted systems.
