# Claim Ledger

Backlog item: GR-007
Status: template

This ledger separates measured results, failed hypotheses, and speculation.
Nothing in this project should be described as demonstrated unless it appears
under measured results with enough evidence to rerun or audit the claim.

## Entry Rules

Every measured-result entry must include:

- stable claim ID;
- date;
- status;
- claim text;
- model identity and version;
- dataset or split identity;
- prompt, training, or decoding settings;
- scoring code or command;
- result file or raw-output reference;
- uncertainty or confidence interval;
- contamination, leakage, and overfitting caveats;
- reviewer or recorder.

Use these status values:

- `measured`: supported by tracked evidence;
- `failed`: tested and not supported;
- `speculative`: plausible but not measured;
- `retracted`: previously recorded but superseded or invalidated.

## Measured Results

### CLAIM-0009: SmolLM2-135M Fails GRR-003 Closed-Label Validation Pairs

Status: measured
Date: 2026-10-02
Recorder: Project maintainers

Claim:

Under closed-label answer-choice scoring, base SmolLM2-135M does not solve any
complete GRR-003 validation pair, even though it produces parseable labels for
every item.

Evidence:

- Result file:
  `../evals/results/grr003-validation-hf-smollm2-135m-base-choice.json`
- Dataset:
  `../data/grr003/grr003-validation.jsonl`
- Dataset SHA-256:
  `4a22bcee662f7f95a4af0222acf670779ad9b974e7e373c7580e22cc94c46ad8`
- Scoring code: `../src/eval_hf_causal.py`
- Rail spec: `../evals/rails/GRR-003-spatial-program.md`

Model/settings:

- model: `HuggingFaceTB/SmolLM2-135M`
- provider: Hugging Face causal LM
- answer mode: `choice`
- prompt template: dynamic scene-program prompt with per-record allowed labels
- decoding: no sampling; `seed=1`; `max_length=512`; `max_new_tokens=8`
- dependencies: `torch==2.14.0+cu130`, `transformers==5.16.1`
- held-out test use: none; validation split only

Observed result:

- validation pairs: `24`
- canonical accuracy: `12/24`
- perturbed accuracy: `7/24`
- both-correct pairs: `0/24`
- flip failures: `24/24`
- invalid-output pairs: `0/24`
- same-answer-when-gold-flips: `24/24`
- relation-family breakdowns are recorded in the result file.

Uncertainty:

- The validation split is small and synthetic.
- This is answer-choice scoring, not free-form instruction-following evidence.
- The result supports only a bounded negative baseline for this model, split,
  prompt contract, and scoring implementation.

Caveats:

- The result does not show that GRR-003 is intrinsically difficult for larger
  or geometry-specialized models.
- Closed-label scoring removes output-format failure, but the model still
  chooses the same label across every label-changing pair.

### CLAIM-0008: GRR-002 Generator Emits Balanced Invariance-Control Splits

Status: measured
Date: 2026-09-30
Recorder: Project maintainers

Claim:

The GR-020 generator can build deterministic GRR-002 canonical/invariant pairs,
balance the five invariance families by split, and validate split leakage before
writing datasets and checksummed manifests.

Evidence:

- Generator code: `../src/generate_grr002.py`
- Dataset manifest: `../data/grr002/manifest.json`
- Train split: `../data/grr002/grr002-train.jsonl`
- Validation split: `../data/grr002/grr002-validation.jsonl`
- Held-out test split: `../data/grr002/grr002-test.jsonl`
- Generator tests: `../tests/test_generate_grr002.py`
- Shared scoring code: `../src/eval_harness.py`
- HF scoring adapter: `../src/eval_hf_causal.py`

Inputs:

- source grammar: GRR-001 symbolic relational generator
- rail spec: `../evals/rails/GRR-002-invariance-control.md`
- held-out test use: none beyond generation and validation

Observed result:

- train records: `70`
- validation records: `15`
- test records: `15`
- train family counts: `14` each for `fact_order_shuffle`, `paraphrase`,
  `entity_renaming`, `irrelevant_distractor_edit`, and
  `non_load_bearing_polarity`
- validation family counts: `3` each
- test family counts: `3` each

Verification:

- `python3 -m unittest tests.test_generate_grr002 -q`
- `python3 -m unittest discover -s tests -q`
- `python3 src/generate_grr002.py`
- `git diff --check`

Uncertainty:

- This is a generator/scoring-infrastructure claim, not a model-quality claim.
- The current GRR-002 implementation uses one invariance family per pair rather
  than multi-family high-difficulty combinations.

Caveats:

- The rail still uses symbolic relational facts, not spatial geometry.
- The shared metric schema preserves legacy `perturbed_*` fields for GRR-001
  compatibility; GRR-002 readers should prefer `comparison_*`,
  `invariant_*`, and `invariance_failure_rate`.

### CLAIM-0007: GRR-001 Pair-Contrastive Exporter Produces Eligible Train Pairs And Diagnostics

Status: measured
Date: 2026-09-09
Recorder: Project maintainers

Claim:

The GR-017 exporter can build pair-shaped GRR-001 training records from accepted
distillation examples while excluding non-train records and preserving
structured diagnostics for incomplete or ineligible pairs.

Evidence:

- Exporter note: `../knowledge/pair-contrastive-exporter.md`
- Pair output:
  `../data/grr001/pair-contrastive/grr001-train-pairs.jsonl`
- Diagnostic output:
  `../data/grr001/pair-contrastive/grr001-train-pair-diagnostics.jsonl`
- Manifest:
  `../data/grr001/pair-contrastive/grr001-train-pair-manifest.json`
- Exporter code: `../src/export_pair_contrastive.py`
- Tests: `../tests/test_export_pair_contrastive.py`

Inputs:

- source train split: `../data/grr001/grr001-train.jsonl`
- source validation split: `../data/grr001/grr001-validation.jsonl`
- accepted distillation examples:
  `../data/grr001/distillation/grr001-llama3-latest-train-validation-accepted.jsonl`
- held-out test use: none

Observed result:

- source records: `85`
- accepted examples: `165`
- eligible pair records: `67`
- diagnostic records: `18`
- diagnostic `incomplete_pair` count: `5`
- diagnostic `non_train_split_excluded` count: `15`

Verification:

- `python3 -m unittest tests.test_export_pair_contrastive -q`
- `python3 -m unittest discover -s tests -q`
- `python3 src/export_pair_contrastive.py`
- `git diff --check`

Uncertainty:

- This is a data-export/process claim, not a model-quality claim.
- No pair-contrastive training run has used these records yet.

Caveats:

- The validation exclusions are expected because the accepted distillation
  bundle includes validation examples, while the exporter emits only train
  pairs for training.
- GRR-002 invariance-control datasets now exist separately under
  `../data/grr002/`; this exporter remains a GRR-001 pair-contrastive exporter.


### CLAIM-0006: HF Answer-Choice Evaluation Removes Invalid Outputs but Preserves Flip Failure

Status: measured
Date: 2026-09-05
Recorder: Project maintainers

Claim:

For binary GRR-001 Hugging Face evaluation, answer-choice scoring removes the
direct-generation invalid-output failure while showing that both tested
SmolLM2 variants still fail every validation flip pair.

Evidence:

- Evaluation note: `../knowledge/hf-answer-choice-eval.md`
- Base choice result:
  `../evals/results/grr001-validation-hf-smollm2-135m-base-choice.json`
- Trained smoke choice result:
  `../evals/results/grr001-validation-hf-smollm2-135m-distilled-smoke-choice.json`
- Evaluation code: `../src/eval_hf_causal.py`

Model/settings:

- base model: `HuggingFaceTB/SmolLM2-135M`
- trained checkpoint:
  `../artifacts/gr-014-smollm2-135m-distilled-smoke`
- dataset: `../data/grr001/grr001-validation.jsonl`
- provider: Hugging Face causal LM
- answer mode: `choice`
- candidate answers: `True`, `False`
- scoring: summed candidate continuation log probability
- held-out test use: none

Observed result:

- base invalid-output pairs: `0/15`
- trained smoke invalid-output pairs: `0/15`
- base both-correct rate: `0/15`
- trained smoke both-correct rate: `0/15`
- base flip-failure rate: `15/15`
- trained smoke flip-failure rate: `15/15`

Uncertainty:

- The validation split has only `15` pairs.
- This is a binary answer-choice measurement mode, not evidence about free-form
  instruction following.

Caveats:

- Answer-choice scoring fixes the measurement interface, not the model.
- Both tested variants still choose the same answer across every validation
  flip pair.
- Longer answer-only training is not justified until pair-level contrast is
  addressed.

### CLAIM-0004: Llama 3 Teacher Outputs Produced First GRR-001 Distillation Dataset

Status: measured
Date: 2026-09-05
Recorder: Project maintainers

Claim:

The GR-012 builder can run local `llama3:latest` over the deterministic GRR-001
train and validation splits, preserve raw teacher outputs, accept only
parseable gold-matching answer-only examples, and preserve rejected diagnostics.

Evidence:

- Builder note: `../knowledge/distillation-data-builder.md`
- Manifest:
  `../data/grr001/distillation/grr001-llama3-latest-train-validation-manifest.json`
- Raw teacher outputs:
  `../data/grr001/distillation/grr001-llama3-latest-train-validation-teacher-outputs.jsonl`
- Accepted examples:
  `../data/grr001/distillation/grr001-llama3-latest-train-validation-accepted.jsonl`
- Rejected diagnostics:
  `../data/grr001/distillation/grr001-llama3-latest-train-validation-rejected.jsonl`
- Builder code: `../src/build_distillation_data.py`

Model/settings:

- provider: Ollama
- model: `llama3:latest`
- temperature: `0.0`
- seed: `1`
- response cap: `8` generated tokens
- source splits: GRR-001 train and validation
- held-out test use: none

Observed result:

- source records: `85`
- teacher-output records: `170`
- accepted examples: `165`
- rejected diagnostics: `5`

Uncertainty:

- This is a data-generation/process claim, not a student-quality claim.
- The accepted examples have not yet been used to train or evaluate a student.

Caveats:

- Teacher outputs are accepted only when they match deterministic gold answers,
  so the accepted file is not evidence that `llama3:latest` is always correct.
- Rejected diagnostics include both answer disagreements and one invalid output.
- Validation examples are included for future checkpoint selection; the held-out
  test split remains unused.

### CLAIM-0003: Local Llama 3 Is a Stronger GRR-001 Teacher Candidate Than SmolLM2

Status: measured
Date: 2026-09-04
Recorder: Project maintainers

Claim:

On the 15-pair GRR-001 validation split, local `llama3:latest` substantially
outperforms local `smollm2:135m` on both-correct rate and flip-failure rate
under the same harness settings.

Evidence:

- Survey note: `../knowledge/teacher-candidate-survey.md`
- Llama 3 result file: `../evals/results/grr001-validation-llama3-latest.json`
- SmolLM2 result file: `../evals/results/grr001-validation-smollm2-135m.json`
- Dataset: `../data/grr001/grr001-validation.jsonl`
- Scoring code: `../src/eval_harness.py`

Model/settings:

- provider: Ollama
- teacher candidate: `llama3:latest`
- student/baseline candidate: `smollm2:135m`
- temperature: `0.0`
- seed: `1`
- response cap: `8` generated tokens
- bootstrap samples: `1000`
- bootstrap seed: `1001`

Metrics:

- `llama3:latest` both-correct rate: `13/15`
- `llama3:latest` flip-failure rate: `2/15`
- `llama3:latest` invalid-output rate: `0/15`
- `smollm2:135m` both-correct rate: `0/15`
- `smollm2:135m` flip-failure rate: `15/15`
- `smollm2:135m` invalid-output rate: `0/15`

Uncertainty:

- Bootstrap confidence intervals are recorded in the result files.
- The validation split has only 15 pairs, so this is teacher-candidate evidence,
  not a final benchmark.

Caveats:

- `llama3:latest` is selected only as the current local teacher candidate for
  distillation-data-builder implementation.
- No generated distillation data has been accepted yet.
- No student improvement has been measured from this claim.
- External/API teachers remain untested and require explicit approval.

### CLAIM-0002: Actual SmolLM2 Training Smoke Runs on CPU

Status: measured
Date: 2026-09-04
Recorder: Project maintainers

Claim:

The isolated Python 3.11 environment can load `HuggingFaceTB/SmolLM2-135M`
through Hugging Face tooling and execute one CPU training step against the
generated GRR-001 training export.

Evidence:

- Result file: `../training-results/smollm2-135m-actual-smoke.json`
- Environment note: `../knowledge/training-environment.md`
- Dataset: `../data/grr001/grr001-train-prompts.jsonl`
- Training code: `../src/train_smollm2_smoke.py`

Model/settings:

- model: `HuggingFaceTB/SmolLM2-135M`
- Python: `3.11.16`
- `torch`: `2.14.0+cu130`
- `transformers`: `5.16.1`
- device: `cpu`
- examples loaded: `2`
- batch size: `1`
- max steps: `1`
- seed: `1`
- checkpoint saved: `false`

Observed result:

- status: `trained`
- steps completed: `1`
- loss: `3.275089979171753`
- elapsed training seconds: `143.925`

Uncertainty:

- This is a process smoke test, not a model-quality measurement.
- No base-versus-trained validation comparison was run.

Caveats:

- No reusable trained checkpoint was saved.
- The result does not demonstrate improved geometric reasoning.
- Full-parameter CPU training is slow enough that future training experiments
  should prefer adapter tuning or deliberately tiny step counts on this host.

### CLAIM-0001: Mini-Smoke Baseline Shows Flip Failures

Status: measured
Date: 2026-09-04
Recorder: Project maintainers

Claim:

The local `smollm2:135m` Ollama baseline can be evaluated end to end by the
GRR-001 mini harness, and on the 3-pair mini fixture it gives the same parsed
answer for both sides of every pair.

Evidence:

- Result file: `../evals/results/grr001-mini-smollm2-135m.json`
- Dataset: `../data/grr001-mini.jsonl`
- Scoring code: `../src/eval_harness.py`
- Rail spec: `../evals/rails/GRR-001-relational-flip.md`

Model/settings:

- provider: Ollama
- model: `smollm2:135m`
- temperature: `0.0`
- seed: `1`
- response cap: `8` generated tokens

Metrics:

- pair count: 3
- canonical accuracy: 1/3
- perturbed accuracy: 2/3
- both-correct rate: 0/3
- flip-failure rate: 3/3
- invalid-output rate: 0/3

Uncertainty:

- Bootstrap confidence intervals are recorded in the result file.
- The fixture is too small for meaningful performance comparison.

Caveats:

- This is a harness smoke result, not a model-quality benchmark.
- The mini fixture is not a held-out benchmark split.
- No training or distillation improvement is measured here.

## Failed Hypotheses

Use this section when a tested idea does not hold up. Failed entries are useful
evidence, not clutter.

### CLAIM-0005: One-Step Distilled SmolLM2 Smoke Does Not Improve Pair-Level Validation

Status: failed
Date: 2026-09-05
Recorder: Project maintainers

Hypothesis:

A one-step CPU smoke checkpoint trained on accepted `llama3:latest` GRR-001
distillation examples would show measurable pair-level validation improvement
over the base Hugging Face `SmolLM2-135M` model.

Evidence:

- Comparison note: `../knowledge/distilled-smollm2-smoke-comparison.md`
- Training summary:
  `../training-results/gr-014-smollm2-135m-distilled-smoke.json`
- Base result:
  `../evals/results/grr001-validation-hf-smollm2-135m-base.json`
- Trained smoke result:
  `../evals/results/grr001-validation-hf-smollm2-135m-distilled-smoke.json`
- Training code: `../src/train_smollm2_smoke.py`
- HF eval code: `../src/eval_hf_causal.py`

Dataset/split:

- training data: accepted GRR-001 distillation examples, `split=train`, first
  `16` examples loaded
- validation data: `../data/grr001/grr001-validation.jsonl`
- held-out test use: none

Model/settings:

- base model: `HuggingFaceTB/SmolLM2-135M`
- trained checkpoint:
  `../artifacts/gr-014-smollm2-135m-distilled-smoke`
- max training steps: `1`
- batch size: `1`
- learning rate: `0.0001`
- device: CPU
- generation: greedy, `8` max new tokens

Observed result:

- base both-correct rate: `0/15`
- trained smoke both-correct rate: `0/15`
- base invalid-output pairs: `15/15`
- trained smoke invalid-output pairs: `15/15`
- trained smoke canonical accuracy improved from `0/15` to `1/15`, but this
  did not produce any both-correct pair and did not fix invalid outputs.

Why it failed:

Direct Hugging Face generation did not reliably produce exactly parseable
`True`/`False` answers. Output-format failure dominated the validation result.

Follow-up:

Add a prompt/parser/generation-format compliance pass before spending CPU time
on longer distillation training runs.

### Template

Status: failed
Date: YYYY-MM-DD
Recorder:

Hypothesis:

Evidence:

- Result file:
- Dataset/split:
- Model/settings:
- Scoring command:

Observed result:

Why it failed:

Follow-up:

## Speculation

Use this section only for ideas that have not yet been measured. Speculation
must not be cited as project evidence.

### Template

Status: speculative
Date: YYYY-MM-DD
Recorder:

Idea:

Why it might matter:

What would make it measurable:

Required evidence:

## Retracted Claims

Use this section when an earlier entry was superseded, invalidated, or found to
depend on a broken measurement.

### Template

Status: retracted
Date: YYYY-MM-DD
Recorder:

Original claim ID:

Reason for retraction:

Replacement entry, if any:
