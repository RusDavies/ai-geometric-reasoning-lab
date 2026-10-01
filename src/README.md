# Source

Implementation code belongs here.

## Eval Harness

Run the first GRR-001 mini baseline against a local Ollama model:

```bash
python3 src/eval_harness.py --model smollm2:135m --dataset data/grr001-mini.jsonl --output runs/grr001-mini-ollama.json
```

The harness writes an auditable JSON result file containing model identity,
Ollama URL, generation options, prompt template, raw outputs, parsed answers,
per-pair results, and summary metrics. Default generation uses temperature
`0.0`, seed `1`, and an eight-token response cap.

Summary output includes:

- canonical accuracy;
- perturbed accuracy;
- both-correct rate;
- flip-failure rate;
- invalid-output rate;
- bootstrap confidence intervals over pairs;
- breakdowns by difficulty, hop count, distractor count, and negation
  involvement.

Run the same pair-level metric contract against a Hugging Face causal language
model or saved local checkpoint:

```bash
python3 src/eval_hf_causal.py --dataset data/grr001/grr001-validation.jsonl --model-id HuggingFaceTB/SmolLM2-135M
```

Use `--answer-mode choice` for binary GRR-001 runs when the goal is to score the
model's preference between the allowed `True` and `False` answers instead of
free-form generation compliance.

GRR-003 records use scene programs and per-record closed labels. The harnesses
build an allowed-label prompt from each item and parse against that label set:

```bash
python3 src/eval_hf_causal.py --dataset data/grr003/grr003-validation.jsonl --answer-mode choice
```

## GRR-001 Generator

Generate deterministic GRR-001 train, validation, and held-out test splits:

```bash
python3 src/generate_grr001.py --output-dir data/grr001 --train-count 70 --validation-count 15 --test-count 15
```

The generator writes JSONL split files plus a manifest containing split seeds,
record counts, and SHA-256 checksums. It validates answer flips, split
vocabulary separation, and duplicate fact-set leakage before writing output.

## GRR-002 Generator

Generate deterministic GRR-002 invariance-control splits:

```bash
python3 src/generate_grr002.py --output-dir data/grr002 --train-count 70 --validation-count 15 --test-count 15
```

The generator emits canonical/invariant pairs using the GRR-001 relational
grammar, balances invariance families by split, records `answer_invariant` as
the expected pair relation, and validates split vocabulary separation before
writing JSONL files and a manifest.

## GRR-003 Generator

Generate deterministic GRR-003 spatial/program geometry splits:

```bash
python3 src/generate_grr003.py --output-dir data/grr003
```

The generator emits text scene programs with exact deterministic labels for
point identity, line and segment membership, containment, distance comparison,
horizontal and vertical order, inside/boundary/outside, and between relations.
It records canonical/perturbed label-changing pairs, checker traces,
relation-family counts, answer-label counts, split seeds, and checksums.

## Training Export

Export GRR-001 records into prompt/target examples:

```bash
python3 src/export_grr001_training.py --input data/grr001/grr001-train.jsonl --output data/grr001/grr001-train-prompts.jsonl
```

Each pair becomes two examples: one canonical side and one perturbed side.

## Pair-Contrastive Export

Export accepted GRR-001 distillation examples into pair-shaped training
records:

```bash
python3 src/export_pair_contrastive.py
```

The exporter uses train and validation source records plus the accepted
train-validation distillation bundle by default. It emits only eligible train
pairs for training, preserves diagnostics for incomplete and non-train records,
and records checksums in a manifest.

## Distillation Data Builder

Build answer-only distillation data from local teacher outputs:

```bash
python3 src/build_distillation_data.py --model llama3:latest
```

By default, the builder uses train and validation source records, excludes the
held-out test split, writes raw teacher outputs, filters accepted examples to
parseable `True`/`False` answers matching deterministic gold answers, preserves
rejected diagnostics, and records checksums in a manifest.

## Training Smoke

Run a dependency-light preflight of the CPU-first training loop:

```bash
python3 src/train_smollm2_smoke.py --dry-run --dataset data/grr001/grr001-train-prompts.jsonl --output training-runs/smollm2-135m-dry-run.json
```

Run the actual CPU training smoke only in an environment with `torch` and
`transformers` installed:

```bash
python3 src/train_smollm2_smoke.py --dataset data/grr001/grr001-train-prompts.jsonl --max-steps 5 --batch-size 1
```

Use `--split train` when training from a mixed split file such as the accepted
distillation bundle.
