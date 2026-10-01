# Reproducibility

This project is intended to be runnable from a fresh clone with a standard
Python interpreter for the deterministic data and unit-test path. Model-backed
evaluation and training paths require optional services or ML dependencies.

## Environment

Minimum Python version: 3.11.

Create a local virtual environment if desired:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The deterministic path uses only the Python standard library:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

For Hugging Face evaluation or training smoke scripts, install the optional ML
dependencies:

```bash
.venv/bin/python -m pip install -e ".[hf]"
```

## Unit Tests

Run all tests:

```bash
python3 -m unittest discover -s tests -q
```

Run focused tests:

```bash
python3 -m unittest tests.test_generate_grr001 -q
python3 -m unittest tests.test_generate_grr002 -q
python3 -m unittest tests.test_generate_grr003 -q
python3 -m unittest tests.test_eval_harness -q
```

## Dataset Generation

Generate deterministic GRR-001 relational-flip splits:

```bash
python3 src/generate_grr001.py \
  --output-dir data/grr001 \
  --train-count 70 \
  --validation-count 15 \
  --test-count 15
```

Generate deterministic GRR-002 invariance-control splits:

```bash
python3 src/generate_grr002.py \
  --output-dir data/grr002 \
  --train-count 70 \
  --validation-count 15 \
  --test-count 15
```

Generate deterministic GRR-003 spatial/program geometry splits:

```bash
python3 src/generate_grr003.py --output-dir data/grr003
```

Export GRR-001 training prompts:

```bash
python3 src/export_grr001_training.py \
  --input data/grr001/grr001-train.jsonl \
  --output data/grr001/grr001-train-prompts.jsonl
```

## Local Ollama Evaluation

The local harness calls an Ollama-compatible HTTP endpoint. Start Ollama and
pull the model separately, then run:

```bash
python3 src/eval_harness.py \
  --model smollm2:135m \
  --dataset data/grr001-mini.jsonl \
  --output runs/grr001-mini-ollama.json
```

The output records model identity, prompt template, decoding options, raw
outputs, parsed answers, pair metrics, confidence intervals, and breakdowns.

## Hugging Face Evaluation

Install optional ML dependencies first:

```bash
python3 -m pip install -e ".[hf]"
```

Run generation-mode evaluation:

```bash
python3 src/eval_hf_causal.py \
  --dataset data/grr001/grr001-validation.jsonl \
  --model-id HuggingFaceTB/SmolLM2-135M \
  --output evals/results/grr001-validation-hf-smollm2-135m-base.json
```

Run answer-choice evaluation:

```bash
python3 src/eval_hf_causal.py \
  --dataset data/grr001/grr001-validation.jsonl \
  --model-id HuggingFaceTB/SmolLM2-135M \
  --answer-mode choice \
  --output evals/results/grr001-validation-hf-smollm2-135m-base-choice.json
```

Run GRR-003 closed-label answer-choice evaluation:

```bash
python3 src/eval_hf_causal.py \
  --dataset data/grr003/grr003-validation.jsonl \
  --model-id HuggingFaceTB/SmolLM2-135M \
  --answer-mode choice \
  --output evals/results/grr003-validation-hf-smollm2-135m-base-choice.json
```

## Training Smoke

Run the dependency-light dry run:

```bash
python3 src/train_smollm2_smoke.py \
  --dry-run \
  --dataset data/grr001/grr001-train-prompts.jsonl \
  --output training-runs/smollm2-135m-dry-run.json
```

Run the actual training smoke only after installing optional ML dependencies:

```bash
python3 src/train_smollm2_smoke.py \
  --dataset data/grr001/grr001-train-prompts.jsonl \
  --max-steps 5 \
  --batch-size 1
```

## Expected Limits

The included small datasets and smoke runs are reproducibility fixtures, not
strong performance claims. Model-quality claims belong in
`claims/CLAIM_LEDGER.md` and must point to tracked code, data, settings, and
result files.
