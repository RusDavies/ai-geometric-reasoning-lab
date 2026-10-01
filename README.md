# ai-geometric-reasoning-lab

Empirical tooling for perturbation-based geometric and relational reasoning
evaluation, small-model training experiments, and bounded claim tracking.

## Purpose

This is an empirical ML project. The first target is not a grand theory of mind; it is a reproducible evaluation and training workflow for models that preserve reasoning structure under controlled perturbation.

The project treats reasoning quality as something to measure through controlled
changes to load-bearing problem structure. If a model answers an original item
correctly but fails after a minimal structural perturbation, the model has not
demonstrated robust reasoning on that pair.

## Bounded Scope

This repository does not claim AGI, consciousness, frontier-model replacement,
or data-center obsolescence. Claims should be read only as far as the tracked
code, data, settings, and result files support them.

Start here:

- `PRODUCT_BRIEF.md` defines the product thesis, scope, success criteria, and first useful outcome.
- `REPRODUCIBILITY.md` gives setup, dependency, test, generation, eval, and training-smoke commands.
- `evals/rails/` defines perturbation reasoning rails.
- `claims/CLAIM_LEDGER.md` separates measured results, failed hypotheses, and speculation.
- `RESEARCH_NOTES.md` and `SOURCES.md` preserve evidence and review notes.
- `src/README.md` describes the current generators, eval harnesses, and training smoke scripts.

## Quickstart

Run the unit tests:

```bash
python3 -m unittest discover -s tests -q
```

Generate deterministic GRR-001 relational-flip splits:

```bash
python3 src/generate_grr001.py --output-dir data/grr001 --train-count 70 --validation-count 15 --test-count 15
```

Generate deterministic GRR-002 invariance-control splits:

```bash
python3 src/generate_grr002.py --output-dir data/grr002 --train-count 70 --validation-count 15 --test-count 15
```

Run the local Ollama eval harness when an Ollama model is available:

```bash
python3 src/eval_harness.py --model smollm2:135m --dataset data/grr001-mini.jsonl --output runs/grr001-mini-ollama.json
```

## Installation Notes

The deterministic generators and unit tests use the Python standard library.

Install the project metadata in editable mode from a virtual environment:

```bash
python3 -m pip install -e .
```

Hugging Face evaluation and training smoke scripts require optional ML
dependencies such as `torch` and `transformers`:

```bash
python3 -m pip install -e ".[hf]"
```

See `REPRODUCIBILITY.md` for the full command set.

## Citation And Sources

Citation metadata is provided in `CITATION.cff`.

Research sources are listed in `SOURCES.md`. Evidence summaries and measured
claims are tracked in `claims/CLAIM_LEDGER.md`.

## License

Apache-2.0. See `LICENSE`.
