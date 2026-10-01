# Data

Data recipes, small fixtures, and generated dataset notes belong here.

## Fixtures

- `grr001-mini.jsonl`: a tiny GRR-001 validation fixture for local harness
  smoke tests and first baseline runs. It is intentionally small enough to run
  quickly and is not a performance benchmark.

## Generated Splits

- `grr001/manifest.json`: generated split manifest with seeds and checksums.
- `grr001/grr001-train.jsonl`: deterministic training split.
- `grr001/grr001-validation.jsonl`: deterministic validation split.
- `grr001/grr001-test.jsonl`: deterministic held-out test split.
- `grr001/grr001-train-prompts.jsonl`: prompt/target training export from the
  deterministic training split.
- `grr001/training-export-manifest.json`: source/output checksums for the
  training export.
- `grr001/distillation/`: teacher-output, accepted-example, rejected-diagnostic,
  and manifest files for answer-only distillation data generation.
- `grr001/pair-contrastive/`: pair-shaped train records, rejected/excluded pair
  diagnostics, and manifest files for pair-contrastive training slices.
- `grr002/manifest.json`: generated GRR-002 invariance-control split manifest
  with seeds, checksums, and invariance-family counts.
- `grr002/grr002-train.jsonl`: deterministic training split.
- `grr002/grr002-validation.jsonl`: deterministic validation split.
- `grr002/grr002-test.jsonl`: deterministic held-out test split.
