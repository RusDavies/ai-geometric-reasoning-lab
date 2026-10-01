# Evaluation Rails

Evaluation rails define reproducible task families for testing geometric reasoning.

Each rail must specify:

- the task grammar;
- canonical and perturbed item construction;
- split policy;
- scoring fields;
- leakage and overfitting controls;
- examples that are illustrative rather than part of the held-out set.

## Rails

- `GRR-001-relational-flip.md`: first symbolic relational perturbation rail.
- `GRR-002-invariance-control.md`: symbolic invariance/control rail for
  non-load-bearing perturbations that should preserve the answer.
- `GRR-003-spatial-program.md`: text/program spatial geometry rail with
  deterministic relation labels.
