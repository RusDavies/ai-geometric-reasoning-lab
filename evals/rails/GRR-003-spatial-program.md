# GRR-003: Spatial Program Geometry Rail

Status: defined
Backlog item: GR-019

## Purpose

Test whether a model can infer spatial and simple geometric relations from a
small deterministic scene program.

GRR-001 and GRR-002 test symbolic relational robustness. GRR-003 introduces
actual geometric content: points, lines, axis-aligned regions, containment,
distance comparison, left/right ordering, inside/outside, and between
relations. It is inspired by program-to-geometry benchmarks such as
GeoGramBench, but the first version is intentionally smaller, text-only, and
deterministically checkable.

## Reasoning Target

The model must:

- parse a compact coordinate scene program;
- bind named points, lines, segments, and boxes;
- compute or compare simple geometric relations;
- track a load-bearing coordinate or shape perturbation;
- return an exact relation label from a closed answer set;
- answer both sides of a paired item coherently.

This rail is not a formal Euclidean theorem prover. It is a bridge from the
earlier symbolic rails to spatial/program geometry with deterministic labels
and a verifier-friendly data contract.

## Task Grammar

Names are generated from closed vocabularies that are split by dataset
partition. Coordinates are small signed integers unless a difficulty level
explicitly permits half-integers.

Scene program:

```text
point <point-name> = (<x>, <y>)
line <line-name> = line(<point-name>, <point-name>)
segment <segment-name> = segment(<point-name>, <point-name>)
box <box-name> = box(<x-min>, <y-min>, <x-max>, <y-max>)
```

Question:

```text
What is the relation for <query-id>?
```

Answer:

```text
<relation-label>
```

The prompt must include the allowed labels for the selected relation family so
the model is not being scored on label discovery.

## Relation Families

Each item chooses one relation family. A generator may add distractor objects,
but the query must identify the objects being scored.

### `point_identity`

Compare two points.

Allowed labels:

```text
same_point
different_point
```

### `line_membership`

Determine whether a point lies on a named line or segment.

Allowed labels:

```text
on_line
off_line
on_segment
off_segment
```

### `containment`

Determine whether a box contains a point or another axis-aligned box.

Allowed labels:

```text
contains
does_not_contain
```

### `distance_comparison`

Compare the distance from an anchor point to two candidate points.

Allowed labels:

```text
closer_to_first
closer_to_second
same_distance
```

Squared Euclidean distance is used so generation and checking can stay exact.

### `horizontal_order`

Compare the x-coordinate ordering of two points.

Allowed labels:

```text
left_of
right_of
same_x
```

### `vertical_order`

Compare the y-coordinate ordering of two points.

Allowed labels:

```text
below
above
same_y
```

### `inside`

Determine whether a point is inside, on the boundary of, or outside a box.

Allowed labels:

```text
inside
boundary
outside
```

### `between`

Determine whether a point lies between two endpoints on the same horizontal,
vertical, or diagonal segment.

Allowed labels:

```text
between
not_between
```

## Pair Construction

Each record contains a canonical item and a perturbed item.

1. Generate a scene program with one target query and optional distractor
   objects.
2. Compute the canonical relation label with the deterministic checker.
3. Apply exactly one load-bearing perturbation to a coordinate, endpoint, or
   box boundary used by the query.
4. Recompute the perturbed relation label.
5. Reject the pair unless the label changes and both labels are in the allowed
   set for the relation family.
6. Reject ambiguous or near-boundary cases unless the relation family
   explicitly targets the boundary label.
7. Record the query objects, changed program statement, and verifier trace.

The pair, not the individual item, is the measurement unit.

## Difficulty Levels

- Level 1: one relation, no distractors, integer coordinates from `0` to `5`.
- Level 2: one relation, one to two distractor points or shapes.
- Level 3: line or segment membership with horizontal or vertical geometry.
- Level 4: containment or inside/outside with one distractor box.
- Level 5: distance comparison using squared distances, no equal-distance
  cases.
- Level 6: between relations across horizontal, vertical, or 45-degree
  diagonal segments.
- Level 7: mixed scene with three to five distractors and half-integer
  coordinates.
- Level 8: mixed scene with six to eight distractors, explicit boundary cases,
  and relation-family-balanced perturbations.

## Record Schema

JSONL record fields:

```json
{
  "rail_id": "GRR-003",
  "record_id": "grr003-train-000001",
  "split": "train",
  "difficulty": 5,
  "relation_family": "distance_comparison",
  "pair_kind": "spatial_label_flip",
  "expected_relation": "label_changes",
  "changed_statement_index": 2,
  "canonical": {
    "program": [
      "point A = (0, 0)",
      "point B = (2, 0)",
      "point C = (5, 0)"
    ],
    "query": "What is the relation for distance(A, B, C)?",
    "allowed_labels": ["closer_to_first", "closer_to_second", "same_distance"],
    "answer": "closer_to_first"
  },
  "perturbed": {
    "program": [
      "point A = (0, 0)",
      "point B = (6, 0)",
      "point C = (5, 0)"
    ],
    "query": "What is the relation for distance(A, B, C)?",
    "allowed_labels": ["closer_to_first", "closer_to_second", "same_distance"],
    "answer": "closer_to_second"
  },
  "query_objects": ["A", "B", "C"],
  "verifier_trace": {
    "canonical": "dist2(A,B)=4; dist2(A,C)=25",
    "perturbed": "dist2(A,B)=36; dist2(A,C)=25"
  },
  "generation_seed": 34567
}
```

## Split Policy

Use deterministic seeds and separate vocabularies by split.

- `train`: 70% of generated records.
- `validation`: 15% of generated records.
- `test`: 15% of generated records.

Leakage controls:

- object names do not cross splits;
- exact scene programs never cross splits;
- canonical and perturbed program pairs never cross splits;
- coordinate tuples are sampled from split-specific seed ranges where possible;
- relation families are balanced within each split;
- held-out test generation seeds are recorded but not used during training;
- documentation examples are excluded from all machine-scored splits.

## Scoring

Per item:

- canonical label correctness;
- perturbed label correctness.

Per pair:

- `both_correct`: canonical and perturbed labels are both correct;
- `label_change_failure`: exactly one side is correct, or the model gives the
  same label to both sides when the gold label changes;
- `both_wrong`: neither side is correct;
- `invalid_output`: output is not one of the allowed labels for the selected
  relation family.

Aggregate metrics:

- canonical label accuracy;
- perturbed label accuracy;
- both-correct rate;
- label-change failure rate;
- invalid-output rate;
- metrics by difficulty, relation family, distractor count, and boundary
  involvement;
- confidence intervals via bootstrap over pairs.

## Baseline Prompt Contract

The first baseline harness should use a closed-label prompt:

```text
Answer the question using only the scene program below. Reply with exactly one
allowed label.

Allowed labels:
<allowed-labels>

Scene program:
<program>

Question:
<query>
```

If a rationale or verifier-trace prompt is tested later, it must be tracked as
a separate prompt condition.

## Acceptance Criteria

GRR-003 is ready for implementation when:

- this rail spec is committed;
- the relation families and allowed labels are documented;
- the JSONL schema is accepted as the initial data contract;
- split, leakage, and scoring rules are documented;
- the rail has a deterministic checker target for every relation family;
- a future generator item can build against this file without another
  requirements conversation.

## Known Limits

- The rail is text/program geometry, not diagram parsing.
- It does not claim AlphaGeometry-style theorem proving, auxiliary
  construction, or formal proof search.
- It tests direct spatial relation inference before representation-level
  geometry claims.
- Boundary cases must be handled carefully because tiny coordinate changes can
  create ambiguous natural-language interpretations even when the checker is
  deterministic.
