# GRR-001: Relational Flip Rail

Status: defined
Backlog item: GR-002

## Purpose

Test whether a model tracks load-bearing relational structure through a controlled perturbation.

The rail deliberately uses nonsense entity/type/property names so the model cannot rely on world knowledge. A canonical item and a perturbed item differ by exactly one load-bearing fact. The correct answer must flip when that fact flips.

## Reasoning Target

The model must:

- parse typed relational facts;
- follow one to four inheritance or implication hops;
- ignore distractor facts;
- handle explicit negation;
- answer both sides of a paired item coherently.

This is the first rail because it is easy to generate, easy to score, and hard enough to expose pattern matching. It is not yet a spatial geometry rail.

## Task Grammar

Names are generated from closed vocabularies that are split by dataset partition.

Entity:

```text
<entity> ::= proper nonsense name
```

Type:

```text
<type> ::= nonsense common noun
```

Property:

```text
<property> ::= adjective-like nonsense-safe property token
```

Facts:

```text
Every <type-a> is a <type-b>.
Each <type-a> is a <type-b>.
<type-a>s are <property>.
<type-a>s are not <property>.
<entity> is a <type>.
```

Question:

```text
True or false: <entity> is <property>.
True or false: <entity> is not <property>.
```

Answer:

```text
True
False
```

## Pair Construction

Each record contains a canonical item and a perturbed item.

1. Generate a reasoning chain from entity to queried property.
2. Add distractor facts that are syntactically similar but not on the proof path.
3. Build the canonical item.
4. Flip exactly one load-bearing polarity or implication in the proof path.
5. Recompute the perturbed answer.
6. Reject the pair unless the answer flips and no distractor path also entails the target.

The pair, not the individual item, is the measurement unit.

## Difficulty Levels

- Level 1: one hop, no distractors.
- Level 2: one hop, one to two distractors.
- Level 3: two hops, one to two distractors.
- Level 4: two hops, three to five distractors.
- Level 5: three hops, three to five distractors.
- Level 6: three hops, six to eight distractors.
- Level 7: four hops, six to eight distractors.
- Level 8: four hops, nine to twelve distractors, with at least one negation distractor.

## Record Schema

JSONL record fields:

```json
{
  "rail_id": "GRR-001",
  "record_id": "grr001-train-000001",
  "split": "train",
  "difficulty": 3,
  "hop_count": 2,
  "distractor_count": 2,
  "load_bearing_fact_index": 1,
  "canonical": {
    "facts": ["Every daff is a pilk.", "Pilks are blue.", "Nox is a daff."],
    "question": "True or false: Nox is blue.",
    "answer": "True"
  },
  "perturbed": {
    "facts": ["Every daff is a pilk.", "Pilks are not blue.", "Nox is a daff."],
    "question": "True or false: Nox is blue.",
    "answer": "False"
  },
  "proof_path": ["Nox -> daff", "daff -> pilk", "pilk -> blue"],
  "generation_seed": 12345
}
```

## Split Policy

Use deterministic seeds and separate vocabularies by split.

- `train`: 70% of generated records.
- `validation`: 15% of generated records.
- `test`: 15% of generated records.

Leakage controls:

- entity names do not cross splits;
- type/property vocabularies do not cross test boundaries;
- exact fact sets never cross splits;
- held-out test generation seeds are recorded but not used during training;
- examples in documentation are excluded from all machine-scored splits.

## Scoring

Per item:

- canonical correctness;
- perturbed correctness.

Per pair:

- `both_correct`: canonical and perturbed answers are both correct;
- `flip_failure`: exactly one side is correct, or the model gives the same answer to both sides when the gold answer flips;
- `both_wrong`: neither side is correct;
- `invalid_output`: output cannot be parsed as true/false.

Aggregate metrics:

- canonical accuracy;
- perturbed accuracy;
- both-correct rate;
- flip failure rate;
- invalid-output rate;
- metrics by difficulty, hop count, distractor count, and negation involvement;
- confidence intervals via bootstrap over pairs.

## Baseline Prompt Contract

The first baseline harness should use a plain prompt:

```text
Answer the question using only the facts below. Reply with exactly True or False.

Facts:
<facts>

Question:
<question>
```

No chain-of-thought extraction is required for the first baseline. If rationale prompting is tested later, it must be tracked as a separate prompt condition.

## Acceptance Criteria

GRR-001 is ready for implementation when:

- this rail spec is committed;
- the JSONL schema above is accepted as the initial data contract;
- the split and leakage rules are documented;
- the scoring labels are documented;
- GR-003 can build against this file without needing another requirements conversation.

## Known Limits

- The rail tests symbolic relational robustness, not visual/spatial geometry.
- It does not prove latent-space geometric structure by itself.
- It can be gamed if the generator is too regular, so later iterations should add grammar variation and adversarial validation.
