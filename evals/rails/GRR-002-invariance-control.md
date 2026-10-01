# GRR-002: Invariance Control Rail

Status: defined
Backlog item: GR-018

## Purpose

Test whether a model preserves the correct answer when a prompt is changed in
ways that should not affect the reasoning target.

GRR-001 checks that the answer flips when a load-bearing fact changes. GRR-002
is the paired control: paraphrases, ordering changes, renamings, irrelevant
distractor edits, and non-load-bearing polarity changes must leave the answer
unchanged. This prevents a trained student or evaluation story from learning
the brittle shortcut "perturbation means flip."

## Reasoning Target

The model must:

- parse the same typed relational facts under harmless surface variation;
- preserve the proof path when facts are reordered or paraphrased;
- ignore distractor edits that are not on the proof path;
- treat entity renamings as alpha-equivalent changes;
- reject non-load-bearing polarity changes as evidence for answer flipping;
- answer both sides of an invariant pair coherently.

## Task Grammar

GRR-002 reuses the GRR-001 symbolic relational grammar for the first version so
that the rail isolates invariance behavior rather than introducing a new task
family.

Names are still generated from closed vocabularies split by dataset partition.

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
All <type-a>s are <type-b>s.
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

## Invariance Families

Each record contains a canonical item and an invariant item. The gold answer
must be identical for both items.

### `paraphrase`

Rewrite one or more facts using equivalent templates.

Example:

```text
Every daff is a pilk.
```

may become:

```text
All daffs are pilks.
```

### `fact_order_shuffle`

Shuffle the order of facts while preserving the same fact set.

### `entity_renaming`

Rename generated entities consistently across facts and question. This is an
alpha-equivalence check; the proof structure is unchanged.

### `irrelevant_distractor_edit`

Add, remove, paraphrase, or change a distractor fact that is not reachable from
the queried entity and cannot entail the queried property.

### `non_load_bearing_polarity`

Flip the polarity of a distractor property statement that is provably outside
the proof path.

Example:

```text
Zurls are glim.
```

may become:

```text
Zurls are not glim.
```

only when `zurl -> glim` is unrelated to the queried entity/property proof.

## Pair Construction

1. Generate a GRR-001-compatible canonical item with a valid proof path.
2. Select one invariance family.
3. Apply exactly one non-load-bearing transformation for the base difficulty
   levels, or a bounded combination for higher difficulty levels.
4. Recompute the answer with the same symbolic checker used for generation.
5. Reject the pair unless the canonical and invariant answers are identical.
6. Reject the pair if the invariant transformation changes the proof path
   target, introduces a second proof path to the opposite answer, or makes the
   item ambiguous.

The pair, not the individual item, is the measurement unit.

## Difficulty Levels

- Level 1: one hop, `fact_order_shuffle`, no distractor edits.
- Level 2: one hop, `paraphrase` or `entity_renaming`, one to two distractors.
- Level 3: two hops, one invariance family, one to two distractors.
- Level 4: two hops, `irrelevant_distractor_edit`, three to five distractors.
- Level 5: three hops, `non_load_bearing_polarity`, three to five distractors.
- Level 6: three hops, two compatible invariance families, six to eight
  distractors.
- Level 7: four hops, two compatible invariance families, six to eight
  distractors.
- Level 8: four hops, three compatible invariance families, nine to twelve
  distractors, with at least one irrelevant negation distractor.

## Record Schema

JSONL record fields:

```json
{
  "rail_id": "GRR-002",
  "record_id": "grr002-train-000001",
  "split": "train",
  "difficulty": 4,
  "hop_count": 2,
  "distractor_count": 4,
  "pair_kind": "invariance",
  "invariance_family": "irrelevant_distractor_edit",
  "expected_relation": "answer_invariant",
  "changed_fact_role": "irrelevant_distractor",
  "canonical": {
    "facts": ["Every daff is a pilk.", "Pilks are blue.", "Nox is a daff.", "Zurls are glim."],
    "question": "True or false: Nox is blue.",
    "answer": "True"
  },
  "invariant": {
    "facts": ["Every daff is a pilk.", "Pilks are blue.", "Nox is a daff.", "Zurls are not glim."],
    "question": "True or false: Nox is blue.",
    "answer": "True"
  },
  "proof_path": ["Nox -> daff", "daff -> pilk", "pilk -> blue"],
  "canonical_proof_path": ["Nox -> daff", "daff -> pilk", "pilk -> blue"],
  "invariant_proof_path": ["Nox -> daff", "daff -> pilk", "pilk -> blue"],
  "generation_seed": 23456
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
- exact canonical fact sets never cross splits;
- invariant transformations are balanced by split and difficulty;
- held-out test generation seeds are recorded but not used during training;
- documentation examples are excluded from all machine-scored splits;
- for `entity_renaming`, original and renamed surface forms must not leak
  between train and held-out splits.

## Scoring

Per item:

- canonical correctness;
- invariant correctness.

Per pair:

- `both_correct`: canonical and invariant answers are both correct;
- `invariance_failure`: canonical is correct but invariant is wrong, or the
  model changes its answer when the gold answer is invariant;
- `both_wrong`: neither side is correct;
- `invalid_output`: output cannot be parsed as true/false.

Aggregate metrics:

- canonical accuracy;
- invariant accuracy;
- both-correct rate;
- invariance failure rate;
- invalid-output rate;
- metrics by difficulty, hop count, distractor count, invariance family, and
  negation involvement;
- confidence intervals via bootstrap over pairs.

## Baseline Prompt Contract

Use the same answer-only prompt contract as GRR-001:

```text
Answer the question using only the facts below. Reply with exactly True or False.

Facts:
<facts>

Question:
<question>
```

No chain-of-thought extraction is required for the first baseline. If proof-path
selection is evaluated later, it must be tracked as a separate prompt condition.

## Acceptance Criteria

GRR-002 is ready for implementation when:

- this rail spec is committed;
- the invariance families are documented;
- the JSONL schema is accepted as the initial data contract;
- split, leakage, and scoring rules are documented;
- GRR-001 and GRR-002 together can distinguish answer-flip competence from
  answer-invariance competence;
- a future generator item can build against this file without another
  requirements conversation.

## Known Limits

- The rail is still symbolic relational reasoning, not spatial or formal
  geometry.
- It controls one failure mode of pair-contrastive training, but does not prove
  proof-aware reasoning by itself.
- The first version relies on the same generator family as GRR-001, so later
  rails should add spatial/program structure and verifier-backed proof traces.
