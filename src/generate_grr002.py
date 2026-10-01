#!/usr/bin/env python3
"""Generate deterministic GRR-002 invariance-control datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

try:
    from . import generate_grr001
except ImportError:
    import generate_grr001  # type: ignore


SPLIT_SEEDS = {
    "train": 41001,
    "validation": 52001,
    "test": 63001,
}

INVARIANCE_FAMILIES = [
    "fact_order_shuffle",
    "paraphrase",
    "entity_renaming",
    "irrelevant_distractor_edit",
    "non_load_bearing_polarity",
]


def plural_stem(plural_type: str) -> str:
    return plural_type[:-1] if plural_type.endswith("s") else plural_type


def paraphrase_fact(fact: str) -> str:
    if fact.startswith("Every ") and " is a " in fact:
        source, target = fact[len("Every ") : -1].split(" is a ")
        return f"All {generate_grr001.plural(source)} are {target}s."
    if fact.startswith("Each ") and " is a " in fact:
        source, target = fact[len("Each ") : -1].split(" is a ")
        return f"Every {source} is a {target}."
    return fact


def ensure_distractor_for_family(
    record: dict[str, Any],
    family: str,
    vocab: generate_grr001.SplitVocab,
    index: int,
) -> None:
    if family not in {"irrelevant_distractor_edit", "non_load_bearing_polarity"}:
        return

    first_distractor_index = record["hop_count"]
    last_distractor_index = len(record["canonical"]["facts"]) - 1
    distractors = record["canonical"]["facts"][first_distractor_index:last_distractor_index]
    if family == "irrelevant_distractor_edit" and distractors:
        return
    if family == "non_load_bearing_polarity" and any(" are " in fact for fact in distractors):
        return

    type_name = vocab.types[index * 20 + record["hop_count"] + record["distractor_count"] + 30]
    property_name = vocab.properties[index + 30]
    new_fact = generate_grr001.property_fact(type_name, property_name, positive=True)
    record["canonical"]["facts"].insert(last_distractor_index, new_fact)
    record["distractor_count"] += 1


def distractor_indices(record: dict[str, Any]) -> list[int]:
    first_distractor_index = record["hop_count"]
    last_distractor_index = len(record["canonical"]["facts"]) - 1
    return list(range(first_distractor_index, last_distractor_index))


def apply_fact_order_shuffle(
    canonical: dict[str, Any], rng: random.Random
) -> tuple[dict[str, Any], str]:
    invariant = {
        "facts": list(canonical["facts"]),
        "question": canonical["question"],
        "answer": canonical["answer"],
    }
    rng.shuffle(invariant["facts"])
    if invariant["facts"] == canonical["facts"]:
        invariant["facts"] = invariant["facts"][1:] + invariant["facts"][:1]
    return invariant, "fact_order"


def apply_paraphrase(
    canonical: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    invariant = {
        "facts": list(canonical["facts"]),
        "question": canonical["question"],
        "answer": canonical["answer"],
    }
    for index, fact in enumerate(invariant["facts"]):
        rewritten = paraphrase_fact(fact)
        if rewritten != fact:
            invariant["facts"][index] = rewritten
            return invariant, "proof_path_fact"

    target_index = record["hop_count"] - 1
    invariant["facts"][target_index] = paraphrase_fact(invariant["facts"][target_index])
    return invariant, "proof_path_fact"


def apply_entity_renaming(
    canonical: dict[str, Any],
    record: dict[str, Any],
    vocab: generate_grr001.SplitVocab,
    index: int,
) -> tuple[dict[str, Any], str, list[str]]:
    old_entity = record["proof_path"][0].split(" -> ")[0]
    new_entity = vocab.entities[index + 100]
    invariant = {
        "facts": [fact.replace(old_entity, new_entity) for fact in canonical["facts"]],
        "question": canonical["question"].replace(old_entity, new_entity),
        "answer": canonical["answer"],
    }
    invariant_proof_path = list(record["proof_path"])
    invariant_proof_path[0] = invariant_proof_path[0].replace(old_entity, new_entity)
    return invariant, "entity_surface", invariant_proof_path


def apply_irrelevant_distractor_edit(
    canonical: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    invariant = {
        "facts": list(canonical["facts"]),
        "question": canonical["question"],
        "answer": canonical["answer"],
    }
    for index in distractor_indices(record):
        rewritten = paraphrase_fact(invariant["facts"][index])
        if rewritten != invariant["facts"][index]:
            invariant["facts"][index] = rewritten
            return invariant, "irrelevant_distractor"

    edit_index = distractor_indices(record)[0]
    invariant["facts"][edit_index] = flip_property_polarity(invariant["facts"][edit_index])
    return invariant, "irrelevant_distractor"


def flip_property_polarity(fact: str) -> str:
    if " are not " in fact:
        return fact.replace(" are not ", " are ", 1)
    if " are " in fact:
        return fact.replace(" are ", " are not ", 1)
    return fact


def apply_non_load_bearing_polarity(
    canonical: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    invariant = {
        "facts": list(canonical["facts"]),
        "question": canonical["question"],
        "answer": canonical["answer"],
    }
    for index in distractor_indices(record):
        flipped = flip_property_polarity(invariant["facts"][index])
        if flipped != invariant["facts"][index]:
            invariant["facts"][index] = flipped
            return invariant, "irrelevant_distractor"
    raise ValueError(f"{record['record_id']}: no distractor property fact to flip")


def family_for_index(index: int) -> str:
    return INVARIANCE_FAMILIES[index % len(INVARIANCE_FAMILIES)]


def generate_record(split: str, index: int, vocab: generate_grr001.SplitVocab) -> dict[str, Any]:
    base_record = generate_grr001.generate_record(split, index, vocab)
    family = family_for_index(index)
    seed = SPLIT_SEEDS[split] + index
    rng = random.Random(seed)
    ensure_distractor_for_family(base_record, family, vocab, index)
    canonical = base_record["canonical"]
    invariant_proof_path = list(base_record["proof_path"])

    if family == "fact_order_shuffle":
        invariant, changed_fact_role = apply_fact_order_shuffle(canonical, rng)
    elif family == "paraphrase":
        invariant, changed_fact_role = apply_paraphrase(canonical, base_record)
    elif family == "entity_renaming":
        invariant, changed_fact_role, invariant_proof_path = apply_entity_renaming(
            canonical, base_record, vocab, index
        )
    elif family == "irrelevant_distractor_edit":
        invariant, changed_fact_role = apply_irrelevant_distractor_edit(canonical, base_record)
    elif family == "non_load_bearing_polarity":
        invariant, changed_fact_role = apply_non_load_bearing_polarity(canonical, base_record)
    else:
        raise ValueError(f"unknown invariance family: {family}")

    return {
        "rail_id": "GRR-002",
        "record_id": f"grr002-{split}-{index + 1:06d}",
        "split": split,
        "difficulty": base_record["difficulty"],
        "hop_count": base_record["hop_count"],
        "distractor_count": base_record["distractor_count"],
        "pair_kind": "invariance",
        "invariance_family": family,
        "expected_relation": "answer_invariant",
        "changed_fact_role": changed_fact_role,
        "canonical": canonical,
        "invariant": invariant,
        "proof_path": base_record["proof_path"],
        "canonical_proof_path": base_record["proof_path"],
        "invariant_proof_path": invariant_proof_path,
        "generation_seed": seed,
    }


def generate_split(split: str, count: int) -> list[dict[str, Any]]:
    vocab = generate_grr001.build_vocab(split, count + 10)
    return [generate_record(split, index, vocab) for index in range(count)]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def fact_signature(record: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(record["canonical"]["facts"]), tuple(record["invariant"]["facts"])


def validate_splits(generated: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    all_entities: dict[str, set[str]] = {}
    all_types: dict[str, set[str]] = {}
    all_properties: dict[str, set[str]] = {}
    fact_signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    for split, records in generated.items():
        entities: set[str] = set()
        types: set[str] = set()
        properties: set[str] = set()
        family_counts = {family: 0 for family in INVARIANCE_FAMILIES}
        for record in records:
            if record["canonical"]["answer"] != record["invariant"]["answer"]:
                errors.append(f"{record['record_id']}: answers are not invariant")
            if record["expected_relation"] != "answer_invariant":
                errors.append(f"{record['record_id']}: wrong expected relation")
            if record["split"] != split:
                errors.append(f"{record['record_id']}: wrong split {record['split']}")
            if fact_signature(record) in fact_signatures:
                errors.append(f"{record['record_id']}: duplicate canonical/invariant fact set")
            fact_signatures.add(fact_signature(record))
            family_counts[record["invariance_family"]] += 1
            entities.add(record["canonical_proof_path"][0].split(" -> ")[0])
            entities.add(record["invariant_proof_path"][0].split(" -> ")[0])
            for step in record["canonical_proof_path"][1:-1]:
                source, target = step.split(" -> ")
                types.update([source, target])
            last_type, prop = record["canonical_proof_path"][-1].split(" -> ")
            types.add(last_type)
            properties.add(prop)

        if records:
            counts = list(family_counts.values())
            if max(counts) - min(counts) > 1:
                errors.append(f"{split}: invariance families are not balanced: {family_counts}")
        all_entities[split] = entities
        all_types[split] = types
        all_properties[split] = properties

    splits = sorted(generated)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            if all_entities[left] & all_entities[right]:
                errors.append(f"{left}/{right}: entity vocabulary overlap")
            if all_types[left] & all_types[right]:
                errors.append(f"{left}/{right}: type vocabulary overlap")
            if all_properties[left] & all_properties[right]:
                errors.append(f"{left}/{right}: property vocabulary overlap")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/grr002")
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=15)
    args = parser.parse_args()

    split_counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    generated = {
        split: generate_split(split, count) for split, count in split_counts.items()
    }
    errors = validate_splits(generated)
    if errors:
        for validation_error in errors:
            print(validation_error)
        return 1

    output_dir = Path(args.output_dir)
    manifest: dict[str, Any] = {
        "rail_id": "GRR-002",
        "generator": "src/generate_grr002.py",
        "split_seeds": SPLIT_SEEDS,
        "split_counts": split_counts,
        "invariance_families": INVARIANCE_FAMILIES,
        "expected_relation": "answer_invariant",
        "files": {},
    }
    for split, records in generated.items():
        path = output_dir / f"grr002-{split}.jsonl"
        digest = write_jsonl(path, records)
        family_counts: dict[str, int] = {}
        for record in records:
            family_counts[record["invariance_family"]] = (
                family_counts.get(record["invariance_family"], 0) + 1
            )
        manifest["files"][split] = {
            "path": str(path),
            "records": len(records),
            "sha256": digest,
            "invariance_family_counts": dict(sorted(family_counts.items())),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
