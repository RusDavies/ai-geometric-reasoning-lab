#!/usr/bin/env python3
"""Generate deterministic GRR-001 relational-flip datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SPLIT_SEEDS = {
    "train": 11001,
    "validation": 22001,
    "test": 33001,
}

SPLIT_PREFIXES = {
    "train": "tr",
    "validation": "va",
    "test": "te",
}

DIFFICULTY_BANDS = {
    1: (1, (0, 0), False),
    2: (1, (1, 2), False),
    3: (2, (1, 2), False),
    4: (2, (3, 5), False),
    5: (3, (3, 5), False),
    6: (3, (6, 8), False),
    7: (4, (6, 8), False),
    8: (4, (9, 12), True),
}


@dataclass(frozen=True)
class SplitVocab:
    entities: list[str]
    types: list[str]
    properties: list[str]


def make_token(prefix: str, kind: str, index: int) -> str:
    stems = [
        "baf",
        "cem",
        "dov",
        "feg",
        "gip",
        "hax",
        "jol",
        "kiv",
        "lum",
        "mep",
        "nid",
        "pav",
        "qor",
        "rux",
        "siv",
        "taz",
        "vop",
        "wex",
        "yim",
        "zun",
    ]
    return f"{prefix}{kind}{stems[index % len(stems)]}{index:03d}"


def build_vocab(split: str, count: int) -> SplitVocab:
    prefix = SPLIT_PREFIXES[split]
    pool_size = max(250, count * 25)
    return SplitVocab(
        entities=[make_token(prefix, "e", index).capitalize() for index in range(pool_size)],
        types=[make_token(prefix, "t", index) for index in range(pool_size)],
        properties=[make_token(prefix, "p", index) for index in range(pool_size)],
    )


def plural(type_name: str) -> str:
    return f"{type_name}s"


def difficulty_for_index(index: int) -> int:
    return (index % 8) + 1


def choose_distractor_count(rng: random.Random, difficulty: int) -> int:
    _, (low, high), _ = DIFFICULTY_BANDS[difficulty]
    return rng.randint(low, high)


def type_fact(source: str, target: str, rng: random.Random) -> str:
    quantifier = rng.choice(["Every", "Each"])
    return f"{quantifier} {source} is a {target}."


def property_fact(type_name: str, property_name: str, positive: bool) -> str:
    if positive:
        return f"{plural(type_name)} are {property_name}."
    return f"{plural(type_name)} are not {property_name}."


def make_distractors(
    rng: random.Random,
    vocab: SplitVocab,
    start_index: int,
    count: int,
    target_property: str,
    require_negation: bool,
) -> list[str]:
    distractors: list[str] = []
    for offset in range(count):
        type_a = vocab.types[start_index + offset * 2]
        type_b = vocab.types[start_index + offset * 2 + 1]
        prop = vocab.properties[start_index + offset + 10]
        if offset % 3 == 0:
            distractors.append(type_fact(type_a, type_b, rng))
        elif offset % 3 == 1:
            distractors.append(property_fact(type_a, prop, positive=True))
        else:
            distractors.append(property_fact(type_a, prop, positive=False))

    if require_negation and not any(" not " in f" {fact} " for fact in distractors):
        type_name = vocab.types[start_index + count * 2 + 2]
        distractors.append(property_fact(type_name, target_property, positive=False))
    return distractors[:count]


def make_item(
    entity: str,
    chain_types: list[str],
    property_name: str,
    target_positive: bool,
    distractors: list[str],
    chain_facts: list[str],
) -> dict[str, Any]:
    facts = [
        *chain_facts,
        property_fact(chain_types[-1], property_name, target_positive),
        *distractors,
        f"{entity} is a {chain_types[0]}.",
    ]
    return {
        "facts": facts,
        "question": f"True or false: {entity} is {property_name}.",
        "answer": "True" if target_positive else "False",
    }


def generate_record(split: str, index: int, vocab: SplitVocab) -> dict[str, Any]:
    seed = SPLIT_SEEDS[split] + index
    rng = random.Random(seed)
    difficulty = difficulty_for_index(index)
    hop_count, _, require_negation = DIFFICULTY_BANDS[difficulty]
    distractor_count = choose_distractor_count(rng, difficulty)

    stride = 20
    base = index * stride
    entity = vocab.entities[index]
    chain_types = vocab.types[base : base + hop_count]
    property_name = vocab.properties[index]
    canonical_positive = bool(index % 2)
    distractors = make_distractors(
        rng,
        vocab,
        base + hop_count + 1,
        distractor_count,
        property_name,
        require_negation,
    )
    chain_facts = [
        type_fact(source, target, rng)
        for source, target in zip(chain_types, chain_types[1:])
    ]

    canonical = make_item(
        entity,
        chain_types,
        property_name,
        canonical_positive,
        distractors,
        chain_facts,
    )
    perturbed = make_item(
        entity,
        chain_types,
        property_name,
        not canonical_positive,
        distractors,
        chain_facts,
    )
    load_bearing_fact_index = hop_count - 1
    proof_path = [f"{entity} -> {chain_types[0]}"]
    proof_path.extend(
        f"{source} -> {target}" for source, target in zip(chain_types, chain_types[1:])
    )
    proof_path.append(f"{chain_types[-1]} -> {property_name}")

    return {
        "rail_id": "GRR-001",
        "record_id": f"grr001-{split}-{index + 1:06d}",
        "split": split,
        "difficulty": difficulty,
        "hop_count": hop_count,
        "distractor_count": distractor_count,
        "load_bearing_fact_index": load_bearing_fact_index,
        "canonical": canonical,
        "perturbed": perturbed,
        "proof_path": proof_path,
        "generation_seed": seed,
    }


def generate_split(split: str, count: int) -> list[dict[str, Any]]:
    vocab = build_vocab(split, count)
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


def fact_signature(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(record["canonical"]["facts"]), tuple(record["perturbed"]["facts"])


def validate_splits(generated: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    all_entities: dict[str, set[str]] = {}
    all_types: dict[str, set[str]] = {}
    all_properties: dict[str, set[str]] = {}
    fact_signatures: set[tuple[str, ...]] = set()

    for split, records in generated.items():
        entities: set[str] = set()
        types: set[str] = set()
        properties: set[str] = set()
        for record in records:
            if record["canonical"]["answer"] == record["perturbed"]["answer"]:
                errors.append(f"{record['record_id']}: answers do not flip")
            if record["split"] != split:
                errors.append(f"{record['record_id']}: wrong split {record['split']}")
            entities.add(record["proof_path"][0].split(" -> ")[0])
            for step in record["proof_path"][1:-1]:
                source, target = step.split(" -> ")
                types.update([source, target])
            last_type, prop = record["proof_path"][-1].split(" -> ")
            types.add(last_type)
            properties.add(prop)
            signature = fact_signature(record)
            if signature in fact_signatures:
                errors.append(f"{record['record_id']}: duplicate canonical/perturbed fact set")
            fact_signatures.add(signature)

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
    parser.add_argument("--output-dir", default="data/grr001")
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
        for error in errors:
            print(error)
        return 1

    output_dir = Path(args.output_dir)
    manifest = {
        "rail_id": "GRR-001",
        "generator": "src/generate_grr001.py",
        "split_seeds": SPLIT_SEEDS,
        "split_counts": split_counts,
        "files": {},
    }
    for split, records in generated.items():
        path = output_dir / f"grr001-{split}.jsonl"
        digest = write_jsonl(path, records)
        manifest["files"][split] = {
            "path": str(path),
            "records": len(records),
            "sha256": digest,
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
