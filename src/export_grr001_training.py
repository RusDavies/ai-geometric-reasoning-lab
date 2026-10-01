#!/usr/bin/env python3
"""Export GRR-001 pairs as prompt/target training examples."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .eval_harness import build_prompt, item_contains_negation, load_jsonl
except ImportError:
    from eval_harness import build_prompt, item_contains_negation, load_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def example_from_side(record: dict[str, Any], side: str) -> dict[str, Any]:
    item = record[side]
    return {
        "record_id": record["record_id"],
        "rail_id": record["rail_id"],
        "split": record["split"],
        "side": side,
        "difficulty": record["difficulty"],
        "hop_count": record["hop_count"],
        "distractor_count": record["distractor_count"],
        "negation_involved": item_contains_negation(record["canonical"])
        or item_contains_negation(record["perturbed"]),
        "prompt": build_prompt(item),
        "target": item["answer"],
    }


def export_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        examples.append(example_from_side(record, "canonical"))
        examples.append(example_from_side(record, "perturbed"))
    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            line = json.dumps(example, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/grr001/grr001-train.jsonl")
    parser.add_argument("--output", default="data/grr001/grr001-train-prompts.jsonl")
    parser.add_argument("--manifest", default="data/grr001/training-export-manifest.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    records = load_jsonl(input_path)
    examples = export_examples(records)
    output_digest = write_jsonl(output_path, examples)
    manifest = {
        "source": str(input_path),
        "source_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": output_digest,
        "records": len(records),
        "examples": len(examples),
        "sides": ["canonical", "perturbed"],
        "target_values": ["True", "False"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
