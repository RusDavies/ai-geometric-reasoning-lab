#!/usr/bin/env python3
"""Export GRR-001 accepted examples as pair-contrastive training records."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .eval_harness import item_contains_negation, load_jsonl
    from .export_grr001_training import sha256_file
except ImportError:
    from eval_harness import item_contains_negation, load_jsonl  # type: ignore
    from export_grr001_training import sha256_file  # type: ignore


SIDES = ("canonical", "perturbed")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def load_source_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for record in load_jsonl(path):
            record_id = record["record_id"]
            if record_id in records:
                raise ValueError(f"duplicate source record_id: {record_id}")
            records[record_id] = record
    return records


def load_accepted_examples(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for example in load_jsonl(path):
        record_id = example["record_id"]
        side = example["side"]
        if side not in SIDES:
            raise ValueError(f"{record_id}: unsupported side {side!r}")
        sides = grouped.setdefault(record_id, {})
        if side in sides:
            raise ValueError(f"{record_id}: duplicate accepted example for {side}")
        sides[side] = example
    return grouped


def pair_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "rail_id": record["rail_id"],
        "split": record["split"],
        "difficulty": record["difficulty"],
        "hop_count": record["hop_count"],
        "distractor_count": record["distractor_count"],
        "negation_involved": item_contains_negation(record["canonical"])
        or item_contains_negation(record["perturbed"]),
        "pair_kind": "flip",
        "perturbation_kind": "load_bearing_polarity_flip",
        "changed_fact_index": record.get("load_bearing_fact_index"),
        "changed_fact_role": "load_bearing",
        "expected_relation": "answer_flip",
        "generation_seed": record.get("generation_seed"),
    }


def side_payload(
    source_record: dict[str, Any],
    accepted_example: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    return {
        "prompt": accepted_example["prompt"],
        "target": accepted_example["target"],
        "gold_answer": source_record[side]["answer"],
        "proof_path": source_record.get("proof_path", []),
        "teacher_provider": accepted_example.get("teacher_provider"),
        "teacher_model": accepted_example.get("teacher_model"),
    }


def diagnostic_record(
    record_id: str,
    reasons: list[str],
    source_record: dict[str, Any] | None,
    accepted_sides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    accepted_sides = accepted_sides or {}
    diagnostic: dict[str, Any] = {
        "record_id": record_id,
        "rejection_reasons": reasons,
        "present_sides": sorted(accepted_sides),
        "missing_sides": [side for side in SIDES if side not in accepted_sides],
        "accepted_targets": {
            side: accepted_sides[side].get("target") for side in sorted(accepted_sides)
        },
    }
    if source_record is not None:
        diagnostic.update(pair_metadata(source_record))
        diagnostic["source_answers"] = {
            side: source_record[side]["answer"] for side in SIDES
        }
    return diagnostic


def rejection_reasons(
    source_record: dict[str, Any],
    accepted_sides: dict[str, dict[str, Any]] | None,
) -> list[str]:
    reasons: list[str] = []
    accepted_sides = accepted_sides or {}
    if source_record["split"] != "train":
        reasons.append("non_train_split_excluded")

    missing_sides = [side for side in SIDES if side not in accepted_sides]
    if missing_sides:
        reasons.append("incomplete_pair")
        return reasons

    source_answers = {side: source_record[side]["answer"] for side in SIDES}
    accepted_targets = {side: accepted_sides[side]["target"] for side in SIDES}

    for side in SIDES:
        if accepted_targets[side] != source_answers[side]:
            reasons.append(f"{side}_target_disagrees_with_source_gold")

    if source_answers["canonical"] == source_answers["perturbed"]:
        reasons.append("source_answers_do_not_flip")
    if accepted_targets["canonical"] == accepted_targets["perturbed"]:
        reasons.append("accepted_targets_do_not_flip")

    return reasons


def build_pair_record(
    source_record: dict[str, Any],
    accepted_sides: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        **pair_metadata(source_record),
        "canonical": side_payload(source_record, accepted_sides["canonical"], "canonical"),
        "perturbed": side_payload(source_record, accepted_sides["perturbed"], "perturbed"),
    }


def export_pair_contrastive_records(
    source_records: dict[str, dict[str, Any]],
    accepted_examples: dict[str, dict[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for record_id in sorted(source_records):
        source_record = source_records[record_id]
        accepted_sides = accepted_examples.get(record_id)
        reasons = rejection_reasons(source_record, accepted_sides)
        if reasons:
            diagnostics.append(
                diagnostic_record(record_id, reasons, source_record, accepted_sides)
            )
            continue
        pairs.append(build_pair_record(source_record, accepted_sides or {}))

    for record_id in sorted(set(accepted_examples) - set(source_records)):
        diagnostics.append(
            diagnostic_record(
                record_id,
                ["accepted_examples_without_source_record"],
                None,
                accepted_examples[record_id],
            )
        )

    return pairs, diagnostics


def build_manifest(
    args: argparse.Namespace,
    source_paths: list[Path],
    accepted_path: Path,
    pair_path: Path,
    diagnostic_path: Path,
    pair_digest: str,
    diagnostic_digest: str,
    source_records: int,
    accepted_examples: int,
    pair_records: int,
    diagnostic_records: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "GR-017",
        "rail_id": "GRR-001",
        "policy": {
            "pair_kind": "flip",
            "expected_relation": "answer_flip",
            "train_split_only": True,
            "requires_both_sides_accepted": True,
            "requires_targets_match_source_gold": True,
            "requires_opposite_targets": True,
            "diagnostics_preserved": True,
            "future_invariance_controls_reserved": True,
        },
        "inputs": {
            "source_records": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "records": len(load_jsonl(path)),
                }
                for path in source_paths
            ],
            "accepted_examples": {
                "path": str(accepted_path),
                "sha256": sha256_file(accepted_path),
                "records": accepted_examples,
            },
        },
        "outputs": {
            "pair_records": {
                "path": str(pair_path),
                "sha256": pair_digest,
                "records": pair_records,
            },
            "diagnostic_records": {
                "path": str(diagnostic_path),
                "sha256": diagnostic_digest,
                "records": diagnostic_records,
            },
        },
        "counts": {
            "source_records": source_records,
            "accepted_examples": accepted_examples,
            "pair_records": pair_records,
            "diagnostic_records": diagnostic_records,
        },
    }


def export_pair_contrastive_data(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = [Path(path) for path in args.source]
    accepted_path = Path(args.accepted)
    pair_path = Path(args.output)
    diagnostic_path = Path(args.diagnostics)
    manifest_path = Path(args.manifest)

    source_records = load_source_records(source_paths)
    accepted_examples = load_accepted_examples(accepted_path)
    pairs, diagnostics = export_pair_contrastive_records(source_records, accepted_examples)

    pair_digest = write_jsonl(pair_path, pairs)
    diagnostic_digest = write_jsonl(diagnostic_path, diagnostics)
    accepted_count = sum(len(sides) for sides in accepted_examples.values())
    manifest = build_manifest(
        args,
        source_paths,
        accepted_path,
        pair_path,
        diagnostic_path,
        pair_digest,
        diagnostic_digest,
        len(source_records),
        accepted_count,
        len(pairs),
        len(diagnostics),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Source GRR-001 JSONL split. Repeat for multiple splits.",
    )
    parser.add_argument(
        "--accepted",
        default="data/grr001/distillation/grr001-llama3-latest-train-validation-accepted.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/grr001/pair-contrastive/grr001-train-pairs.jsonl",
    )
    parser.add_argument(
        "--diagnostics",
        default="data/grr001/pair-contrastive/grr001-train-pair-diagnostics.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="data/grr001/pair-contrastive/grr001-train-pair-manifest.json",
    )
    args = parser.parse_args(argv)
    args.source = args.source or [
        "data/grr001/grr001-train.jsonl",
        "data/grr001/grr001-validation.jsonl",
    ]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_pair_contrastive_data(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
