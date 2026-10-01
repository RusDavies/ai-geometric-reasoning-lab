#!/usr/bin/env python3
"""Build answer-only GRR-001 distillation data from teacher outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib import error

try:
    from .eval_harness import (
        build_prompt,
        item_contains_negation,
        load_jsonl,
        parse_true_false,
        run_ollama,
    )
    from .export_grr001_training import sha256_file
except ImportError:
    from eval_harness import (  # type: ignore
        build_prompt,
        item_contains_negation,
        load_jsonl,
        parse_true_false,
        run_ollama,
    )
    from export_grr001_training import sha256_file  # type: ignore


GenerateTeacherOutput = Callable[[dict[str, Any]], str]


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-").lower()


def source_side_metadata(record: dict[str, Any], side: str) -> dict[str, Any]:
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
        "generation_seed": record.get("generation_seed"),
        "load_bearing_fact_index": record.get("load_bearing_fact_index"),
    }


def teacher_record_from_output(
    record: dict[str, Any],
    side: str,
    raw_output: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    item = record[side]
    parsed = parse_true_false(raw_output)
    gold = item["answer"]
    metadata = source_side_metadata(record, side)
    output_record = {
        **metadata,
        "prompt": build_prompt(item),
        "gold_answer": gold,
        "teacher": {
            "provider": args.provider,
            "model": args.model,
            "ollama_url": args.ollama_url if args.provider == "ollama" else None,
            "generation_options": {
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
            },
            "raw_output": raw_output,
            "parsed_answer": parsed,
        },
        "correct": parsed == gold,
        "invalid_output": parsed is None,
    }
    output_record["rejection_reasons"] = rejection_reasons(output_record)
    return output_record


def rejection_reasons(teacher_record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if teacher_record["invalid_output"]:
        reasons.append("invalid_true_false_output")
    if not teacher_record["correct"]:
        reasons.append("teacher_answer_disagrees_with_gold")
    return reasons


def accepted_example(teacher_record: dict[str, Any]) -> dict[str, Any]:
    parsed_answer = teacher_record["teacher"]["parsed_answer"]
    if parsed_answer not in {"True", "False"}:
        raise ValueError("accepted teacher record must have parsed True/False answer")
    return {
        "record_id": teacher_record["record_id"],
        "rail_id": teacher_record["rail_id"],
        "split": teacher_record["split"],
        "side": teacher_record["side"],
        "difficulty": teacher_record["difficulty"],
        "hop_count": teacher_record["hop_count"],
        "distractor_count": teacher_record["distractor_count"],
        "negation_involved": teacher_record["negation_involved"],
        "teacher_provider": teacher_record["teacher"]["provider"],
        "teacher_model": teacher_record["teacher"]["model"],
        "prompt": teacher_record["prompt"],
        "target": parsed_answer,
    }


def split_teacher_records(
    teacher_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for teacher_record in teacher_records:
        if teacher_record["rejection_reasons"]:
            rejected.append(teacher_record)
        else:
            accepted.append(accepted_example(teacher_record))
    return accepted, rejected


def read_fixture_outputs(path: Path) -> dict[tuple[str, str], str]:
    outputs: dict[tuple[str, str], str] = {}
    for record in load_jsonl(path):
        key = (record["record_id"], record["side"])
        outputs[key] = record["raw_output"]
    return outputs


def collect_teacher_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    generate_output: GenerateTeacherOutput,
) -> list[dict[str, Any]]:
    teacher_records: list[dict[str, Any]] = []
    total = len(records) * 2
    for record in records:
        for side in ("canonical", "perturbed"):
            item = record[side]
            raw_output = generate_output(item)
            teacher_records.append(teacher_record_from_output(record, side, raw_output, args))
            if args.progress_every and len(teacher_records) % args.progress_every == 0:
                print(
                    f"generated {len(teacher_records)}/{total} teacher outputs",
                    file=sys.stderr,
                    flush=True,
                )
    return teacher_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode("utf-8"))
            handle.write(line)
    return digest.hexdigest()


def load_source_records(paths: list[Path], limit_per_file: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path, limit_per_file))
    return records


def build_manifest(
    args: argparse.Namespace,
    input_paths: list[Path],
    raw_output_path: Path,
    accepted_output_path: Path,
    rejected_output_path: Path,
    raw_digest: str,
    accepted_digest: str,
    rejected_digest: str,
    source_records: int,
    raw_records: int,
    accepted_records: int,
    rejected_records: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "GR-012",
        "rail_id": "GRR-001",
        "policy": {
            "distillation_type": "answer_only",
            "accepted_rule": "parseable True/False output matching deterministic gold answer",
            "rejected_outputs_preserved": True,
            "held_out_test_excluded": True,
        },
        "teacher": {
            "provider": args.provider,
            "model": args.model,
            "ollama_url": args.ollama_url if args.provider == "ollama" else None,
            "generation_options": {
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
            },
        },
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(load_jsonl(path)),
            }
            for path in input_paths
        ],
        "outputs": {
            "raw_teacher_outputs": {
                "path": str(raw_output_path),
                "sha256": raw_digest,
                "records": raw_records,
            },
            "accepted_distillation_examples": {
                "path": str(accepted_output_path),
                "sha256": accepted_digest,
                "records": accepted_records,
            },
            "rejected_diagnostics": {
                "path": str(rejected_output_path),
                "sha256": rejected_digest,
                "records": rejected_records,
            },
        },
        "counts": {
            "source_records": source_records,
            "teacher_output_records": raw_records,
            "accepted_examples": accepted_records,
            "rejected_diagnostics": rejected_records,
        },
    }


def build_distillation_data(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(path) for path in args.input]
    source_records = load_source_records(input_paths, args.limit_per_file)

    if args.fixture_outputs:
        fixture_outputs = read_fixture_outputs(Path(args.fixture_outputs))
        teacher_records: list[dict[str, Any]] = []
        for record in source_records:
            for side in ("canonical", "perturbed"):
                key = (record["record_id"], side)
                if key not in fixture_outputs:
                    raise SystemExit(f"missing fixture output for {record['record_id']} {side}")
                teacher_records.append(
                    teacher_record_from_output(record, side, fixture_outputs[key], args)
                )
    else:

        def generate_output(item: dict[str, Any]) -> str:
            if args.provider != "ollama":
                raise SystemExit(f"unsupported provider without fixture outputs: {args.provider}")
            return run_ollama(args.model, build_prompt(item), args)

        teacher_records = collect_teacher_records(source_records, args, generate_output)

    accepted, rejected = split_teacher_records(teacher_records)
    raw_output_path = Path(args.raw_output)
    accepted_output_path = Path(args.accepted_output)
    rejected_output_path = Path(args.rejected_output)
    manifest_path = Path(args.manifest)

    raw_digest = write_jsonl(raw_output_path, teacher_records)
    accepted_digest = write_jsonl(accepted_output_path, accepted)
    rejected_digest = write_jsonl(rejected_output_path, rejected)
    manifest = build_manifest(
        args,
        input_paths,
        raw_output_path,
        accepted_output_path,
        rejected_output_path,
        raw_digest,
        accepted_digest,
        rejected_digest,
        len(source_records),
        len(teacher_records),
        len(accepted),
        len(rejected),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Source GRR-001 JSONL split. Repeat for multiple splits.",
    )
    parser.add_argument("--model", default="llama3:latest")
    parser.add_argument("--provider", default="ollama", choices=["ollama"])
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=8)
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--fixture-outputs")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--raw-output")
    parser.add_argument("--accepted-output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--manifest")

    default_inputs = [
        "data/grr001/grr001-train.jsonl",
        "data/grr001/grr001-validation.jsonl",
    ]
    args = parser.parse_args(argv)
    args.input = args.input or default_inputs

    model_slug = safe_model_name(args.model)
    default_stem = f"grr001-{model_slug}-train-validation"
    args.raw_output = (
        args.raw_output
        or f"data/grr001/distillation/{default_stem}-teacher-outputs.jsonl"
    )
    args.accepted_output = (
        args.accepted_output
        or f"data/grr001/distillation/{default_stem}-accepted.jsonl"
    )
    args.rejected_output = (
        args.rejected_output
        or f"data/grr001/distillation/{default_stem}-rejected.jsonl"
    )
    args.manifest = args.manifest or f"data/grr001/distillation/{default_stem}-manifest.json"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_distillation_data(args)
    except (RuntimeError, TimeoutError, error.URLError) as exc:
        print(f"distillation-data build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
