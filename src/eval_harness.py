#!/usr/bin/env python3
"""Minimal paired eval harness for GRR-001/GRR-002 records."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


PROMPT_TEMPLATE = """Answer the question using only the facts below. Reply with exactly True or False.

Facts:
{facts}

Question:
{question}
"""


@dataclass(frozen=True)
class ItemResult:
    raw_output: str
    parsed_answer: str | None
    correct: bool
    invalid_output: bool


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if limit is not None and len(records) >= limit:
                break
    return records


def build_prompt(item: dict[str, Any]) -> str:
    facts = "\n".join(f"- {fact}" for fact in item["facts"])
    return PROMPT_TEMPLATE.format(facts=facts, question=item["question"])


def parse_true_false(output: str) -> str | None:
    tokens = output.strip().replace(".", " ").replace(",", " ").split()
    if not tokens:
        return None
    first = tokens[0].casefold()
    if first == "true":
        return "True"
    if first == "false":
        return "False"
    return None


def run_ollama(model: str, prompt: str, args: argparse.Namespace) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "seed": args.seed,
            "num_predict": args.num_predict,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    api_request = request.Request(
        args.ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(api_request, timeout=args.timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["response"].strip()


def evaluate_item(model: str, item: dict[str, Any], args: argparse.Namespace) -> ItemResult:
    raw_output = run_ollama(model, build_prompt(item), args)
    parsed = parse_true_false(raw_output)
    gold = item["answer"]
    return ItemResult(
        raw_output=raw_output,
        parsed_answer=parsed,
        correct=parsed == gold,
        invalid_output=parsed is None,
    )


def rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def item_contains_negation(item: dict[str, Any]) -> bool:
    return any(" not " in f" {fact.casefold()} " for fact in item["facts"])


def comparison_side_key(record: dict[str, Any]) -> str:
    if "perturbed" in record:
        return "perturbed"
    if "invariant" in record:
        return "invariant"
    raise KeyError(f"{record.get('record_id', '<unknown>')}: missing comparison side")


def comparison_result(result: dict[str, Any]) -> dict[str, Any]:
    if "comparison" in result:
        return result["comparison"]
    if "perturbed" in result:
        return result["perturbed"]
    if "invariant" in result:
        return result["invariant"]
    raise KeyError(f"{result.get('record_id', '<unknown>')}: missing comparison result")


def result_answer_changed(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["parsed_answer"] is not None
        and right["parsed_answer"] is not None
        and left["parsed_answer"] != right["parsed_answer"]
    )


def pair_metrics(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(pair_results)
    canonical_correct = sum(1 for result in pair_results if result["canonical"]["correct"])
    comparison_correct = sum(1 for result in pair_results if comparison_result(result)["correct"])
    both_correct = sum(1 for result in pair_results if result["both_correct"])
    both_wrong = sum(1 for result in pair_results if result["both_wrong"])
    invalid_pairs = sum(1 for result in pair_results if result["invalid_output"])
    same_answer_when_gold_flips = sum(1 for result in pair_results if result["same_answer_when_gold_flips"])
    answer_changed_when_gold_invariant = sum(
        1 for result in pair_results if result.get("answer_changed_when_gold_invariant", False)
    )
    exactly_one_correct = sum(1 for result in pair_results if result["exactly_one_correct"])
    flip_failures = sum(1 for result in pair_results if result["flip_failure"])
    invariance_failures = sum(1 for result in pair_results if result.get("invariance_failure", False))

    metrics = {
        "pair_count": total,
        "canonical_accuracy": rate(canonical_correct, total),
        "comparison_accuracy": rate(comparison_correct, total),
        "perturbed_accuracy": rate(comparison_correct, total),
        "invariant_accuracy": rate(comparison_correct, total),
        "both_correct_rate": rate(both_correct, total),
        "flip_failure_rate": rate(flip_failures, total),
        "invariance_failure_rate": rate(invariance_failures, total),
        "invalid_output_rate": rate(invalid_pairs, total),
        "counts": {
            "canonical_correct": canonical_correct,
            "comparison_correct": comparison_correct,
            "perturbed_correct": comparison_correct,
            "invariant_correct": comparison_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "exactly_one_correct": exactly_one_correct,
            "same_answer_when_gold_flips": same_answer_when_gold_flips,
            "answer_changed_when_gold_invariant": answer_changed_when_gold_invariant,
            "invariance_failures": invariance_failures,
            "invalid_pairs": invalid_pairs,
        },
    }
    return metrics


def bootstrap_confidence_intervals(
    pair_results: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    if not pair_results or samples <= 0:
        return {}

    rng = random.Random(seed)
    metric_samples: dict[str, list[float]] = {
        "canonical_accuracy": [],
        "comparison_accuracy": [],
        "perturbed_accuracy": [],
        "invariant_accuracy": [],
        "both_correct_rate": [],
        "flip_failure_rate": [],
        "invariance_failure_rate": [],
        "invalid_output_rate": [],
    }
    for _ in range(samples):
        resample = [rng.choice(pair_results) for _ in pair_results]
        metrics = pair_metrics(resample)
        for key in metric_samples:
            metric_samples[key].append(metrics[key])

    intervals: dict[str, dict[str, float]] = {}
    for key, values in metric_samples.items():
        ordered = sorted(values)
        low_index = int(0.025 * (len(ordered) - 1))
        high_index = int(0.975 * (len(ordered) - 1))
        intervals[key] = {
            "low": ordered[low_index],
            "high": ordered[high_index],
        }
    return intervals


def grouped_metrics(pair_results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    group_specs = {
        "by_difficulty": "difficulty",
        "by_hop_count": "hop_count",
        "by_distractor_count": "distractor_count",
        "by_negation_involved": "negation_involved",
        "by_invariance_family": "invariance_family",
    }
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for group_name, field_name in group_specs.items():
        grouped_results: dict[str, list[dict[str, Any]]] = {}
        for result in pair_results:
            if field_name not in result:
                continue
            key = str(result[field_name]).lower()
            grouped_results.setdefault(key, []).append(result)
        if grouped_results:
            groups[group_name] = {
                key: pair_metrics(results) for key, results in sorted(grouped_results.items())
            }
    return groups


def summarize(pair_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary = pair_metrics(pair_results)
    summary["confidence_intervals"] = {
        "method": "bootstrap_pairs",
        "confidence": 0.95,
        "samples": args.bootstrap_samples,
        "seed": args.bootstrap_seed,
        "metrics": bootstrap_confidence_intervals(
            pair_results, args.bootstrap_samples, args.bootstrap_seed
        ),
    }
    summary["groups"] = grouped_metrics(pair_results)
    return summary


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    records = load_jsonl(dataset_path, args.limit)
    pair_results: list[dict[str, Any]] = []
    for record in records:
        canonical = evaluate_item(args.model, record["canonical"], args)
        comparison_side = comparison_side_key(record)
        comparison = evaluate_item(args.model, record[comparison_side], args)
        expected_relation = record.get("expected_relation", "answer_flip")
        both_correct = canonical.correct and comparison.correct
        both_wrong = not canonical.correct and not comparison.correct
        invalid_output = canonical.invalid_output or comparison.invalid_output
        same_answer_when_gold_flips = (
            expected_relation == "answer_flip"
            and canonical.parsed_answer is not None
            and canonical.parsed_answer == comparison.parsed_answer
        )
        answer_changed_when_gold_invariant = (
            expected_relation == "answer_invariant"
            and result_answer_changed(canonical.__dict__, comparison.__dict__)
        )
        exactly_one_correct = canonical.correct != comparison.correct
        flip_failure = expected_relation == "answer_flip" and (
            exactly_one_correct or same_answer_when_gold_flips
        )
        invariance_failure = expected_relation == "answer_invariant" and (
            (canonical.correct and not comparison.correct) or answer_changed_when_gold_invariant
        )
        pair_results.append(
            {
                "record_id": record["record_id"],
                "rail_id": record["rail_id"],
                "difficulty": record["difficulty"],
                "hop_count": record["hop_count"],
                "distractor_count": record["distractor_count"],
                "pair_kind": record.get("pair_kind"),
                "expected_relation": expected_relation,
                "comparison_side": comparison_side,
                "invariance_family": record.get("invariance_family"),
                "negation_involved": item_contains_negation(record["canonical"])
                or item_contains_negation(record[comparison_side]),
                "canonical": canonical.__dict__,
                comparison_side: comparison.__dict__,
                "comparison": comparison.__dict__,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "exactly_one_correct": exactly_one_correct,
                "same_answer_when_gold_flips": same_answer_when_gold_flips,
                "answer_changed_when_gold_invariant": answer_changed_when_gold_invariant,
                "flip_failure": flip_failure,
                "invariance_failure": invariance_failure,
                "invalid_output": invalid_output,
            }
        )

    return {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "provider": "ollama",
            "ollama_url": args.ollama_url,
            "generation_options": {
                "temperature": args.temperature,
                "seed": args.seed,
                "num_predict": args.num_predict,
            },
            "dataset": str(dataset_path),
            "prompt_template": PROMPT_TEMPLATE,
        },
        "summary": summarize(pair_results, args),
        "pairs": pair_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/grr001-mini.jsonl")
    parser.add_argument("--model", default="smollm2:135m")
    parser.add_argument("--output", default="runs/grr001-mini-ollama.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-predict", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1001)
    args = parser.parse_args()

    try:
        result = run_eval(args)
    except (RuntimeError, TimeoutError, error.URLError) as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
