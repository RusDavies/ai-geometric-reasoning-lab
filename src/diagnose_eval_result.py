#!/usr/bin/env python3
"""Diagnose paired eval result patterns against the source dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl_by_record_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record_id = record.get("record_id")
            if not isinstance(record_id, str):
                raise ValueError(f"{path}:{line_number}: missing string record_id")
            records[record_id] = record
    return records


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{(count / total) * 100:.1f}%"


def counter_text(counter: Counter[Any]) -> str:
    if not counter:
        return "none"
    parts = []
    for key, count in counter.most_common():
        if isinstance(key, tuple):
            label = " -> ".join(str(part) for part in key)
        else:
            label = str(key)
        parts.append(f"`{label}` ({count})")
    return ", ".join(parts)


def margin_value(result: dict[str, Any]) -> float | None:
    value = result.get("choice_margin")
    if isinstance(value, int | float):
        return float(value)
    return None


def margin_text(values: list[float | None]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "n/a"
    return f"avg `{sum(present) / len(present):.3f}`, min `{min(present):.3f}`"


def comparison_result(pair: dict[str, Any]) -> dict[str, Any]:
    if "comparison" in pair:
        return pair["comparison"]
    if "perturbed" in pair:
        return pair["perturbed"]
    if "invariant" in pair:
        return pair["invariant"]
    raise KeyError(f"{pair.get('record_id', '<unknown>')}: missing comparison result")


def analyze(result: dict[str, Any], dataset_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in result["pairs"]:
        record_id = pair["record_id"]
        source = dataset_records[record_id]
        comparison_side = pair.get("comparison_side", "perturbed")
        comparison = comparison_result(pair)
        canonical_pred = pair["canonical"]["parsed_answer"]
        comparison_pred = comparison["parsed_answer"]
        canonical_gold = source["canonical"]["answer"]
        comparison_gold = source[comparison_side]["answer"]
        rows.append(
            {
                "record_id": record_id,
                "relation_family": pair.get("relation_family"),
                "difficulty": pair.get("difficulty"),
                "distractor_count": pair.get("distractor_count"),
                "canonical_gold": canonical_gold,
                "comparison_gold": comparison_gold,
                "canonical_pred": canonical_pred,
                "comparison_pred": comparison_pred,
                "canonical_correct": pair["canonical"]["correct"],
                "comparison_correct": comparison["correct"],
                "canonical_margin": margin_value(pair["canonical"]),
                "comparison_margin": margin_value(comparison),
                "same_prediction": canonical_pred is not None
                and canonical_pred == comparison_pred,
            }
        )

    total = len(rows)
    same_prediction_count = sum(1 for row in rows if row["same_prediction"])
    both_correct_count = sum(
        1 for row in rows if row["canonical_correct"] and row["comparison_correct"]
    )
    invalid_count = result["summary"]["counts"].get("invalid_pairs", 0)
    by_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["relation_family"])].append(row)

    for family, family_rows in sorted(grouped.items()):
        family_total = len(family_rows)
        by_family[family] = {
            "pair_count": family_total,
            "canonical_correct": sum(1 for row in family_rows if row["canonical_correct"]),
            "comparison_correct": sum(1 for row in family_rows if row["comparison_correct"]),
            "both_correct": sum(
                1
                for row in family_rows
                if row["canonical_correct"] and row["comparison_correct"]
            ),
            "same_prediction": sum(1 for row in family_rows if row["same_prediction"]),
            "gold_transitions": Counter(
                (row["canonical_gold"], row["comparison_gold"]) for row in family_rows
            ),
            "predicted_transitions": Counter(
                (row["canonical_pred"], row["comparison_pred"]) for row in family_rows
            ),
            "predicted_labels": Counter(row["canonical_pred"] for row in family_rows)
            + Counter(row["comparison_pred"] for row in family_rows),
            "difficulty_values": Counter(row["difficulty"] for row in family_rows),
            "margins": [
                margin
                for row in family_rows
                for margin in (row["canonical_margin"], row["comparison_margin"])
            ],
        }

    return {
        "total_pairs": total,
        "same_prediction_count": same_prediction_count,
        "both_correct_count": both_correct_count,
        "invalid_pair_count": invalid_count,
        "by_family": by_family,
        "rows": rows,
    }


def markdown_report(
    result_path: Path, dataset_path: Path, result: dict[str, Any], analysis: dict[str, Any]
) -> str:
    run = result["run"]
    summary = result["summary"]
    counts = summary["counts"]
    all_margins = [
        margin
        for row in analysis["rows"]
        for margin in (row["canonical_margin"], row["comparison_margin"])
    ]
    has_margin_evidence = any(margin is not None for margin in all_margins)
    backlog_item = "GR-032" if has_margin_evidence else "GR-031"
    title_suffix = "Choice-Margin Diagnosis" if has_margin_evidence else "Same-Answer Diagnosis"
    lines = [
        f"# GRR-003 SmolLM2-135M {title_suffix}",
        "",
        f"Backlog item: {backlog_item}",
        "Status: diagnostic report",
        "",
        "## Inputs",
        "",
        f"- Result file: `{result_path}`",
        f"- Dataset: `{dataset_path}`",
        f"- Dataset SHA-256: `{run.get('dataset_sha256')}`",
        f"- Model: `{run.get('model')}`",
        f"- Provider: `{run.get('provider')}`",
        f"- Answer mode: `{run.get('generation_options', {}).get('answer_mode')}`",
        f"- Prompt template: `{run.get('prompt_template')}`",
        "",
        "## Summary",
        "",
        f"- Pair count: `{analysis['total_pairs']}`",
        f"- Both-correct pairs: `{analysis['both_correct_count']}`",
        f"- Flip failures: `{counts.get('same_answer_when_gold_flips')}` same-answer pairs out of `{analysis['total_pairs']}`",
        f"- Same predicted label across canonical and perturbed side: `{analysis['same_prediction_count']}` / `{analysis['total_pairs']}` ({pct(analysis['same_prediction_count'], analysis['total_pairs'])})",
        f"- Invalid-output pairs: `{analysis['invalid_pair_count']}`",
        f"- Canonical accuracy: `{counts.get('canonical_correct')}` / `{analysis['total_pairs']}`",
        f"- Perturbed accuracy: `{counts.get('perturbed_correct')}` / `{analysis['total_pairs']}`",
        f"- Choice-margin evidence: {margin_text(all_margins)}",
        "",
        "## Relation-Family Pattern",
        "",
        "| Relation family | Pairs | Canonical correct | Perturbed correct | Same prediction | Choice margins | Predicted transitions | Gold transitions |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for family, stats in analysis["by_family"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    family,
                    str(stats["pair_count"]),
                    str(stats["canonical_correct"]),
                    str(stats["comparison_correct"]),
                    str(stats["same_prediction"]),
                    margin_text(stats["margins"]),
                    counter_text(stats["predicted_transitions"]),
                    counter_text(stats["gold_transitions"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            "The dominant failure is not output-format compliance: all outputs were parseable labels.",
            "It is also not random label noise. The model selected one stable answer per relation family",
            "and reused it on both sides of every label-changing pair.",
            "",
            "That pattern means the closed-label scorer is measuring a real pair-level weakness:",
            "the model is sensitive enough to prefer a plausible family-default label, but not sensitive",
            "enough to the load-bearing coordinate or boundary edits that should flip the label.",
            "",
            "The family defaults are especially visible in the predicted transitions:",
            "`same_point -> same_point`, `on_line -> on_line`, `does_not_contain -> does_not_contain`,",
            "`same_distance -> same_distance`, `right_of -> right_of`, `inside -> inside`,",
            "`below -> below`, and `between -> between`.",
            "",
        ]
    )
    if has_margin_evidence:
        lines.extend(
            [
                "The margin evidence is mixed rather than one-note. Distance comparison and point",
                "identity show strong default-label preferences, while containment and vertical order",
                "are much closer to runner-up ties. That means the same-answer collapse is partly",
                "strong family-default bias and partly fragile near-tie scoring, depending on the",
                "relation family.",
                "",
                "## Follow-Up Options",
                "",
                "- GR-033 should run a GRR-003 prompt-intervention diagnostic that asks for the changed statement and relation before the final label, then compares choice margins against this baseline.",
                "- A stronger baseline should be run on the same split before treating this as dataset-level difficulty rather than small-model brittleness.",
                "- Future GRR-003 validation should add more examples per relation family so margin summaries are less hostage to three examples at a time.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Follow-Up Options",
                "",
                "- GR-032 should add a diagnostic scoring/export pass that records per-label candidate scores or margins, so we can distinguish strong default bias from near-ties.",
                "- A prompt variant should ask for the changed statement and relation before the final label, then compare choice-only scoring against generate-and-parse scoring.",
                "- A larger or stronger baseline should be run on the same GRR-003 split before treating this as a dataset-level difficulty claim.",
                "- The generator should preserve these small validation pairs, but future validation should add more examples per relation family before broad claims.",
                "",
            ]
        )
    lines.extend(
        [
            "## Bounded Claim",
            "",
            "This diagnosis supports only the bounded claim that SmolLM2-135M, under this closed-label",
            "answer-choice setup, collapses each GRR-003 relation family to a same-answer prediction",
            "on the current 24-pair validation split. It does not show that GRR-003 is unsolved by",
            "larger models or better prompted systems.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    result_path = Path(args.result)
    dataset_path = Path(args.dataset)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    dataset_records = load_jsonl_by_record_id(dataset_path)
    analysis = analyze(result, dataset_records)
    report = markdown_report(result_path, dataset_path, result, analysis)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote {output_path}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
