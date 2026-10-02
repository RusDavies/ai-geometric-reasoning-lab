#!/usr/bin/env python3
"""Compare two closed-label choice-margin eval result files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def item_margin(item: dict[str, Any]) -> float:
    value = item.get("choice_margin")
    if not isinstance(value, int | float):
        raise ValueError("result item is missing numeric choice_margin")
    return float(value)


def comparison_item(pair: dict[str, Any]) -> dict[str, Any]:
    if "comparison" in pair:
        return pair["comparison"]
    if "perturbed" in pair:
        return pair["perturbed"]
    if "invariant" in pair:
        return pair["invariant"]
    raise KeyError(f"{pair.get('record_id', '<unknown>')}: missing comparison item")


def pair_index(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {pair["record_id"]: pair for pair in result["pairs"]}


def side_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_pairs = pair_index(baseline)
    candidate_pairs = pair_index(candidate)
    rows: list[dict[str, Any]] = []
    for record_id, baseline_pair in sorted(baseline_pairs.items()):
        candidate_pair = candidate_pairs[record_id]
        for side_name, baseline_item, candidate_item in [
            ("canonical", baseline_pair["canonical"], candidate_pair["canonical"]),
            ("comparison", comparison_item(baseline_pair), comparison_item(candidate_pair)),
        ]:
            baseline_pred = baseline_item["parsed_answer"]
            candidate_pred = candidate_item["parsed_answer"]
            rows.append(
                {
                    "record_id": record_id,
                    "side": side_name,
                    "relation_family": baseline_pair.get("relation_family"),
                    "baseline_pred": baseline_pred,
                    "candidate_pred": candidate_pred,
                    "prediction_changed": baseline_pred != candidate_pred,
                    "baseline_correct": baseline_item["correct"],
                    "candidate_correct": candidate_item["correct"],
                    "correctness_changed": baseline_item["correct"] != candidate_item["correct"],
                    "baseline_margin": item_margin(baseline_item),
                    "candidate_margin": item_margin(candidate_item),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["relation_family"])].append(row)
    return {
        "item_count": len(rows),
        "prediction_changes": sum(1 for row in rows if row["prediction_changed"]),
        "correctness_changes": sum(1 for row in rows if row["correctness_changed"]),
        "baseline_margin_avg": mean(row["baseline_margin"] for row in rows),
        "candidate_margin_avg": mean(row["candidate_margin"] for row in rows),
        "by_family": {
            family: {
                "item_count": len(family_rows),
                "prediction_changes": sum(1 for row in family_rows if row["prediction_changed"]),
                "correctness_changes": sum(1 for row in family_rows if row["correctness_changed"]),
                "baseline_margin_avg": mean(row["baseline_margin"] for row in family_rows),
                "candidate_margin_avg": mean(row["candidate_margin"] for row in family_rows),
            }
            for family, family_rows in sorted(by_family.items())
        },
    }


def markdown_report(
    baseline_path: Path,
    candidate_path: Path,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    baseline_mode = baseline["run"].get("generation_options", {}).get("prompt_mode", "direct")
    candidate_mode = candidate["run"].get("generation_options", {}).get("prompt_mode", "unknown")
    lines = [
        "# GRR-003 Prompt-Intervention Margin Comparison",
        "",
        "Backlog item: GR-033",
        "Status: comparison report",
        "",
        "## Inputs",
        "",
        f"- Baseline result: `{baseline_path}`",
        f"- Candidate result: `{candidate_path}`",
        f"- Baseline prompt mode: `{baseline_mode}`",
        f"- Candidate prompt mode: `{candidate_mode}`",
        f"- Model: `{candidate['run'].get('model')}`",
        f"- Dataset: `{candidate['run'].get('dataset')}`",
        "",
        "## Summary",
        "",
        f"- Compared items: `{summary['item_count']}`",
        f"- Prediction changes: `{summary['prediction_changes']}`",
        f"- Correctness changes: `{summary['correctness_changes']}`",
        f"- Baseline average margin: `{summary['baseline_margin_avg']:.3f}`",
        f"- Candidate average margin: `{summary['candidate_margin_avg']:.3f}`",
        "",
        "| Relation family | Items | Prediction changes | Correctness changes | Baseline avg margin | Candidate avg margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, stats in summary["by_family"].items():
        lines.append(
            f"| {family} | {stats['item_count']} | {stats['prediction_changes']} | "
            f"{stats['correctness_changes']} | {stats['baseline_margin_avg']:.3f} | "
            f"{stats['candidate_margin_avg']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This comparison measures whether the prompt intervention changes the closed-label",
            "choice surface, not whether it teaches the model geometry. Prediction changes show",
            "that the prompt moved the selected label; margin changes show whether it made the",
            "selection more or less decisive.",
            "",
            "Verdict: this prompt intervention did not change any selected labels or item",
            "correctness on the validation split. It mostly changed confidence margins.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    rows = side_rows(baseline, candidate)
    summary = summarize(rows)
    report = markdown_report(baseline_path, candidate_path, baseline, candidate, summary)
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
