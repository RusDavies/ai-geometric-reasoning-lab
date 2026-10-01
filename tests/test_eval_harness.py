#!/usr/bin/env python3

import argparse
import unittest

from src import eval_harness


def item_result(parsed_answer: str | None, correct: bool) -> dict:
    return {
        "raw_output": "" if parsed_answer is None else parsed_answer,
        "parsed_answer": parsed_answer,
        "correct": correct,
        "invalid_output": parsed_answer is None,
    }


class EvalHarnessMetricsTests(unittest.TestCase):
    def test_pair_metrics_do_not_double_count_flip_failures(self) -> None:
        pairs = [
            {
                "canonical": item_result("True", True),
                "perturbed": item_result("True", False),
                "both_correct": False,
                "both_wrong": False,
                "exactly_one_correct": True,
                "same_answer_when_gold_flips": True,
                "flip_failure": True,
                "invalid_output": False,
            }
        ]

        metrics = eval_harness.pair_metrics(pairs)

        self.assertEqual(metrics["counts"]["exactly_one_correct"], 1)
        self.assertEqual(metrics["counts"]["same_answer_when_gold_flips"], 1)
        self.assertEqual(metrics["flip_failure_rate"], 1.0)

    def test_grouped_metrics_include_required_breakdowns(self) -> None:
        pairs = [
            {
                "difficulty": 1,
                "hop_count": 1,
                "distractor_count": 0,
                "negation_involved": False,
                "canonical": item_result("True", True),
                "comparison": item_result("False", True),
                "both_correct": True,
                "both_wrong": False,
                "exactly_one_correct": False,
                "same_answer_when_gold_flips": False,
                "answer_changed_when_gold_invariant": False,
                "flip_failure": False,
                "invariance_failure": False,
                "invalid_output": False,
            },
            {
                "difficulty": 2,
                "hop_count": 1,
                "distractor_count": 1,
                "negation_involved": True,
                "canonical": item_result(None, False),
                "comparison": item_result("False", False),
                "both_correct": False,
                "both_wrong": True,
                "exactly_one_correct": False,
                "same_answer_when_gold_flips": False,
                "answer_changed_when_gold_invariant": False,
                "flip_failure": False,
                "invariance_failure": False,
                "invalid_output": True,
            },
        ]

        groups = eval_harness.grouped_metrics(pairs)

        self.assertIn("1", groups["by_difficulty"])
        self.assertIn("2", groups["by_difficulty"])
        self.assertIn("true", groups["by_negation_involved"])
        self.assertIn("false", groups["by_negation_involved"])

    def test_summary_includes_bootstrap_intervals(self) -> None:
        args = argparse.Namespace(bootstrap_samples=50, bootstrap_seed=7)
        pairs = [
            {
                "difficulty": 1,
                "hop_count": 1,
                "distractor_count": 0,
                "negation_involved": False,
                "canonical": item_result("True", True),
                "comparison": item_result("False", True),
                "both_correct": True,
                "both_wrong": False,
                "exactly_one_correct": False,
                "same_answer_when_gold_flips": False,
                "answer_changed_when_gold_invariant": False,
                "flip_failure": False,
                "invariance_failure": False,
                "invalid_output": False,
            }
        ]

        summary = eval_harness.summarize(pairs, args)

        self.assertEqual(summary["confidence_intervals"]["method"], "bootstrap_pairs")
        self.assertIn("canonical_accuracy", summary["confidence_intervals"]["metrics"])

    def test_invariance_metrics_report_answer_changes(self) -> None:
        pairs = [
            {
                "canonical": item_result("True", True),
                "comparison": item_result("False", False),
                "both_correct": False,
                "both_wrong": False,
                "exactly_one_correct": True,
                "same_answer_when_gold_flips": False,
                "answer_changed_when_gold_invariant": True,
                "flip_failure": False,
                "invariance_failure": True,
                "invalid_output": False,
            }
        ]

        metrics = eval_harness.pair_metrics(pairs)

        self.assertEqual(metrics["counts"]["answer_changed_when_gold_invariant"], 1)
        self.assertEqual(metrics["invariance_failure_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
