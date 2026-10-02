#!/usr/bin/env python3

import unittest

from src import compare_choice_margin_results


def item(label: str, correct: bool, margin: float) -> dict:
    return {
        "parsed_answer": label,
        "correct": correct,
        "choice_margin": margin,
    }


class CompareChoiceMarginResultsTests(unittest.TestCase):
    def test_summary_counts_prediction_and_correctness_changes(self) -> None:
        baseline = {
            "pairs": [
                {
                    "record_id": "r1",
                    "relation_family": "point_identity",
                    "canonical": item("same_point", True, 2.0),
                    "comparison": item("same_point", False, 2.5),
                }
            ]
        }
        candidate = {
            "pairs": [
                {
                    "record_id": "r1",
                    "relation_family": "point_identity",
                    "canonical": item("different_point", False, 0.5),
                    "comparison": item("same_point", False, 0.75),
                }
            ]
        }

        rows = compare_choice_margin_results.side_rows(baseline, candidate)
        summary = compare_choice_margin_results.summarize(rows)

        self.assertEqual(summary["item_count"], 2)
        self.assertEqual(summary["prediction_changes"], 1)
        self.assertEqual(summary["correctness_changes"], 1)
        self.assertAlmostEqual(summary["baseline_margin_avg"], 2.25)
        self.assertAlmostEqual(summary["candidate_margin_avg"], 0.625)


if __name__ == "__main__":
    unittest.main()
