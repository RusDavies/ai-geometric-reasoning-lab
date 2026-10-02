#!/usr/bin/env python3

import unittest
from collections import Counter

from src import diagnose_eval_result


class DiagnoseEvalResultTests(unittest.TestCase):
    def test_analyze_joins_dataset_gold_answers(self) -> None:
        result = {
            "summary": {"counts": {"invalid_pairs": 0}},
            "pairs": [
                {
                    "record_id": "r1",
                    "relation_family": "point_identity",
                    "difficulty": 1,
                    "distractor_count": 0,
                    "comparison_side": "perturbed",
                    "canonical": {
                        "parsed_answer": "same_point",
                        "correct": True,
                        "choice_margin": 1.5,
                    },
                    "comparison": {
                        "parsed_answer": "same_point",
                        "correct": False,
                        "choice_margin": 1.25,
                    },
                }
            ],
        }
        dataset = {
            "r1": {
                "canonical": {"answer": "same_point"},
                "perturbed": {"answer": "different_point"},
            }
        }

        analysis = diagnose_eval_result.analyze(result, dataset)

        self.assertEqual(analysis["total_pairs"], 1)
        self.assertEqual(analysis["same_prediction_count"], 1)
        self.assertEqual(analysis["both_correct_count"], 0)
        family = analysis["by_family"]["point_identity"]
        self.assertEqual(family["gold_transitions"][("same_point", "different_point")], 1)
        self.assertEqual(family["predicted_transitions"][("same_point", "same_point")], 1)
        self.assertEqual(family["margins"], [1.5, 1.25])

    def test_markdown_report_includes_follow_up_options(self) -> None:
        result = {
            "run": {
                "dataset_sha256": "abc",
                "model": "test-model",
                "provider": "test-provider",
                "generation_options": {"answer_mode": "choice"},
                "prompt_template": "test-template",
            },
            "summary": {
                "counts": {
                    "same_answer_when_gold_flips": 1,
                    "canonical_correct": 1,
                    "perturbed_correct": 0,
                }
            },
        }
        analysis = {
            "total_pairs": 1,
            "both_correct_count": 0,
            "same_prediction_count": 1,
            "invalid_pair_count": 0,
            "rows": [{"canonical_margin": 1.5, "comparison_margin": 1.25}],
            "by_family": {
                "point_identity": {
                    "pair_count": 1,
                    "canonical_correct": 1,
                    "comparison_correct": 0,
                    "same_prediction": 1,
                    "margins": [1.5, 1.25],
                    "predicted_transitions": Counter({("same_point", "same_point"): 1}),
                    "gold_transitions": Counter({("same_point", "different_point"): 1}),
                }
            },
        }

        report = diagnose_eval_result.markdown_report(
            result_path="result.json",
            dataset_path="dataset.jsonl",
            result=result,
            analysis=analysis,
        )

        self.assertIn("GR-032", report)
        self.assertIn("GR-033", report)
        self.assertIn("same_point -> same_point", report)
        self.assertIn("avg `1.375`", report)


if __name__ == "__main__":
    unittest.main()
