#!/usr/bin/env python3

import unittest

from src import eval_hf_causal, eval_harness, generate_grr001, generate_grr002


class EvalHFCausalTests(unittest.TestCase):
    def test_select_answer_from_scores_chooses_highest_score(self) -> None:
        answer = eval_hf_causal.select_answer_from_scores({"True": -2.0, "False": -0.5})

        self.assertEqual(answer, "False")

    def test_evaluate_records_reuses_pair_metrics_contract(self) -> None:
        records = generate_grr001.generate_split("validation", 1)

        def run_item(item: dict) -> eval_harness.ItemResult:
            return eval_harness.ItemResult(
                raw_output=item["answer"],
                parsed_answer=item["answer"],
                correct=True,
                invalid_output=False,
            )

        pair_results = eval_hf_causal.evaluate_records(records, run_item)

        self.assertEqual(len(pair_results), 1)
        self.assertTrue(pair_results[0]["both_correct"])
        self.assertFalse(pair_results[0]["flip_failure"])

    def test_evaluate_records_handles_invariance_pairs(self) -> None:
        records = generate_grr002.generate_split("validation", 1)

        def run_item(item: dict) -> eval_harness.ItemResult:
            parsed = "False" if item["answer"] == "True" else "True"
            return eval_harness.ItemResult(
                raw_output=parsed,
                parsed_answer=parsed,
                correct=False,
                invalid_output=False,
            )

        pair_results = eval_hf_causal.evaluate_records(records, run_item)

        self.assertEqual(pair_results[0]["comparison_side"], "invariant")
        self.assertFalse(pair_results[0]["flip_failure"])
        self.assertFalse(pair_results[0]["invariance_failure"])


if __name__ == "__main__":
    unittest.main()
