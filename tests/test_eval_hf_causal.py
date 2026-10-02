#!/usr/bin/env python3

import unittest

from src import eval_hf_causal, eval_harness, generate_grr001, generate_grr002, generate_grr003


class EvalHFCausalTests(unittest.TestCase):
    def test_select_answer_from_scores_chooses_highest_score(self) -> None:
        answer = eval_hf_causal.select_answer_from_scores({"True": -2.0, "False": -0.5})

        self.assertEqual(answer, "False")

    def test_choice_margin_reports_runner_up_gap(self) -> None:
        runner_up, margin = eval_hf_causal.choice_runner_up_and_margin(
            {"True": -2.0, "False": -0.5, "Maybe": -1.25},
            "False",
        )

        self.assertEqual(runner_up, "Maybe")
        self.assertAlmostEqual(margin, 0.75)

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

    def test_evaluate_records_handles_grr003_label_changes(self) -> None:
        records = generate_grr003.generate_split("validation", 1)
        seen: dict[str, object] = {}

        def run_item(item: dict) -> eval_harness.ItemResult:
            seen["context"] = item.get("_pair_context")
            seen["side"] = item.get("_side")
            return eval_harness.ItemResult(
                raw_output=item["answer"],
                parsed_answer=item["answer"],
                correct=True,
                invalid_output=False,
            )

        pair_results = eval_hf_causal.evaluate_records(records, run_item)

        self.assertEqual(pair_results[0]["expected_relation"], "label_changes")
        self.assertEqual(pair_results[0]["relation_family"], "point_identity")
        self.assertIs(seen["context"], records[0])
        self.assertEqual(seen["side"], "perturbed")
        self.assertTrue(pair_results[0]["both_correct"])
        self.assertFalse(pair_results[0]["flip_failure"])

    def test_changed_statement_prompt_includes_pair_context(self) -> None:
        record = generate_grr003.generate_split("validation", 1)[0]
        item = dict(record["canonical"])
        item["_pair_context"] = record
        item["_side"] = "canonical"

        prompt = eval_hf_causal.changed_statement_prompt(item)

        self.assertIn("Pair context:", prompt)
        self.assertIn("Scoring side: canonical", prompt)
        self.assertIn(f"Relation family: {record['relation_family']}", prompt)
        self.assertIn("Canonical changed statement:", prompt)
        self.assertIn("Perturbed changed statement:", prompt)

    def test_hf_choice_item_uses_item_allowed_labels(self) -> None:
        item = {
            "program": ["point A = (0, 0)", "point B = (1, 0)"],
            "query": "What is the relation for horizontal_order(A, B)?",
            "allowed_labels": ["left_of", "right_of", "same_x"],
            "answer": "right_of",
        }
        seen: dict[str, list[str]] = {}
        original = eval_hf_causal.candidate_logprobs

        def fake_candidate_logprobs(prompt, model, tokenizer, candidates):
            seen["candidates"] = candidates
            return {"left_of": -3.0, "right_of": -0.1, "same_x": -2.0}

        try:
            eval_hf_causal.candidate_logprobs = fake_candidate_logprobs
            result = eval_hf_causal.run_hf_choice_item(item, object(), object(), object())
        finally:
            eval_hf_causal.candidate_logprobs = original

        self.assertEqual(seen["candidates"], item["allowed_labels"])
        self.assertEqual(result.parsed_answer, "right_of")
        self.assertTrue(result.correct)
        self.assertEqual(result.choice_scores, {"left_of": -3.0, "right_of": -0.1, "same_x": -2.0})
        self.assertEqual(result.choice_runner_up, "same_x")
        self.assertAlmostEqual(result.choice_margin or 0.0, 1.9)


if __name__ == "__main__":
    unittest.main()
