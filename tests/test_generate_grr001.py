#!/usr/bin/env python3

import unittest

from src import generate_grr001


class GenerateGRR001Tests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_grr001.generate_split("validation", 5)
        second = generate_grr001.generate_split("validation", 5)

        self.assertEqual(first, second)

    def test_answers_flip_for_every_pair(self) -> None:
        records = generate_grr001.generate_split("train", 20)

        for record in records:
            self.assertNotEqual(
                record["canonical"]["answer"],
                record["perturbed"]["answer"],
                record["record_id"],
            )

    def test_pairs_differ_by_one_target_fact(self) -> None:
        records = generate_grr001.generate_split("train", 20)

        for record in records:
            canonical = record["canonical"]["facts"]
            perturbed = record["perturbed"]["facts"]
            differences = [
                index
                for index, (left, right) in enumerate(zip(canonical, perturbed))
                if left != right
            ]
            self.assertEqual(differences, [record["load_bearing_fact_index"]])

    def test_split_validation_rejects_leakage(self) -> None:
        generated = {
            "train": generate_grr001.generate_split("train", 5),
            "validation": generate_grr001.generate_split("validation", 5),
            "test": generate_grr001.generate_split("test", 5),
        }

        self.assertEqual(generate_grr001.validate_splits(generated), [])

    def test_default_split_sizes_generate_and_validate(self) -> None:
        generated = {
            "train": generate_grr001.generate_split("train", 70),
            "validation": generate_grr001.generate_split("validation", 15),
            "test": generate_grr001.generate_split("test", 15),
        }

        self.assertEqual(generate_grr001.validate_splits(generated), [])
        self.assertEqual(len(generated["train"]), 70)
        self.assertEqual(len(generated["validation"]), 15)
        self.assertEqual(len(generated["test"]), 15)

    def test_difficulty_eight_has_negation_distractor(self) -> None:
        records = generate_grr001.generate_split("test", 8)
        level_eight = records[7]
        target_fact_index = level_eight["load_bearing_fact_index"]
        distractors = [
            fact
            for index, fact in enumerate(level_eight["canonical"]["facts"])
            if index != target_fact_index and " not " in f" {fact} "
        ]

        self.assertTrue(distractors)


if __name__ == "__main__":
    unittest.main()
