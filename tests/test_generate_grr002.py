#!/usr/bin/env python3

import unittest

from src import generate_grr002


class GenerateGRR002Tests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_grr002.generate_split("validation", 10)
        second = generate_grr002.generate_split("validation", 10)

        self.assertEqual(first, second)

    def test_answers_are_invariant_for_every_pair(self) -> None:
        records = generate_grr002.generate_split("train", 20)

        for record in records:
            self.assertEqual(
                record["canonical"]["answer"],
                record["invariant"]["answer"],
                record["record_id"],
            )
            self.assertEqual(record["expected_relation"], "answer_invariant")

    def test_families_are_balanced(self) -> None:
        records = generate_grr002.generate_split("train", 25)
        family_counts = {
            family: sum(1 for record in records if record["invariance_family"] == family)
            for family in generate_grr002.INVARIANCE_FAMILIES
        }

        self.assertLessEqual(max(family_counts.values()) - min(family_counts.values()), 1)

    def test_non_load_bearing_polarity_changes_only_distractor(self) -> None:
        records = generate_grr002.generate_split("test", 5)
        record = records[4]
        self.assertEqual(record["invariance_family"], "non_load_bearing_polarity")
        differences = [
            index
            for index, (left, right) in enumerate(
                zip(record["canonical"]["facts"], record["invariant"]["facts"])
            )
            if left != right
        ]

        self.assertEqual(len(differences), 1)
        self.assertGreaterEqual(differences[0], record["hop_count"])
        self.assertLess(differences[0], len(record["canonical"]["facts"]) - 1)

    def test_split_validation_rejects_leakage(self) -> None:
        generated = {
            "train": generate_grr002.generate_split("train", 10),
            "validation": generate_grr002.generate_split("validation", 10),
            "test": generate_grr002.generate_split("test", 10),
        }

        self.assertEqual(generate_grr002.validate_splits(generated), [])

    def test_default_split_sizes_generate_and_validate(self) -> None:
        generated = {
            "train": generate_grr002.generate_split("train", 70),
            "validation": generate_grr002.generate_split("validation", 15),
            "test": generate_grr002.generate_split("test", 15),
        }

        self.assertEqual(generate_grr002.validate_splits(generated), [])
        self.assertEqual(len(generated["train"]), 70)
        self.assertEqual(len(generated["validation"]), 15)
        self.assertEqual(len(generated["test"]), 15)


if __name__ == "__main__":
    unittest.main()
