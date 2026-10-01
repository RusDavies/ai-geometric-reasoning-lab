#!/usr/bin/env python3

import copy
import unittest

from src import generate_grr003


class GenerateGRR003Tests(unittest.TestCase):
    def test_generation_is_deterministic(self) -> None:
        first = generate_grr003.generate_split("validation", 10)
        second = generate_grr003.generate_split("validation", 10)

        self.assertEqual(first, second)

    def test_checker_covers_core_relation_labels(self) -> None:
        program = [
            "point A = (0, 0)",
            "point B = (4, 0)",
            "point C = (2, 0)",
            "line L = line(A, B)",
            "segment S = segment(A, B)",
            "box X = box(0, 0, 4, 4)",
        ]

        checks = [
            ("point_identity", ["A", "A"], "same_point"),
            ("line_membership", ["C", "L"], "on_line"),
            ("line_membership", ["C", "S"], "on_segment"),
            ("containment", ["X", "C"], "contains"),
            ("distance_comparison", ["A", "C", "B"], "closer_to_first"),
            ("horizontal_order", ["A", "B"], "left_of"),
            ("vertical_order", ["A", "B"], "same_y"),
            ("inside", ["C", "X"], "boundary"),
            ("between", ["C", "A", "B"], "between"),
        ]

        for family, query_objects, expected in checks:
            with self.subTest(family=family, query_objects=query_objects):
                label, _ = generate_grr003.evaluate_relation(program, family, query_objects)
                self.assertEqual(label, expected)

    def test_labels_change_and_checker_agrees_for_every_pair(self) -> None:
        records = generate_grr003.generate_split("train", 40)

        for record in records:
            canonical_label, _ = generate_grr003.evaluate_relation(
                record["canonical"]["program"],
                record["relation_family"],
                record["query_objects"],
            )
            perturbed_label, _ = generate_grr003.evaluate_relation(
                record["perturbed"]["program"],
                record["relation_family"],
                record["query_objects"],
            )

            self.assertEqual(canonical_label, record["canonical"]["answer"])
            self.assertEqual(perturbed_label, record["perturbed"]["answer"])
            self.assertNotEqual(canonical_label, perturbed_label)
            self.assertEqual(record["expected_relation"], "label_changes")
            differences = [
                index
                for index, (left, right) in enumerate(
                    zip(record["canonical"]["program"], record["perturbed"]["program"])
                )
                if left != right
            ]
            self.assertEqual(differences, [record["changed_statement_index"]])

    def test_default_split_sizes_generate_and_validate(self) -> None:
        generated = {
            "train": generate_grr003.generate_split("train", 112),
            "validation": generate_grr003.generate_split("validation", 24),
            "test": generate_grr003.generate_split("test", 24),
        }

        self.assertEqual(generate_grr003.validate_splits(generated), [])
        self.assertEqual(len(generated["train"]), 112)
        self.assertEqual(len(generated["validation"]), 24)
        self.assertEqual(len(generated["test"]), 24)

    def test_relation_families_and_answer_labels_are_covered(self) -> None:
        records = generate_grr003.generate_split("train", 72)
        families = {record["relation_family"] for record in records}
        labels = {
            record[side]["answer"]
            for record in records
            for side in ("canonical", "perturbed")
        }
        expected_labels = {
            label
            for family_labels in generate_grr003.RELATION_LABELS.values()
            for label in family_labels
        }

        self.assertEqual(families, set(generate_grr003.RELATION_FAMILIES))
        self.assertEqual(labels, expected_labels)

    def test_split_validation_rejects_object_name_overlap(self) -> None:
        generated = {
            "train": generate_grr003.generate_split("train", 1),
            "validation": generate_grr003.generate_split("validation", 1),
        }
        generated["validation"][0] = copy.deepcopy(generated["train"][0])
        generated["validation"][0]["split"] = "validation"
        generated["validation"][0]["record_id"] = "grr003-validation-999999"

        errors = generate_grr003.validate_splits(generated)

        self.assertTrue(any("object vocabulary overlap" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
