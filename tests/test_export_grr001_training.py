#!/usr/bin/env python3

import unittest

from src import export_grr001_training, generate_grr001


class ExportGRR001TrainingTests(unittest.TestCase):
    def test_export_creates_two_examples_per_record(self) -> None:
        records = generate_grr001.generate_split("train", 3)
        examples = export_grr001_training.export_examples(records)

        self.assertEqual(len(examples), 6)
        self.assertEqual({example["side"] for example in examples}, {"canonical", "perturbed"})
        self.assertEqual({example["target"] for example in examples}, {"True", "False"})

    def test_export_preserves_pair_metadata(self) -> None:
        record = generate_grr001.generate_split("validation", 1)[0]
        examples = export_grr001_training.export_examples([record])

        for example in examples:
            self.assertEqual(example["record_id"], record["record_id"])
            self.assertEqual(example["difficulty"], record["difficulty"])
            self.assertIn("Question:", example["prompt"])


if __name__ == "__main__":
    unittest.main()
