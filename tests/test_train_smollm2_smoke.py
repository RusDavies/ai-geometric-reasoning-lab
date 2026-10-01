#!/usr/bin/env python3

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src import train_smollm2_smoke


class TrainSmolLM2SmokeTests(unittest.TestCase):
    def test_dry_run_validates_examples_and_records_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "examples.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "grr001-train-000001",
                        "split": "train",
                        "prompt": "Question:\nTrue or false: Foo is bar.\n",
                        "target": "True",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                dataset=str(dataset),
                model_id="HuggingFaceTB/SmolLM2-135M",
                max_steps=5,
                batch_size=1,
                learning_rate=1e-4,
                max_length=512,
                seed=1,
                split="train",
            )
            examples = train_smollm2_smoke.load_examples(dataset, limit=None, split="train")
            summary = train_smollm2_smoke.dry_run(args, examples)

        self.assertEqual(summary["status"], "dry_run")
        self.assertEqual(summary["example_count"], 1)
        self.assertEqual(summary["model_id"], "HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(summary["split_filter"], "train")

    def test_split_filter_skips_other_splits_before_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "examples.jsonl"
            dataset.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_id": "grr001-validation-000001",
                                "split": "validation",
                                "prompt": "Question:\nTrue or false: Foo is bar.\n",
                                "target": "True",
                            }
                        ),
                        json.dumps(
                            {
                                "record_id": "grr001-train-000001",
                                "split": "train",
                                "prompt": "Question:\nTrue or false: Baz is qux.\n",
                                "target": "False",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            examples = train_smollm2_smoke.load_examples(dataset, limit=1, split="train")

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["record_id"], "grr001-train-000001")

    def test_rejects_invalid_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "examples.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "record_id": "grr001-train-000001",
                        "prompt": "Question:\nTrue or false: Foo is bar.\n",
                        "target": "Maybe",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                train_smollm2_smoke.load_examples(dataset, limit=None)


if __name__ == "__main__":
    unittest.main()
