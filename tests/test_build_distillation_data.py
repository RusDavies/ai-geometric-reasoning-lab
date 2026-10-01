#!/usr/bin/env python3

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src import build_distillation_data, generate_grr001


def args_for_outputs(tmpdir: str) -> argparse.Namespace:
    return argparse.Namespace(
        provider="ollama",
        model="llama3:latest",
        ollama_url="http://127.0.0.1:11434/api/generate",
        temperature=0.0,
        seed=1,
        num_predict=8,
        input=[],
        raw_output=str(Path(tmpdir) / "raw.jsonl"),
        accepted_output=str(Path(tmpdir) / "accepted.jsonl"),
        rejected_output=str(Path(tmpdir) / "rejected.jsonl"),
        manifest=str(Path(tmpdir) / "manifest.json"),
        fixture_outputs=None,
        limit_per_file=None,
        progress_every=10,
    )


class BuildDistillationDataTests(unittest.TestCase):
    def test_teacher_record_preserves_raw_output_and_rejection_reason(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        args = args_for_outputs("/tmp")

        teacher_record = build_distillation_data.teacher_record_from_output(
            record, "canonical", "Maybe", args
        )

        self.assertEqual(teacher_record["teacher"]["raw_output"], "Maybe")
        self.assertIsNone(teacher_record["teacher"]["parsed_answer"])
        self.assertEqual(
            teacher_record["rejection_reasons"],
            ["invalid_true_false_output", "teacher_answer_disagrees_with_gold"],
        )

    def test_split_teacher_records_accepts_only_correct_parseable_answers(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        args = args_for_outputs("/tmp")
        accepted_record = build_distillation_data.teacher_record_from_output(
            record, "canonical", record["canonical"]["answer"], args
        )
        rejected_record = build_distillation_data.teacher_record_from_output(
            record, "perturbed", record["canonical"]["answer"], args
        )

        accepted, rejected = build_distillation_data.split_teacher_records(
            [accepted_record, rejected_record]
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["target"], record["canonical"]["answer"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(
            rejected[0]["rejection_reasons"], ["teacher_answer_disagrees_with_gold"]
        )

    def test_build_distillation_data_writes_outputs_and_manifest(self) -> None:
        records = generate_grr001.generate_split("validation", 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.jsonl"
            source.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            args = args_for_outputs(tmpdir)
            args.input = [str(source)]
            fixture = Path(tmpdir) / "fixture.jsonl"
            fixture_records = []
            for record in records:
                for side in ("canonical", "perturbed"):
                    fixture_records.append(
                        {
                            "record_id": record["record_id"],
                            "side": side,
                            "raw_output": record[side]["answer"],
                        }
                    )
            fixture.write_text(
                "".join(json.dumps(record) + "\n" for record in fixture_records),
                encoding="utf-8",
            )
            args.fixture_outputs = str(fixture)

            manifest = build_distillation_data.build_distillation_data(args)

            self.assertEqual(manifest["counts"]["source_records"], 2)
            self.assertEqual(manifest["counts"]["accepted_examples"], 4)
            self.assertEqual(manifest["counts"]["rejected_diagnostics"], 0)
            self.assertTrue(Path(args.raw_output).exists())
            self.assertTrue(Path(args.accepted_output).exists())


if __name__ == "__main__":
    unittest.main()
