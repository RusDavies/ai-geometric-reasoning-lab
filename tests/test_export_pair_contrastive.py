#!/usr/bin/env python3

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src import build_distillation_data, export_pair_contrastive, generate_grr001


def accepted_example(record: dict, side: str, target: str | None = None) -> dict:
    args = argparse.Namespace(
        provider="ollama",
        model="llama3:latest",
        ollama_url="http://127.0.0.1:11434/api/generate",
        temperature=0.0,
        seed=1,
        num_predict=8,
    )
    teacher_record = build_distillation_data.teacher_record_from_output(
        record, side, target or record[side]["answer"], args
    )
    example = build_distillation_data.accepted_example(teacher_record)
    if target is not None:
        example["target"] = target
    return example


class ExportPairContrastiveTests(unittest.TestCase):
    def test_builds_eligible_train_pair_with_diagnostic_fields(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        source_records = {record["record_id"]: record}
        accepted = {
            record["record_id"]: {
                "canonical": accepted_example(record, "canonical"),
                "perturbed": accepted_example(record, "perturbed"),
            }
        }

        pairs, diagnostics = export_pair_contrastive.export_pair_contrastive_records(
            source_records, accepted
        )

        self.assertEqual(len(pairs), 1)
        self.assertEqual(diagnostics, [])
        pair = pairs[0]
        self.assertEqual(pair["pair_kind"], "flip")
        self.assertEqual(pair["expected_relation"], "answer_flip")
        self.assertEqual(pair["changed_fact_role"], "load_bearing")
        self.assertEqual(pair["canonical"]["proof_path"], record["proof_path"])
        self.assertNotEqual(pair["canonical"]["target"], pair["perturbed"]["target"])

    def test_preserves_incomplete_pair_diagnostic(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        accepted = {
            record["record_id"]: {
                "canonical": accepted_example(record, "canonical"),
            }
        }

        pairs, diagnostics = export_pair_contrastive.export_pair_contrastive_records(
            {record["record_id"]: record}, accepted
        )

        self.assertEqual(pairs, [])
        self.assertEqual(diagnostics[0]["rejection_reasons"], ["incomplete_pair"])
        self.assertEqual(diagnostics[0]["missing_sides"], ["perturbed"])

    def test_rejects_non_flipping_accepted_targets(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        same_target = record["canonical"]["answer"]
        accepted = {
            record["record_id"]: {
                "canonical": accepted_example(record, "canonical"),
                "perturbed": accepted_example(record, "perturbed", same_target),
            }
        }

        pairs, diagnostics = export_pair_contrastive.export_pair_contrastive_records(
            {record["record_id"]: record}, accepted
        )

        self.assertEqual(pairs, [])
        self.assertIn(
            "perturbed_target_disagrees_with_source_gold",
            diagnostics[0]["rejection_reasons"],
        )
        self.assertIn("accepted_targets_do_not_flip", diagnostics[0]["rejection_reasons"])

    def test_excludes_validation_and_test_records(self) -> None:
        record = generate_grr001.generate_split("validation", 1)[0]
        accepted = {
            record["record_id"]: {
                "canonical": accepted_example(record, "canonical"),
                "perturbed": accepted_example(record, "perturbed"),
            }
        }

        pairs, diagnostics = export_pair_contrastive.export_pair_contrastive_records(
            {record["record_id"]: record}, accepted
        )

        self.assertEqual(pairs, [])
        self.assertEqual(diagnostics[0]["rejection_reasons"], ["non_train_split_excluded"])

    def test_writes_outputs_and_manifest(self) -> None:
        record = generate_grr001.generate_split("train", 1)[0]
        accepted_examples = [
            accepted_example(record, "canonical"),
            accepted_example(record, "perturbed"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.jsonl"
            accepted_path = Path(tmpdir) / "accepted.jsonl"
            output_path = Path(tmpdir) / "pairs.jsonl"
            diagnostics_path = Path(tmpdir) / "diagnostics.jsonl"
            manifest_path = Path(tmpdir) / "manifest.json"
            source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            accepted_path.write_text(
                "".join(json.dumps(example) + "\n" for example in accepted_examples),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                source=[str(source_path)],
                accepted=str(accepted_path),
                output=str(output_path),
                diagnostics=str(diagnostics_path),
                manifest=str(manifest_path),
            )

            manifest = export_pair_contrastive.export_pair_contrastive_data(args)

            self.assertEqual(manifest["counts"]["pair_records"], 1)
            self.assertEqual(manifest["counts"]["diagnostic_records"], 0)
            self.assertTrue(output_path.exists())
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
