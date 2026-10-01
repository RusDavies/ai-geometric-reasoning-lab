#!/usr/bin/env python3
"""CPU-first SmolLM2-135M training smoke loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_examples(
    path: Path, limit: int | None, split: str | None = None
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            example = json.loads(stripped)
            if split is not None and example.get("split") != split:
                continue
            if example.get("target") not in {"True", "False"}:
                raise ValueError(f"{path}:{line_number}: target must be True or False")
            if "prompt" not in example or "record_id" not in example:
                raise ValueError(f"{path}:{line_number}: missing prompt or record_id")
            examples.append(example)
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"{path}: no examples loaded")
    return examples


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ["torch", "transformers"]:
        try:
            module = __import__(name)
        except Exception as exc:
            versions[name] = f"unavailable: {exc.__class__.__name__}: {exc}"
        else:
            versions[name] = getattr(module, "__version__", "installed")
    return versions


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_summary(args: argparse.Namespace, examples: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    return {
        "status": "planned",
        "model_id": args.model_id,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split_filter": args.split,
        "example_count": len(examples),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "device": "cpu",
        "python": sys.version,
        "platform": platform.platform(),
        "dependency_versions": dependency_versions(),
    }


def dry_run(args: argparse.Namespace, examples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = base_summary(args, examples)
    summary["status"] = "dry_run"
    summary["note"] = "Validated data and settings without importing ML training dependencies."
    summary["sample_record_ids"] = [example["record_id"] for example in examples[:5]]
    return summary


def train(args: argparse.Namespace, examples: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "training requires torch and transformers; run --dry-run for preflight"
        ) from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to("cpu")
    model.train()

    class PromptDataset(Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> dict[str, Any]:
            example = examples[index]
            prompt = example["prompt"]
            target = example["target"]
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full = tokenizer(
                prompt + target,
                truncation=True,
                max_length=args.max_length,
                padding="max_length",
            )
            labels = list(full["input_ids"])
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            labels = [
                label if mask else -100
                for label, mask in zip(labels, full["attention_mask"])
            ]
            return {
                "input_ids": torch.tensor(full["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(full["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    loader = DataLoader(PromptDataset(), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[float] = []
    started = time.time()
    step = 0

    while step < args.max_steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
            step += 1
            if step >= args.max_steps:
                break

    output_dir = Path(args.output_dir)
    saved_model = False
    if args.save_model:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        saved_model = True

    summary = base_summary(args, examples)
    summary.update(
        {
            "status": "trained",
            "steps_completed": step,
            "losses": losses,
            "elapsed_seconds": round(time.time() - started, 3),
            "output_dir": str(output_dir),
            "saved_model": saved_model,
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/grr001/grr001-train-prompts.jsonl")
    parser.add_argument("--output", default="training-runs/smollm2-135m-smoke.json")
    parser.add_argument("--output-dir", default="artifacts/smollm2-135m-smoke")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--split")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    try:
        examples = load_examples(Path(args.dataset), args.limit, args.split)
        summary = dry_run(args, examples) if args.dry_run else train(args, examples)
    except Exception as exc:
        print(f"training smoke failed: {exc}", file=sys.stderr)
        return 1

    write_summary(Path(args.output), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
