#!/usr/bin/env python3
"""Evaluate GRR-001/GRR-002/GRR-003 pairs against a Hugging Face causal language model."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from .eval_harness import (
        PROMPT_TEMPLATE,
        ItemResult,
        build_prompt,
        comparison_side_key,
        is_label_change_expected,
        item_contains_negation,
        item_allowed_labels,
        load_jsonl,
        parse_answer,
        parse_true_false,
        result_answer_changed,
        summarize,
    )
    from .export_grr001_training import sha256_file
except ImportError:
    from eval_harness import (  # type: ignore
        PROMPT_TEMPLATE,
        ItemResult,
        build_prompt,
        comparison_side_key,
        is_label_change_expected,
        item_contains_negation,
        item_allowed_labels,
        load_jsonl,
        parse_answer,
        parse_true_false,
        result_answer_changed,
        summarize,
    )
    from export_grr001_training import sha256_file  # type: ignore


RunItem = Callable[[dict[str, Any]], ItemResult]


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


def evaluate_records(records: list[dict[str, Any]], run_item: RunItem) -> list[dict[str, Any]]:
    pair_results: list[dict[str, Any]] = []
    for record in records:
        canonical = run_item(record["canonical"])
        comparison_side = comparison_side_key(record)
        comparison = run_item(record[comparison_side])
        expected_relation = record.get("expected_relation", "answer_flip")
        both_correct = canonical.correct and comparison.correct
        both_wrong = not canonical.correct and not comparison.correct
        invalid_output = canonical.invalid_output or comparison.invalid_output
        same_answer_when_gold_flips = (
            is_label_change_expected(expected_relation)
            and canonical.parsed_answer is not None
            and canonical.parsed_answer == comparison.parsed_answer
        )
        answer_changed_when_gold_invariant = (
            expected_relation == "answer_invariant"
            and result_answer_changed(canonical.__dict__, comparison.__dict__)
        )
        exactly_one_correct = canonical.correct != comparison.correct
        flip_failure = is_label_change_expected(expected_relation) and (
            exactly_one_correct or same_answer_when_gold_flips
        )
        invariance_failure = expected_relation == "answer_invariant" and (
            (canonical.correct and not comparison.correct) or answer_changed_when_gold_invariant
        )
        pair_results.append(
            {
                "record_id": record["record_id"],
                "rail_id": record["rail_id"],
                "difficulty": record["difficulty"],
                "hop_count": record.get("hop_count"),
                "distractor_count": record["distractor_count"],
                "pair_kind": record.get("pair_kind"),
                "expected_relation": expected_relation,
                "comparison_side": comparison_side,
                "relation_family": record.get("relation_family"),
                "invariance_family": record.get("invariance_family"),
                "negation_involved": item_contains_negation(record["canonical"])
                or item_contains_negation(record[comparison_side]),
                "canonical": canonical.__dict__,
                comparison_side: comparison.__dict__,
                "comparison": comparison.__dict__,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "exactly_one_correct": exactly_one_correct,
                "same_answer_when_gold_flips": same_answer_when_gold_flips,
                "answer_changed_when_gold_invariant": answer_changed_when_gold_invariant,
                "flip_failure": flip_failure,
                "invariance_failure": invariance_failure,
                "invalid_output": invalid_output,
            }
        )
    return pair_results


def select_answer_from_scores(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def candidate_logprobs(
    prompt: str, model: Any, tokenizer: Any, candidates: list[str]
) -> dict[str, float]:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    scores: dict[str, float] = {}
    for candidate in candidates:
        candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=input_ids[:, :-1])
        logits = output.logits[0]
        score = 0.0
        for offset, token_id in enumerate(candidate_ids):
            prediction_index = len(prompt_ids) + offset - 1
            token_logprobs = torch.log_softmax(logits[prediction_index], dim=-1)
            score += float(token_logprobs[token_id].detach().cpu())
        scores[candidate] = score
    return scores


def run_hf_choice_item(
    item: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace
) -> ItemResult:
    prompt = build_prompt(item)
    scores = candidate_logprobs(prompt, model, tokenizer, item_allowed_labels(item))
    answer = select_answer_from_scores(scores)
    return ItemResult(
        raw_output=answer,
        parsed_answer=answer,
        correct=answer == item["answer"],
        invalid_output=False,
    )


def run_hf_item(item: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace) -> ItemResult:
    import torch

    prompt = build_prompt(item)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][encoded["input_ids"].shape[-1] :]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed = parse_answer(raw_output, item_allowed_labels(item))
    return ItemResult(
        raw_output=raw_output,
        parsed_answer=parsed,
        correct=parsed == item["answer"],
        invalid_output=parsed is None,
    )


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("HF eval requires torch and transformers") from exc

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.to("cpu")
    model.eval()

    dataset_path = Path(args.dataset)
    records = load_jsonl(dataset_path, args.limit)
    if args.answer_mode == "choice":
        run_item = lambda item: run_hf_choice_item(item, model, tokenizer, args)
    else:
        run_item = lambda item: run_hf_item(item, model, tokenizer, args)
    pair_results = evaluate_records(
        records, run_item
    )
    return {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model_id,
            "provider": "huggingface_causal_lm",
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "prompt_template": "dynamic: facts prompt or scene-program allowed-label prompt",
            "generation_options": {
                "answer_mode": args.answer_mode,
                "do_sample": False,
                "max_new_tokens": args.max_new_tokens,
                "max_length": args.max_length,
                "seed": args.seed,
            },
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_versions": dependency_versions(),
        },
        "summary": summarize(pair_results, args),
        "pairs": pair_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/grr001/grr001-validation.jsonl")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output", default="evals/results/grr001-validation-hf-smollm2-135m.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--answer-mode", choices=["generate", "choice"], default="generate")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1001)
    args = parser.parse_args()

    try:
        result = run_eval(args)
    except Exception as exc:
        print(f"HF eval failed: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
