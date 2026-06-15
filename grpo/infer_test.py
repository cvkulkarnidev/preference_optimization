"""Generate predictions from a trained GRPO model or LoRA adapter.

Writes:
  - predictions.jsonl
  - metrics.json

Expected test JSONL format:
{"response_text": "...", "genui_json": "..."}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import canonical_json, load_jsonl, make_prompt
from rewards import (
    length_sanity_reward,
    no_markdown_or_prose_reward,
    reference_similarity_reward,
    schema_key_overlap_reward,
    valid_json_reward,
    widget_type_match_reward,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, help="Base model path used before LoRA training")
    parser.add_argument("--model_or_adapter_path", required=True, help="Final full model path or LoRA adapter output path")
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--is_lora_adapter", type=str2bool, default=True)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--bf16", type=str2bool, default=True)
    return parser.parse_args()


def parse_json_ok(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def top_level_type(text: str) -> Optional[str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("type", "widget", "name", "component"):
                value = obj.get(key)
                if isinstance(value, str):
                    return value.lower().strip()
    except Exception:
        return None
    return None


def load_model(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_or_adapter_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.is_lora_adapter:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, args.model_or_adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_or_adapter_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_one(model, tokenizer, prompt: str, args) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def compute_rewards(prompt: str, completion: str, reference: str) -> Dict[str, float]:
    prompts = [prompt]
    completions = [completion]
    refs = [reference]
    reward_parts = {
        "valid_json_reward": valid_json_reward(prompts, completions)[0],
        "no_markdown_or_prose_reward": no_markdown_or_prose_reward(prompts, completions)[0],
        "widget_type_match_reward": widget_type_match_reward(prompts, completions, genui_json=refs)[0],
        "schema_key_overlap_reward": schema_key_overlap_reward(prompts, completions, genui_json=refs)[0],
        "reference_similarity_reward": reference_similarity_reward(prompts, completions, genui_json=refs)[0],
        "length_sanity_reward": length_sanity_reward(prompts, completions, genui_json=refs)[0],
    }
    reward_parts["total_reward"] = float(sum(reward_parts.values()))
    return reward_parts


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args)
    rows = load_jsonl(args.test_jsonl)

    pred_path = output_dir / "predictions.jsonl"
    metrics: Dict[str, Any] = {
        "num_samples": 0,
        "valid_json_count": 0,
        "exact_canonical_match_count": 0,
        "widget_type_match_count": 0,
        "total_reward_sum": 0.0,
    }

    with pred_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            response_text = str(row["response_text"]).strip()
            reference = canonical_json(row["genui_json"])
            prompt = make_prompt(response_text)
            prediction = generate_one(model, tokenizer, prompt, args)
            rewards = compute_rewards(prompt, prediction, reference)

            pred_canon = canonical_json(prediction)
            ref_canon = canonical_json(reference)
            valid = parse_json_ok(prediction)
            widget_match = top_level_type(prediction) is not None and top_level_type(prediction) == top_level_type(reference)
            exact_match = valid and pred_canon == ref_canon

            metrics["num_samples"] += 1
            metrics["valid_json_count"] += int(valid)
            metrics["exact_canonical_match_count"] += int(exact_match)
            metrics["widget_type_match_count"] += int(widget_match)
            metrics["total_reward_sum"] += rewards["total_reward"]

            out = {
                "idx": idx,
                "response_text": response_text,
                "reference_genui_json": reference,
                "prediction": prediction,
                "prediction_canonical": pred_canon,
                "valid_json": valid,
                "exact_canonical_match": exact_match,
                "widget_type_match": widget_match,
                "rewards": rewards,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = max(metrics["num_samples"], 1)
    metrics["valid_json_rate"] = metrics["valid_json_count"] / n
    metrics["exact_canonical_match_rate"] = metrics["exact_canonical_match_count"] / n
    metrics["widget_type_match_rate"] = metrics["widget_type_match_count"] / n
    metrics["avg_total_reward"] = metrics["total_reward_sum"] / n

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Predictions written to: {pred_path}")


if __name__ == "__main__":
    main()
