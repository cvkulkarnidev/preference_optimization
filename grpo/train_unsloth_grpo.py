"""Unsloth-based GRPO training for GenUI JSON generation.

Run from inside the grpo/ folder:
    python train_unsloth_grpo.py --model_path /path/to/model --train_jsonl /path/to/data.jsonl --output_dir ./outputs/unsloth_grpo_genui

Expected JSONL keys:
    response_text: prompt/input text
    genui_json: target JSON string/object
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from unsloth import FastLanguageModel, PatchFastRL

# Patch TRL GRPO before importing GRPOTrainer/GRPOConfig.
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from data_utils import load_train_eval_datasets
from rewards import REWARD_FUNCS


@dataclass
class UnslothGRPOConfig:
    model_path: str
    train_jsonl: str
    output_dir: str
    eval_jsonl: Optional[str] = None
    validation_split: float = 0.05
    seed: int = 42

    max_seq_length: int = 8192
    max_prompt_length: int = 4096
    max_completion_length: int = 4096
    num_generations: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.04

    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    report_to: str = "tensorboard"
    run_name: str = "unsloth_grpo_genui"
    resume_from_checkpoint: Optional[str] = None

    load_in_4bit: bool = True
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.75

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    bf16: bool = False
    fp16: bool = True
    tf32: bool = True


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> UnslothGRPOConfig:
    parser = argparse.ArgumentParser()
    for field_name, field_def in UnslothGRPOConfig.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name}"
        required = default.__class__.__name__ == "_MISSING_TYPE"

        if required:
            parser.add_argument(arg_name, required=True)
        elif isinstance(default, bool):
            parser.add_argument(arg_name, type=str2bool, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg_name, type=float, default=default)
        else:
            parser.add_argument(arg_name, default=default)

    return UnslothGRPOConfig(**vars(parser.parse_args()))


def normalize_precision(cfg: UnslothGRPOConfig) -> UnslothGRPOConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch. Unsloth GRPO is intended for GPU training.")

    print(f"[runtime] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[runtime] torch: {torch.__version__}, cuda build: {torch.version.cuda}")

    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.bf16 and not torch.cuda.is_bf16_supported():
        print("[runtime] bf16 is not supported. Switching to fp16.")
        cfg.bf16 = False
        cfg.fp16 = True

    if cfg.bf16 and cfg.fp16:
        print("[runtime] Both bf16 and fp16 enabled. Keeping bf16, disabling fp16.")
        cfg.fp16 = False

    return cfg


def build_model_and_tokenizer(cfg: UnslothGRPOConfig):
    dtype = None
    if cfg.bf16:
        dtype = torch.bfloat16
    elif cfg.fp16:
        dtype = torch.float16

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_path,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=dtype,
        fast_inference=cfg.fast_inference,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if hasattr(model, "generation_config"):
        model.generation_config.remove_invalid_values = True
        model.generation_config.renormalize_logits = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    if cfg.use_lora:
        target_modules = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
        print("[LoRA] target modules:", target_modules)
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )

    return model, tokenizer


def build_grpo_config(cfg: UnslothGRPOConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        report_to=cfg.report_to,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,
        remove_unused_columns=False,
    )


def main() -> None:
    cfg = normalize_precision(parse_args())
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open(Path(cfg.output_dir) / "unsloth_train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_dataset, eval_dataset = load_train_eval_datasets(
        train_jsonl=cfg.train_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        validation_split=cfg.validation_split,
        seed=cfg.seed,
    )

    model, tokenizer = build_model_and_tokenizer(cfg)
    training_args = build_grpo_config(cfg)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(Path(cfg.output_dir) / "DONE", "w", encoding="utf-8") as f:
        f.write("Unsloth GRPO training completed.\n")


if __name__ == "__main__":
    main()
