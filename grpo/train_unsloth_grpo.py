"""Unsloth-based GRPO training for GenUI JSON generation.

Run from inside the grpo/ folder:
    python train_unsloth_grpo.py --model_path /path/to/model --train_jsonl /path/to/data.jsonl --output_dir ./outputs/unsloth_grpo_genui

Expected JSONL keys:
    response_text: prompt/input text
    genui_json: target JSON string/object
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Optional

# Prevent accidental hub downloads. The model path must be local.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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

    max_seq_length: int = 10240
    max_prompt_length: int = 2048
    max_completion_length: int = 8192
    num_generations: int = 2
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.04

    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
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
        required = default is MISSING

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


def validate_grpo_batch_config(cfg: UnslothGRPOConfig) -> None:
    """Fail early with the exact values TRL will check."""
    num_processes = int(os.environ.get("WORLD_SIZE", "1"))
    train_global_generation_batch = cfg.per_device_train_batch_size * num_processes
    eval_global_generation_batch = cfg.per_device_eval_batch_size * num_processes

    print("[GRPO batch check]")
    print(f"  per_device_train_batch_size = {cfg.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size  = {cfg.per_device_eval_batch_size}")
    print(f"  num_processes / WORLD_SIZE  = {num_processes}")
    print(f"  num_generations             = {cfg.num_generations}")
    print(f"  checked train global batch   = {train_global_generation_batch}")
    print(f"  checked eval global batch    = {eval_global_generation_batch}")

    if cfg.num_generations < 2:
        raise ValueError("GRPO requires num_generations >= 2.")

    if train_global_generation_batch % cfg.num_generations != 0:
        raise ValueError(
            "Invalid GRPO train batch configuration: "
            f"({cfg.per_device_train_batch_size} * {num_processes}) must be divisible by "
            f"num_generations={cfg.num_generations}. "
            "Increase PER_DEVICE_TRAIN_BATCH_SIZE or reduce NUM_GENERATIONS."
        )

    if eval_global_generation_batch % cfg.num_generations != 0:
        raise ValueError(
            "Invalid GRPO eval batch configuration: "
            f"({cfg.per_device_eval_batch_size} * {num_processes}) must be divisible by "
            f"num_generations={cfg.num_generations}. "
            "Increase PER_DEVICE_EVAL_BATCH_SIZE or reduce NUM_GENERATIONS."
        )


def _assert_local_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Model path does not exist locally: {path}\n"
            "Refusing to download any model from Hugging Face. Fix MODEL_PATH in run_unsloth_grpo.sh."
        )

    expected_any = ["config.json", "tokenizer.json", "tokenizer.model", "model.safetensors.index.json"]
    if not any((path / name).exists() for name in expected_any):
        raise FileNotFoundError(
            f"The local model directory does not look complete: {path}\n"
            f"Expected at least one of: {expected_any}\n"
            "Refusing to download any fallback model."
        )
    return path


def _strip_unsupported_generate_kwargs(model) -> None:
    """Patch model.generate to ignore multimodal kwargs leaked by Unsloth/Gemma processors.

    Some Gemma/Unsloth paths pass `mm_token_type_ids` during generation. Text-only
    causal models reject it inside transformers' `_validate_model_kwargs`. We strip it
    at the outer model.generate boundary so GRPO generation can continue.
    """
    original_generate = model.generate

    def generate_without_unused_kwargs(self, *args, **kwargs):
        removed = []
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in kwargs:
                kwargs.pop(key, None)
                removed.append(key)
        if removed:
            print(f"[generate] Removed unused kwargs: {removed}")
        return original_generate(*args, **kwargs)

    model.generate = MethodType(generate_without_unused_kwargs, model)

    # PEFT sometimes delegates generation to the base model. Patch that too if present.
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "generate"):
        original_base_generate = base_model.generate

        def base_generate_without_unused_kwargs(self, *args, **kwargs):
            removed = []
            for key in ("mm_token_type_ids", "token_type_ids"):
                if key in kwargs:
                    kwargs.pop(key, None)
                    removed.append(key)
            if removed:
                print(f"[base generate] Removed unused kwargs: {removed}")
            return original_base_generate(*args, **kwargs)

        base_model.generate = MethodType(base_generate_without_unused_kwargs, base_model)


def build_model_and_tokenizer(cfg: UnslothGRPOConfig):
    model_path = _assert_local_model_path(cfg.model_path)

    dtype = None
    if cfg.bf16:
        dtype = torch.bfloat16
    elif cfg.fp16:
        dtype = torch.float16

    kwargs = dict(
        model_name=str(model_path),
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=dtype,
        fast_inference=cfg.fast_inference,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )

    # Use local-only loading when supported by the installed Unsloth version.
    supported = set(inspect.signature(FastLanguageModel.from_pretrained).parameters)
    for key in ("local_files_only", "trust_remote_code"):
        if key == "local_files_only" and key in supported:
            kwargs[key] = True
        if key == "trust_remote_code" and key in supported:
            kwargs[key] = True

    print(f"[model] Loading local model only from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)

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

    _strip_unsupported_generate_kwargs(model)
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
    validate_grpo_batch_config(cfg)
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
