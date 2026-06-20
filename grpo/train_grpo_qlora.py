"""Memory-controlled QLoRA GRPO fallback.

This is a non-Unsloth fallback for cases where Unsloth loads the local Gemma model
near full GPU capacity before GRPO generation begins.

Run from inside grpo/:
    python train_grpo_qlora.py --model_path /local/model --train_jsonl /path/data.jsonl --output_dir ./outputs/grpo_qlora_genui
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from data_utils import load_train_eval_datasets
from rewards import REWARD_FUNCS


@dataclass
class QLoRAGRPOConfig:
    model_path: str
    train_jsonl: str
    output_dir: str
    eval_jsonl: Optional[str] = None
    validation_split: float = 0.05
    seed: int = 42

    max_prompt_length: int = 512
    max_completion_length: int = 512
    num_generations: int = 2
    temperature: float = 0.7
    top_p: float = 0.9
    beta: float = 0.04

    learning_rate: float = 2e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 5
    eval_strategy: str = "no"
    eval_steps: int = 0
    save_steps: int = 500
    save_total_limit: int = 2
    report_to: str = "tensorboard"
    run_name: str = "grpo_qlora_genui"
    resume_from_checkpoint: Optional[str] = None

    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"

    bf16: bool = False
    fp16: bool = True
    tf32: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    gradient_checkpointing: bool = True

    min_free_gpu_gb_after_load: float = 3.0


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> QLoRAGRPOConfig:
    parser = argparse.ArgumentParser()
    for field_name, field_def in QLoRAGRPOConfig.__dataclass_fields__.items():
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
    return QLoRAGRPOConfig(**vars(parser.parse_args()))


def gpu_mem(label: str) -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    free, total = torch.cuda.mem_get_info(0)
    allocated = torch.cuda.memory_allocated(0)
    print(
        f"[gpu memory] {label}: "
        f"free={free / 1024**3:.3f} GB, "
        f"total={total / 1024**3:.3f} GB, "
        f"allocated={allocated / 1024**3:.3f} GB, "
        f"reserved={torch.cuda.memory_reserved(0) / 1024**3:.3f} GB"
    )
    return free / 1024**3, total / 1024**3, allocated / 1024**3


def assert_local_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Local model path missing: {path}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"config.json missing in local model path: {path}")
    return path


def validate_grpo_batch_config(cfg: QLoRAGRPOConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    train_global = cfg.per_device_train_batch_size * world_size
    eval_global = cfg.per_device_eval_batch_size * world_size
    print("[GRPO batch check]")
    print(f"  train batch check = {cfg.per_device_train_batch_size} * {world_size} = {train_global}")
    print(f"  eval batch check  = {cfg.per_device_eval_batch_size} * {world_size} = {eval_global}")
    print(f"  num_generations   = {cfg.num_generations}")
    if cfg.num_generations < 2:
        raise ValueError("GRPO requires num_generations >= 2")
    if train_global % cfg.num_generations != 0:
        raise ValueError("Train global batch must be divisible by num_generations")
    if cfg.eval_strategy != "no" and eval_global % cfg.num_generations != 0:
        raise ValueError("Eval global batch must be divisible by num_generations")


def dtype_from_name(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(cfg: QLoRAGRPOConfig):
    model_path = assert_local_model_path(cfg.model_path)
    gpu_mem("startup")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype_from_name(cfg.bnb_4bit_compute_dtype),
    )

    print(f"[model] Loading 4-bit local model only from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if cfg.fp16 else torch.bfloat16 if cfg.bf16 else torch.float32,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    free_gb, _, _ = gpu_mem("after 4-bit model load")
    if free_gb < cfg.min_free_gpu_gb_after_load:
        raise RuntimeError(
            f"Only {free_gb:.2f} GB GPU memory is free after model load. "
            f"Need at least {cfg.min_free_gpu_gb_after_load:.2f} GB for GRPO generation. "
            "This means the model is too large/not quantized enough for this GPU, or another process is using memory."
        )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
    print("[LoRA] target modules:", target_modules)
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    gpu_mem("after k-bit prep")
    return model, tokenizer, peft_config


def build_grpo_config(cfg: QLoRAGRPOConfig) -> GRPOConfig:
    kwargs = dict(
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
        eval_strategy=cfg.eval_strategy,
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
    if cfg.eval_strategy != "no":
        kwargs["eval_steps"] = cfg.eval_steps

    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    ignored = sorted(set(kwargs) - set(filtered))
    if ignored:
        print("[GRPOConfig] Ignoring unsupported args:", ignored)
    return GRPOConfig(**filtered)


def main() -> None:
    cfg = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch")
    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    validate_grpo_batch_config(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(Path(cfg.output_dir) / "qlora_train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_dataset, eval_dataset = load_train_eval_datasets(
        train_jsonl=cfg.train_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        validation_split=cfg.validation_split,
        seed=cfg.seed,
    )
    if cfg.eval_strategy == "no":
        eval_dataset = None

    model, tokenizer, peft_config = load_model_and_tokenizer(cfg)
    training_args = build_grpo_config(cfg)
    gpu_mem("before trainer init")

    trainer_kwargs = dict(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    sig = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "processing_class" in sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)
    gpu_mem("after trainer init")
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    with open(Path(cfg.output_dir) / "DONE", "w", encoding="utf-8") as f:
        f.write("QLoRA GRPO training completed.\n")


if __name__ == "__main__":
    main()
