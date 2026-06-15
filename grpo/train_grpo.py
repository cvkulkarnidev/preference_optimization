"""Train a causal LM with GRPO for response_text -> genui_json generation.

Example:
    accelerate launch grpo/train_grpo.py \
      --model_path /home/c.kulkarni/hf_models/google/gemma-4-E2B-it \
      --train_jsonl /path/to/train.jsonl \
      --output_dir ./outputs/grpo_genui \
      --use_lora true \
      --load_in_4bit false
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:
    raise ImportError("Please install TRL: pip install trl") from exc

from data_utils import load_train_eval_datasets
from rewards import REWARD_FUNCS


@dataclass
class ScriptConfig:
    model_path: str
    train_jsonl: str
    output_dir: str
    eval_jsonl: Optional[str] = None
    validation_split: float = 0.05
    seed: int = 42

    max_prompt_length: int = 1024
    max_completion_length: int = 512
    num_generations: int = 4

    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3

    temperature: float = 0.9
    top_p: float = 0.95
    beta: float = 0.04

    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    lora_modules_to_save: Optional[str] = None
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"

    report_to: str = "tensorboard"
    run_name: str = "grpo_genui"
    resume_from_checkpoint: Optional[str] = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> ScriptConfig:
    parser = argparse.ArgumentParser()
    for field_name, field_def in ScriptConfig.__dataclass_fields__.items():
        default = field_def.default
        arg_name = f"--{field_name}"
        if isinstance(default, bool):
            parser.add_argument(arg_name, type=str2bool, default=default)
        elif isinstance(default, int):
            parser.add_argument(arg_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(arg_name, type=float, default=default)
        else:
            parser.add_argument(arg_name, default=default)

    args = parser.parse_args()
    missing = [name for name in ("model_path", "train_jsonl", "output_dir") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")
    return ScriptConfig(**vars(args))


class RewardLoggingCallback(TrainerCallback):
    """Logs reward-related metrics that TRL exposes in `logs`.

    TRL already logs loss, grad_norm, reward, reward_std, KL, completion length,
    etc. This callback keeps a small JSONL history as a backup next to TensorBoard.
    """

    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "reward_logs.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        row = {"step": int(state.global_step)}
        for key, value in logs.items():
            try:
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().item()
                if isinstance(value, (int, float, str, bool)) or value is None:
                    row[key] = value
            except Exception:
                continue
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_model_and_tokenizer(cfg: ScriptConfig):
    torch_dtype = torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else torch.float32)

    quantization_config = None
    if cfg.load_in_4bit:
        compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype if not cfg.load_in_4bit else None,
        quantization_config=quantization_config,
        device_map=None,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False if cfg.gradient_checkpointing else True

    return model, tokenizer


def build_peft_config(cfg: ScriptConfig):
    if not cfg.use_lora:
        return None

    from peft import LoraConfig

    target_modules = [x.strip() for x in cfg.lora_target_modules.split(",") if x.strip()]
    modules_to_save = None
    if cfg.lora_modules_to_save:
        modules_to_save = [x.strip() for x in cfg.lora_modules_to_save.split(",") if x.strip()]

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        task_type="CAUSAL_LM",
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    with open(Path(cfg.output_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_dataset, eval_dataset = load_train_eval_datasets(
        train_jsonl=cfg.train_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        validation_split=cfg.validation_split,
        seed=cfg.seed,
    )

    model, tokenizer = build_model_and_tokenizer(cfg)
    peft_config = build_peft_config(cfg)

    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        report_to=[x.strip() for x in cfg.report_to.split(",") if x.strip()],
        logging_dir=str(Path(cfg.output_dir) / "logs"),
        seed=cfg.seed,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        beta=cfg.beta,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        remove_unused_columns=False,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(RewardLoggingCallback(cfg.output_dir))

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Save final adapter/full model path marker for downstream inference scripts.
    with open(Path(cfg.output_dir) / "DONE", "w", encoding="utf-8") as f:
        f.write("GRPO training completed.\n")


if __name__ == "__main__":
    main()
