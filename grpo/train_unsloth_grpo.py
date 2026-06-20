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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:32")

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

    # Extreme low-memory defaults so direct python execution also smoke-tests safely.
    max_seq_length: int = 256
    max_prompt_length: int = 128
    max_completion_length: int = 128
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
    run_name: str = "unsloth_grpo_genui"
    resume_from_checkpoint: Optional[str] = None

    load_in_4bit: bool = True
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.20

    use_lora: bool = True
    lora_r: int = 2
    lora_alpha: int = 4
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,v_proj"

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


def _gpu_mem(label: str) -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(
            f"[gpu memory] {label}: "
            f"free={free / 1024**3:.3f} GB, total={total / 1024**3:.3f} GB, "
            f"allocated={allocated:.3f} GB, reserved={reserved:.3f} GB"
        )


def normalize_precision(cfg: UnslothGRPOConfig) -> UnslothGRPOConfig:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to PyTorch. Unsloth GRPO is intended for GPU training.")

    print(f"[runtime] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[runtime] torch: {torch.__version__}, cuda build: {torch.version.cuda}")
    _gpu_mem("startup")

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
            f"num_generations={cfg.num_generations}."
        )

    if cfg.eval_strategy != "no" and eval_global_generation_batch % cfg.num_generations != 0:
        raise ValueError(
            "Invalid GRPO eval batch configuration: "
            f"({cfg.per_device_eval_batch_size} * {num_processes}) must be divisible by "
            f"num_generations={cfg.num_generations}."
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


def _accepts_kwarg(callable_obj, key: str) -> bool:
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False
    return key in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _patch_forward_logits_to_keep(model) -> None:
    """Avoid full-vocabulary logits for every prompt token during generation prefill.

    The observed OOM comes from Gemma4 `lm_head(hidden_states[:, slice_indices, :])`.
    During generation, only last-token logits are needed for the next-token decision.
    Some Unsloth/Gemma paths do not pass this optimization automatically, so we add it.
    """
    seen: set[int] = set()

    def patch_one(module, label: str) -> None:
        if module is None or not hasattr(module, "forward") or id(module) in seen:
            return
        seen.add(id(module))

        original_forward = module.forward
        supports_logits_to_keep = _accepts_kwarg(original_forward, "logits_to_keep")
        supports_num_logits_to_keep = _accepts_kwarg(original_forward, "num_logits_to_keep")
        if not supports_logits_to_keep and not supports_num_logits_to_keep:
            return

        def forward_with_limited_logits(self, *args, **kwargs):
            if not self.training:
                if supports_logits_to_keep:
                    kwargs.setdefault("logits_to_keep", 1)
                elif supports_num_logits_to_keep:
                    kwargs.setdefault("num_logits_to_keep", 1)
            return original_forward(*args, **kwargs)

        module.forward = MethodType(forward_with_limited_logits, module)
        print(f"[memory] Patched generation logits_to_keep on {label}")

    candidates = [
        (model, "model"),
        (getattr(model, "base_model", None), "model.base_model"),
        (getattr(getattr(model, "base_model", None), "model", None), "model.base_model.model"),
        (getattr(model, "model", None), "model.model"),
        (getattr(getattr(model, "model", None), "model", None), "model.model.model"),
    ]
    for module, label in candidates:
        patch_one(module, label)


def _strip_unsupported_generate_kwargs(model) -> None:
    """Patch model.generate to ignore leaked multimodal kwargs and request last-token logits."""
    original_generate = model.generate

    def generate_without_unused_kwargs(self, *args, **kwargs):
        removed = []
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in kwargs:
                kwargs.pop(key, None)
                removed.append(key)
        if removed:
            print(f"[generate] Removed unused kwargs: {removed}")

        # Generation only needs next-token logits; full prompt logits can allocate tens of GB.
        kwargs.setdefault("logits_to_keep", 1)
        kwargs.setdefault("output_scores", False)
        kwargs.setdefault("return_dict_in_generate", False)

        try:
            return original_generate(*args, **kwargs)
        except ValueError as exc:
            # If this transformers/Unsloth version rejects logits_to_keep as a generate kwarg,
            # retry without it. The forward patch above may still handle the optimization.
            if "logits_to_keep" in str(exc) and "model_kwargs" in str(exc):
                kwargs.pop("logits_to_keep", None)
                print("[generate] logits_to_keep rejected by generate(); retrying without generate kwarg")
                return original_generate(*args, **kwargs)
            raise

    model.generate = MethodType(generate_without_unused_kwargs, model)

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
            kwargs.setdefault("logits_to_keep", 1)
            kwargs.setdefault("output_scores", False)
            kwargs.setdefault("return_dict_in_generate", False)
            try:
                return original_base_generate(*args, **kwargs)
            except ValueError as exc:
                if "logits_to_keep" in str(exc) and "model_kwargs" in str(exc):
                    kwargs.pop("logits_to_keep", None)
                    print("[base generate] logits_to_keep rejected by generate(); retrying without generate kwarg")
                    return original_base_generate(*args, **kwargs)
                raise

        base_model.generate = MethodType(base_generate_without_unused_kwargs, base_model)


def _print_quantization_report(model) -> None:
    try:
        import bitsandbytes as bnb

        linear4bit = getattr(bnb.nn, "Linear4bit", None)
        if linear4bit is not None:
            n_4bit = sum(1 for module in model.modules() if isinstance(module, linear4bit))
            print(f"[quantization] Linear4bit modules: {n_4bit}")
    except Exception as exc:
        print(f"[quantization] Could not inspect bitsandbytes modules: {exc}")

    print("[quantization] model.is_loaded_in_4bit:", getattr(model, "is_loaded_in_4bit", None))
    print("[quantization] model dtype:", getattr(model, "dtype", None))


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

    supported = set(inspect.signature(FastLanguageModel.from_pretrained).parameters)
    if "local_files_only" in supported:
        kwargs["local_files_only"] = True
    if "trust_remote_code" in supported:
        kwargs["trust_remote_code"] = True

    print(f"[model] Loading local model only from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
    _gpu_mem("after model load")
    _print_quantization_report(model)

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
        _gpu_mem("after LoRA attach")
        _print_quantization_report(model)

    _patch_forward_logits_to_keep(model)
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
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if cfg.eval_strategy != "no" else None,
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
    if cfg.eval_strategy == "no":
        eval_dataset = None

    model, tokenizer = build_model_and_tokenizer(cfg)
    training_args = build_grpo_config(cfg)
    _gpu_mem("before trainer init")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    _gpu_mem("after trainer init")

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(Path(cfg.output_dir) / "DONE", "w", encoding="utf-8") as f:
        f.write("Unsloth GRPO training completed.\n")


if __name__ == "__main__":
    main()
