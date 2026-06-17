"""GPU-only entrypoint for GRPO training.

This wrapper prevents silent CPU fallback and also fixes Gemma 4 LoRA target
resolution by selecting only real torch.nn.Linear modules. This avoids PEFT
errors on custom wrappers such as Gemma4ClippableLinear.
"""

from __future__ import annotations

from typing import List

import torch
from peft import LoraConfig

import train_grpo


COMMON_LORA_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def normalize_runtime_config_gpu_only(cfg: train_grpo.ScriptConfig) -> train_grpo.ScriptConfig:
    if cfg.use_cpu:
        raise RuntimeError(
            "This is the GPU-only launcher, but --use_cpu true was passed. "
            "Set USE_CPU=false in grpo/run_grpo.sh or run train_grpo.py directly for CPU mode."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not visible to PyTorch, but GPU training was requested. "
            "Check nvidia-smi, CUDA_VISIBLE_DEVICES, your torch CUDA wheel, and accelerate config."
        )

    print(f"[runtime] Using GPU: {torch.cuda.get_device_name(0)}")

    if cfg.bf16 and not torch.cuda.is_bf16_supported():
        print("[runtime] bf16 is not supported on this GPU. Falling back to fp16.")
        cfg.bf16 = False
        cfg.fp16 = True

    if cfg.bf16 and cfg.fp16:
        print("[runtime] Both bf16 and fp16 are enabled. Using bf16 and disabling fp16.")
        cfg.fp16 = False

    return cfg


def _linear_lora_target_names(model: torch.nn.Module) -> List[str]:
    """Return exact module names for LoRA that are backed by torch.nn.Linear.

    Gemma 4 may expose projection modules through custom wrappers such as
    Gemma4ClippableLinear. PEFT cannot inject LoRA into those wrappers directly.
    This function therefore searches the loaded model and returns only actual
    torch.nn.Linear children whose names belong to common attention/MLP paths.
    """
    exact_names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        name_parts = name.split(".")
        if any(part in COMMON_LORA_SUFFIXES for part in name_parts):
            exact_names.append(name)

    # Fallback: if the custom model hides projection names differently, use all
    # Linear layers except obvious output heads. This is still safer than passing
    # wrapper names that PEFT cannot support.
    if not exact_names:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
                exact_names.append(name)

    # Stable + deduplicated
    return sorted(set(exact_names))


def _configure_finite_safe_generation(model: torch.nn.Module, tokenizer) -> None:
    """Guard generation against NaN/Inf logits before multinomial sampling.

    The CUDA assertion `probability tensor contains inf, nan or element < 0`
    usually happens inside sampling when logits become non-finite, especially
    with fp16 + long context/completion. These generation_config flags add the
    Transformers InfNanRemoveLogitsProcessor and renormalize probabilities.
    """
    if hasattr(model, "generation_config"):
        model.generation_config.remove_invalid_values = True
        model.generation_config.renormalize_logits = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        # Keep sampling enabled for GRPO, but avoid extremely sharp distributions.
        if getattr(model.generation_config, "temperature", None) is not None:
            model.generation_config.temperature = max(float(model.generation_config.temperature), 0.7)
        if getattr(model.generation_config, "top_p", None) is not None:
            model.generation_config.top_p = min(float(model.generation_config.top_p), 0.95)

    print("[generation] Enabled remove_invalid_values=True and renormalize_logits=True")


def build_peft_config_gpu_safe(cfg: train_grpo.ScriptConfig):
    if not cfg.use_lora:
        return None

    # Load a temporary lightweight reference to discover supported module names.
    # train_grpo.main() calls build_peft_config after the actual model is loaded,
    # but the original function only receives cfg. We therefore monkey-patch this
    # using a closure in patched_main below so the real loaded model is used.
    raise RuntimeError("build_peft_config_gpu_safe must be replaced by patched_main closure")


def patched_main() -> None:
    cfg = normalize_runtime_config_gpu_only(train_grpo.parse_args())
    import os
    import json
    from dataclasses import asdict
    from pathlib import Path

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.tf32 and torch.cuda.is_available() and not cfg.use_cpu:
        torch.backends.cuda.matmul.allow_tf32 = True

    with open(Path(cfg.output_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    train_dataset, eval_dataset = train_grpo.load_train_eval_datasets(
        train_jsonl=cfg.train_jsonl,
        eval_jsonl=cfg.eval_jsonl,
        validation_split=cfg.validation_split,
        seed=cfg.seed,
    )

    model, tokenizer = train_grpo.build_model_and_tokenizer(cfg)
    _configure_finite_safe_generation(model, tokenizer)

    peft_config = None
    if cfg.use_lora:
        target_modules = _linear_lora_target_names(model)
        if not target_modules:
            raise RuntimeError(
                "No supported torch.nn.Linear modules were found for LoRA. "
                "Set USE_LORA=false or inspect model.named_modules()."
            )
        print(f"[LoRA] Using {len(target_modules)} supported torch.nn.Linear target modules.")
        print("[LoRA] First targets:", target_modules[:20])

        modules_to_save = None
        if cfg.lora_modules_to_save:
            modules_to_save = [x.strip() for x in cfg.lora_modules_to_save.split(",") if x.strip()]

        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = train_grpo.build_grpo_config(cfg)

    trainer = train_grpo.GRPOTrainer(
        model=model,
        reward_funcs=train_grpo.REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.add_callback(train_grpo.RewardLoggingCallback(cfg.output_dir))
    trainer.add_callback(
        train_grpo.PeriodicPredictionCallback(
            output_dir=cfg.output_dir,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            save_steps=cfg.prediction_save_steps,
            num_samples=cfg.prediction_num_samples,
            max_prompt_length=cfg.max_prompt_length,
            max_new_tokens=cfg.prediction_max_new_tokens,
            do_sample=cfg.prediction_do_sample,
            temperature=cfg.prediction_temperature,
            top_p=cfg.prediction_top_p,
        )
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(Path(cfg.output_dir) / "DONE", "w", encoding="utf-8") as f:
        f.write("GRPO training completed.\n")


if __name__ == "__main__":
    patched_main()
