"""Train a causal LM with GRPO for response_text -> genui_json generation."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:
    raise ImportError("Please install TRL: pip install trl") from exc

from data_utils import canonical_json, load_train_eval_datasets
from rewards import (
    REWARD_FUNCS,
    length_sanity_reward,
    no_markdown_or_prose_reward,
    reference_similarity_reward,
    schema_key_overlap_reward,
    valid_json_reward,
    widget_type_match_reward,
)


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
    prediction_save_steps: int = 100
    prediction_num_samples: int = 16
    prediction_max_new_tokens: int = 512
    prediction_do_sample: bool = False
    prediction_temperature: float = 0.0
    prediction_top_p: float = 1.0
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

    args = parser.parse_args()
    return ScriptConfig(**vars(args))


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_json_ok(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _top_level_type(text: str) -> Optional[str]:
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


def _compute_reward_parts(prompt: str, completion: str, reference: str) -> Dict[str, float]:
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


class RewardLoggingCallback(TrainerCallback):
    """Keep a JSONL backup of trainer logs next to TensorBoard logs."""

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


class PeriodicPredictionCallback(TrainerCallback):
    """Save eval-set predictions every N optimizer steps during training."""

    def __init__(
        self,
        output_dir: str,
        eval_dataset,
        tokenizer,
        save_steps: int,
        num_samples: int,
        max_prompt_length: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.save_steps = save_steps
        self.num_samples = num_samples
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self._last_saved_step = -1

    def on_step_end(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if self.save_steps <= 0 or self.num_samples <= 0:
            return
        if state.global_step <= 0 or state.global_step % self.save_steps != 0:
            return
        if state.global_step == self._last_saved_step:
            return
        if getattr(args, "process_index", 0) != 0:
            return
        if model is None:
            return

        self._last_saved_step = int(state.global_step)
        self._save_predictions(model=model, step=int(state.global_step))

    def _generate_one(self, model, prompt: str) -> str:
        device = _model_device(model)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _save_predictions(self, model, step: int) -> None:
        was_training = model.training
        model.eval()

        step_dir = self.output_dir / "predictions" / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        pred_path = step_dir / "predictions.jsonl"
        metrics_path = step_dir / "metrics.json"

        metrics: Dict[str, Any] = {
            "step": step,
            "num_samples": 0,
            "valid_json_count": 0,
            "exact_canonical_match_count": 0,
            "widget_type_match_count": 0,
            "total_reward_sum": 0.0,
        }

        n = min(self.num_samples, len(self.eval_dataset))
        with pred_path.open("w", encoding="utf-8") as f:
            for idx in range(n):
                row = self.eval_dataset[int(idx)]
                prompt = str(row["prompt"])
                response_text = str(row.get("response_text", ""))
                reference = canonical_json(row["genui_json"])
                prediction = self._generate_one(model, prompt)
                rewards = _compute_reward_parts(prompt, prediction, reference)

                pred_canon = canonical_json(prediction)
                ref_canon = canonical_json(reference)
                valid = _parse_json_ok(prediction)
                widget_match = _top_level_type(prediction) is not None and _top_level_type(prediction) == _top_level_type(reference)
                exact_match = valid and pred_canon == ref_canon

                metrics["num_samples"] += 1
                metrics["valid_json_count"] += int(valid)
                metrics["exact_canonical_match_count"] += int(exact_match)
                metrics["widget_type_match_count"] += int(widget_match)
                metrics["total_reward_sum"] += rewards["total_reward"]

                out = {
                    "step": step,
                    "idx": idx,
                    "response_text": response_text,
                    "prompt": prompt,
                    "reference_genui_json": reference,
                    "prediction": prediction,
                    "prediction_canonical": pred_canon,
                    "valid_json": valid,
                    "exact_canonical_match": exact_match,
                    "widget_type_match": widget_match,
                    "rewards": rewards,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        denom = max(metrics["num_samples"], 1)
        metrics["valid_json_rate"] = metrics["valid_json_count"] / denom
        metrics["exact_canonical_match_rate"] = metrics["exact_canonical_match_count"] / denom
        metrics["widget_type_match_rate"] = metrics["widget_type_match_count"] / denom
        metrics["avg_total_reward"] = metrics["total_reward_sum"] / denom

        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if was_training:
            model.train()


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


def build_grpo_config(cfg: ScriptConfig) -> GRPOConfig:
    """Build GRPOConfig while tolerating TRL version differences.

    Some TRL releases do not expose newer fields like max_prompt_length,
    max_completion_length, num_generations, log_completions, or use the older
    TrainingArguments name evaluation_strategy instead of eval_strategy.
    Unsupported keys are ignored with a printed warning instead of crashing.
    """
    kwargs: Dict[str, Any] = {
        "output_dir": cfg.output_dir,
        "run_name": cfg.run_name,
        "report_to": [x.strip() for x in cfg.report_to.split(",") if x.strip()],
        "logging_dir": str(Path(cfg.output_dir) / "logs"),
        "seed": cfg.seed,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_ratio": cfg.warmup_ratio,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "logging_steps": cfg.logging_steps,
        "eval_strategy": "steps",
        "eval_steps": cfg.eval_steps,
        "save_strategy": "steps",
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "bf16": cfg.bf16,
        "fp16": cfg.fp16,
        "beta": cfg.beta,
        "max_prompt_length": cfg.max_prompt_length,
        "max_completion_length": cfg.max_completion_length,
        "num_generations": cfg.num_generations,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "remove_unused_columns": False,
        "log_completions": True,
    }

    signature = inspect.signature(GRPOConfig.__init__)
    supported = set(signature.parameters.keys())

    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    filtered = {key: value for key, value in kwargs.items() if key in supported}
    ignored = sorted(set(kwargs) - set(filtered))
    if ignored:
        print(f"[GRPOConfig compatibility] Ignoring unsupported args for installed TRL version: {ignored}")

    return GRPOConfig(**filtered)


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
    training_args = build_grpo_config(cfg)

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
    trainer.add_callback(
        PeriodicPredictionCallback(
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
    main()
