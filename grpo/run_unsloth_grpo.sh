#!/usr/bin/env bash
set -euo pipefail

# Run from inside grpo/:
#   cd grpo
#   bash run_unsloth_grpo.sh

# Force local-only loading. This prevents accidental downloads such as gpt-oss-20b.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Reduce CUDA fragmentation on generation workloads.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:32"

MODEL_PATH="/home/c_kulkarni/models/gemma-4-E2B-it"
TRAIN_JSONL="/home/c_kulkarni/grpo/genui_processed_clean_merged.jsonl"
EVAL_JSONL=""
OUTPUT_DIR="./outputs/unsloth_grpo_genui"

VALIDATION_SPLIT=0.05
SEED=42

# Extreme low-memory smoke-test config.
# This is only to verify that GRPO can step without OOM.
# Increase these only after the smoke test works.
MAX_PROMPT_LENGTH=128
MAX_COMPLETION_LENGTH=128
MAX_SEQ_LENGTH=256

# GRPO minimum valid setting.
NUM_GENERATIONS=2
PER_DEVICE_TRAIN_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=2

TEMPERATURE=0.7
TOP_P=0.9
BETA=0.04

LEARNING_RATE=2e-6
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1
GRADIENT_ACCUMULATION_STEPS=8

# Monitoring + checkpointing
LOGGING_STEPS=5
EVAL_STRATEGY="no"
EVAL_STEPS=0
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=2
REPORT_TO="tensorboard"
RUN_NAME="unsloth_grpo_genui"
RESUME_FROM_CHECKPOINT=""

LOAD_IN_4BIT=true
FAST_INFERENCE=false
GPU_MEMORY_UTILIZATION=0.20

USE_LORA=true
LORA_R=2
LORA_ALPHA=4
LORA_DROPOUT=0.05
# Start with q/v only to reduce adapter + gradient memory. Add more modules after it runs.
LORA_TARGET_MODULES="q_proj,v_proj"

BF16=false
FP16=true
TF32=true

mkdir -p "${OUTPUT_DIR}"

echo "Python: $(which python)"
python - <<'PY'
import os
import sys
from pathlib import Path
import torch

model_path = Path("/home/c_kulkarni/models/gemma-4-E2B-it")
train_jsonl = Path("/home/c_kulkarni/grpo/genui_processed_clean_merged.jsonl")

print("python executable:", sys.executable)
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("HF_HUB_OFFLINE:", os.environ.get("HF_HUB_OFFLINE"))
print("TRANSFORMERS_OFFLINE:", os.environ.get("TRANSFORMERS_OFFLINE"))
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("model path exists:", model_path.exists(), model_path)
print("train jsonl exists:", train_jsonl.exists(), train_jsonl)

if not model_path.exists():
    raise SystemExit(f"Model path missing: {model_path}. Refusing to download fallback model.")
if not train_jsonl.exists():
    raise SystemExit(f"Train JSONL missing: {train_jsonl}")
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print("gpu free before load GB:", round(free / 1024**3, 3))
    print("gpu total GB:", round(total / 1024**3, 3))
    print("gpu memory allocated before load GB:", round(torch.cuda.memory_allocated(0) / 1024**3, 3))
    print("gpu memory reserved before load GB:", round(torch.cuda.memory_reserved(0) / 1024**3, 3))
else:
    raise SystemExit("CUDA is not visible in this Python environment.")
PY

CMD=(
  python train_unsloth_grpo.py
  --model_path "${MODEL_PATH}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split "${VALIDATION_SPLIT}"
  --seed "${SEED}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --num_generations "${NUM_GENERATIONS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --beta "${BETA}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --logging_steps "${LOGGING_STEPS}"
  --eval_strategy "${EVAL_STRATEGY}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --report_to "${REPORT_TO}"
  --run_name "${RUN_NAME}"
  --load_in_4bit "${LOAD_IN_4BIT}"
  --fast_inference "${FAST_INFERENCE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --use_lora "${USE_LORA}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --bf16 "${BF16}"
  --fp16 "${FP16}"
  --tf32 "${TF32}"
)

if [[ -n "${EVAL_JSONL}" ]]; then
  CMD+=(--eval_jsonl "${EVAL_JSONL}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${CMD[@]}"
