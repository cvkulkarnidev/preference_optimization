#!/usr/bin/env bash
set -euo pipefail

# Run from inside grpo/:
#   cd grpo
#   bash run_unsloth_grpo.sh

MODEL_PATH="/home/c_kulkarni/models/gemma-4-E2B-it"
TRAIN_JSONL="/home/c_kulkarni/grpo/genui_processed_clean_merged.jsonl"
EVAL_JSONL=""
OUTPUT_DIR="./outputs/unsloth_grpo_genui"

VALIDATION_SPLIT=0.05
SEED=42

MAX_SEQ_LENGTH=8192
MAX_PROMPT_LENGTH=4096
MAX_COMPLETION_LENGTH=4096
NUM_GENERATIONS=1
TEMPERATURE=0.7
TOP_P=0.9
BETA=0.04

LEARNING_RATE=5e-6
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8

LOGGING_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
REPORT_TO="tensorboard"
RUN_NAME="unsloth_grpo_genui"
RESUME_FROM_CHECKPOINT=""

LOAD_IN_4BIT=true
FAST_INFERENCE=false
GPU_MEMORY_UTILIZATION=0.75

USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

BF16=false
FP16=true
TF32=true

mkdir -p "${OUTPUT_DIR}"

echo "Python: $(which python)"
python - <<'PY'
import sys
import torch
print("python executable:", sys.executable)
print("torch version:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
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
