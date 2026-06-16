#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Editable paths
# -----------------------------
MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it"
TRAIN_JSONL="/home/c.kulkarni/mdc/GenUI-LM/training_scripts/sft/gen_data_v2/genui_processed_clean_merged.jsonl"
EVAL_JSONL=""   # Keep empty to auto-split train into train/val
OUTPUT_DIR="./outputs/grpo_genui"

# -----------------------------
# Dataset / split
# -----------------------------
VALIDATION_SPLIT=0.05
SEED=42

# -----------------------------
# Runtime / precision
# For one GPU, keep USE_CPU=false.
# FP16=true is safer than BF16 on older/non-bf16 GPUs.
# If your GPU supports bf16, you can set BF16=true and FP16=false.
# -----------------------------
USE_CPU=false
BF16=false
FP16=true
TF32=true

# -----------------------------
# GRPO generation settings
# Important: global train batch size should be divisible by NUM_GENERATIONS.
# global batch = PER_DEVICE_TRAIN_BATCH_SIZE * num_processes * GRADIENT_ACCUMULATION_STEPS
# For one GPU below: 1 * 1 * 8 = 8, divisible by NUM_GENERATIONS=4.
# -----------------------------
MAX_PROMPT_LENGTH=4096
MAX_COMPLETION_LENGTH=4096
NUM_GENERATIONS=4
TEMPERATURE=0.9
TOP_P=0.95
BETA=0.04

# -----------------------------
# Optimization
# -----------------------------
LEARNING_RATE=5e-6
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
NUM_TRAIN_EPOCHS=1
MAX_STEPS=-1
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
GRADIENT_CHECKPOINTING=true

# -----------------------------
# Logging/checkpoints
# -----------------------------
LOGGING_STEPS=5
EVAL_STEPS=100
SAVE_STEPS=100
SAVE_TOTAL_LIMIT=3
REPORT_TO="tensorboard"
RUN_NAME="grpo_genui"
RESUME_FROM_CHECKPOINT=""

# -----------------------------
# Periodic prediction saving during training
# Set PREDICTION_SAVE_STEPS=0 to disable.
# Outputs are written to: ${OUTPUT_DIR}/predictions/step_<N>/
# -----------------------------
PREDICTION_SAVE_STEPS=100
PREDICTION_NUM_SAMPLES=16
PREDICTION_MAX_NEW_TOKENS=4096
PREDICTION_DO_SAMPLE=false
PREDICTION_TEMPERATURE=0.0
PREDICTION_TOP_P=1.0

# -----------------------------
# LoRA / QLoRA settings
# -----------------------------
USE_LORA=true
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_MODULES_TO_SAVE=""
LOAD_IN_4BIT=false
BNB_4BIT_COMPUTE_DTYPE="bfloat16"

mkdir -p "${OUTPUT_DIR}"

# -----------------------------
# CUDA preflight
# This uses the same python that runs training.
# -----------------------------
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
    raise SystemExit(
        "CUDA is not visible in this Python environment. Fix the active env / torch CUDA wheel before training."
    )
PY

CMD=(
  python grpo/train_grpo_gpu.py
  --model_path "${MODEL_PATH}"
  --train_jsonl "${TRAIN_JSONL}"
  --output_dir "${OUTPUT_DIR}"
  --validation_split "${VALIDATION_SPLIT}"
  --seed "${SEED}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --num_generations "${NUM_GENERATIONS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING}"
  --logging_steps "${LOGGING_STEPS}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --prediction_save_steps "${PREDICTION_SAVE_STEPS}"
  --prediction_num_samples "${PREDICTION_NUM_SAMPLES}"
  --prediction_max_new_tokens "${PREDICTION_MAX_NEW_TOKENS}"
  --prediction_do_sample "${PREDICTION_DO_SAMPLE}"
  --prediction_temperature "${PREDICTION_TEMPERATURE}"
  --prediction_top_p "${PREDICTION_TOP_P}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --beta "${BETA}"
  --bf16 "${BF16}"
  --fp16 "${FP16}"
  --tf32 "${TF32}"
  --use_cpu "${USE_CPU}"
  --use_lora "${USE_LORA}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --lora_target_modules "${LORA_TARGET_MODULES}"
  --load_in_4bit "${LOAD_IN_4BIT}"
  --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}"
  --report_to "${REPORT_TO}"
  --run_name "${RUN_NAME}"
)

if [[ -n "${EVAL_JSONL}" ]]; then
  CMD+=(--eval_jsonl "${EVAL_JSONL}")
fi

if [[ -n "${LORA_MODULES_TO_SAVE}" ]]; then
  CMD+=(--lora_modules_to_save "${LORA_MODULES_TO_SAVE}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${CMD[@]}"
