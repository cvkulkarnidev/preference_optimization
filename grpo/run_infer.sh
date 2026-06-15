#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_PATH="/home/c.kulkarni/hf_models/google/gemma-4-E2B-it"
MODEL_OR_ADAPTER_PATH="./outputs/grpo_genui"
TEST_JSONL="/path/to/test.jsonl"
OUTPUT_DIR="./outputs/grpo_genui/test_predictions"

# true when MODEL_OR_ADAPTER_PATH contains a LoRA adapter; false for a fully saved model
IS_LORA_ADAPTER=true

MAX_PROMPT_LENGTH=1024
MAX_NEW_TOKENS=512
DO_SAMPLE=false
TEMPERATURE=0.0
TOP_P=1.0
BF16=true

python grpo/infer_test.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --model_or_adapter_path "${MODEL_OR_ADAPTER_PATH}" \
  --test_jsonl "${TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}" \
  --is_lora_adapter "${IS_LORA_ADAPTER}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --do_sample "${DO_SAMPLE}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --bf16 "${BF16}"
