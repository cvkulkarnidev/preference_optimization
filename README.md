# Preference Optimization: GRPO for GenUI JSON

This repository contains a practical Hugging Face TRL-based GRPO pipeline for training a causal language model to generate GenUI JSON from user text.

Expected JSONL data format:

```json
{"response_text": "Create project for ABC on tomorrow", "genui_json": "{\"type\":\"date_picker\",\"label\":\"Select date\"}"}
```

The pipeline includes:

- JSONL loading for `response_text -> genui_json`
- automatic train/validation split
- GRPO training using `trl.GRPOTrainer`
- configurable reward functions
- TensorBoard logging
- checkpoint saving
- test prediction generation
- metrics written as JSON

## Files

```text
grpo/
  train_grpo.py      # Main GRPO training entrypoint
  infer_test.py      # Inference and prediction writer
  data_utils.py      # JSONL loading and prompt formatting
  rewards.py         # Reward functions
  requirements.txt   # Python dependencies
  run_grpo.sh        # Editable launch script
  run_infer.sh       # Editable inference script
```

## Install

```bash
cd preference_optimization
pip install -r grpo/requirements.txt
```

For multi-GPU training, launch with Accelerate:

```bash
accelerate config
bash grpo/run_grpo.sh
```

## TensorBoard

```bash
tensorboard --logdir ./outputs/grpo_genui/logs
```

## Notes

The reward functions are intentionally modular and conservative. You can edit `grpo/rewards.py` to change the scoring logic for your final GenUI schema.
