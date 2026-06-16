"""GPU-only entrypoint for GRPO training.

This wrapper prevents silent CPU fallback. Use this when you expect training to run
on a CUDA GPU. If CUDA is not visible to PyTorch, it raises a clear error instead
of changing use_cpu to true.
"""

from __future__ import annotations

import torch

import train_grpo


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


train_grpo.normalize_runtime_config = normalize_runtime_config_gpu_only

if __name__ == "__main__":
    train_grpo.main()
