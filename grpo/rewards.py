"""Reward functions for GRPO training.

The functions are intentionally modular so you can edit the scoring rules later.
TRL GRPOTrainer accepts reward functions with signature:
    reward_func(prompts, completions, **kwargs) -> list[float]

This file assumes each dataset row contains `genui_json`, which is the canonical
reference JSON string.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple


MARKDOWN_OR_PROSE_PATTERNS = [
    r"```",
    r"^\s*here\s+is\b",
    r"^\s*sure\b",
    r"^\s*the\s+json\b",
    r"^\s*output\s*:",
]


REQUIRED_TOP_LEVEL_KEYS: Tuple[str, ...] = ()
FORBIDDEN_TOP_LEVEL_KEYS: Tuple[str, ...] = ("debug", "internal", "admin", "password", "secret")


def _completion_to_text(completion: Any) -> str:
    """Handle both plain string completions and chat-style completions."""
    if isinstance(completion, str):
        return completion.strip()

    if isinstance(completion, list):
        # Chat-style: [{"role": "assistant", "content": "..."}]
        parts: List[str] = []
        for msg in completion:
            if isinstance(msg, dict) and "content" in msg:
                parts.append(str(msg["content"]))
            else:
                parts.append(str(msg))
        return "\n".join(parts).strip()

    return str(completion).strip()


def _parse_json(text: str) -> Tuple[Optional[Any], bool]:
    try:
        return json.loads(text), True
    except Exception:
        return None, False


def _canonical(value: Any) -> str:
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(value).strip()


def _top_level_type(obj: Any) -> Optional[str]:
    if isinstance(obj, dict):
        for key in ("type", "widget", "name", "component"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
    return None


def _flatten_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.append(path)
            keys.extend(_flatten_keys(value, path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            keys.extend(_flatten_keys(value, f"{prefix}[]" if prefix else "[]"))
    return keys


def valid_json_reward(prompts: List[str], completions: List[Any], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        _, ok = _parse_json(text)
        rewards.append(1.0 if ok else -2.0)
    return rewards


def main_stack_first_key_reward(prompts: List[str], completions: List[Any], **kwargs: Any) -> List[float]:
    """Small structural reward: root object should start with main_stack.

    Keep this reward mild. If it is too strong, the model can over-optimize the
    root shape and produce odd content inside the JSON.
    """
    rewards: List[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        obj, ok = _parse_json(text)

        if not ok or not isinstance(obj, dict) or not obj:
            rewards.append(0.0)
            continue

        first_key = next(iter(obj.keys()))
        if first_key == "main_stack":
            rewards.append(0.5)
        elif "main_stack" in obj:
            rewards.append(0.1)
        else:
            rewards.append(-0.25)

    return rewards


def no_markdown_or_prose_reward(prompts: List[str], completions: List[Any], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        lower = text.lower().strip()
        has_bad_pattern = any(re.search(pattern, lower) for pattern in MARKDOWN_OR_PROSE_PATTERNS)
        starts_json = lower.startswith("{") or lower.startswith("[")
        ends_json = lower.endswith("}") or lower.endswith("]")
        rewards.append(0.5 if starts_json and ends_json and not has_bad_pattern else -0.75)
    return rewards


def widget_type_match_reward(prompts: List[str], completions: List[Any], genui_json: List[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion, reference in zip(completions, genui_json):
        text = _completion_to_text(completion)
        pred, pred_ok = _parse_json(text)
        ref, ref_ok = _parse_json(reference)
        if not pred_ok or not ref_ok:
            rewards.append(0.0)
            continue

        pred_type = _top_level_type(pred)
        ref_type = _top_level_type(ref)
        if ref_type is None:
            rewards.append(0.0)
        elif pred_type == ref_type:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards


def schema_key_overlap_reward(prompts: List[str], completions: List[Any], genui_json: List[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion, reference in zip(completions, genui_json):
        text = _completion_to_text(completion)
        pred, pred_ok = _parse_json(text)
        ref, ref_ok = _parse_json(reference)
        if not pred_ok or not ref_ok:
            rewards.append(0.0)
            continue

        if not isinstance(pred, dict) or not isinstance(ref, dict):
            rewards.append(0.0)
            continue

        pred_keys = set(_flatten_keys(pred))
        ref_keys = set(_flatten_keys(ref))
        if not ref_keys:
            rewards.append(0.0)
            continue

        recall = len(pred_keys & ref_keys) / max(len(ref_keys), 1)
        precision = len(pred_keys & ref_keys) / max(len(pred_keys), 1)
        f1 = 2 * recall * precision / max(recall + precision, 1e-8)

        reward = float(f1)
        missing_required = [k for k in REQUIRED_TOP_LEVEL_KEYS if k not in pred]
        forbidden_present = [k for k in FORBIDDEN_TOP_LEVEL_KEYS if k in pred]
        reward -= 0.25 * len(missing_required)
        reward -= 0.5 * len(forbidden_present)
        rewards.append(max(-1.0, min(1.0, reward)))
    return rewards


def reference_similarity_reward(prompts: List[str], completions: List[Any], genui_json: List[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion, reference in zip(completions, genui_json):
        text = _completion_to_text(completion)
        pred, pred_ok = _parse_json(text)
        if pred_ok:
            pred_text = _canonical(pred)
        else:
            pred_text = text.strip()
        ref_text = _canonical(reference)
        similarity = SequenceMatcher(None, pred_text, ref_text).ratio()
        rewards.append(2.0 * float(similarity))
    return rewards


def length_sanity_reward(prompts: List[str], completions: List[Any], genui_json: List[str], **kwargs: Any) -> List[float]:
    rewards: List[float] = []
    for completion, reference in zip(completions, genui_json):
        text = _completion_to_text(completion)
        ref_len = max(len(str(reference)), 1)
        pred_len = len(text)
        ratio = pred_len / ref_len
        if 0.4 <= ratio <= 2.5:
            rewards.append(0.25)
        elif ratio > 5.0:
            rewards.append(-0.5)
        else:
            rewards.append(-0.1)
    return rewards


# Conservative active rewards only.
# The richer reference/schema rewards above are kept for later experiments, but they
# are disabled by default because they made outputs drift/turn odd in early GRPO.
REWARD_FUNCS = [
    valid_json_reward,
    main_stack_first_key_reward,
]
