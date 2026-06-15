"""Data utilities for GRPO training.

Expected input JSONL format:
{"response_text": "...", "genui_json": "..."}

`genui_json` may be a JSON string or an already parsed object. The training dataset
keeps both the raw user text and the canonical reference JSON string so reward
functions can compare generated completions against the reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset


DEFAULT_PROMPT_TEMPLATE = """You are a GenUI JSON generation assistant.

Your task is to convert the user request into the correct GenUI JSON.
Return only raw JSON.
Do not include markdown fences.
Do not include explanation.
Do not include any text before or after the JSON.

User request:
{response_text}

GenUI JSON:
"""


def canonical_json(value: Any) -> str:
    """Return a stable JSON string when possible; otherwise return stripped text."""
    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    text = str(value).strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return text


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            if "response_text" not in row or "genui_json" not in row:
                raise KeyError(
                    f"Missing required keys at {path}:{line_no}. "
                    "Expected keys: response_text, genui_json"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def make_prompt(response_text: str, prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return prompt_template.format(response_text=str(response_text).strip())


def rows_to_dataset(rows: List[Dict[str, Any]], prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> Dataset:
    formatted: List[Dict[str, Any]] = []
    for row in rows:
        response_text = str(row["response_text"]).strip()
        reference_json = canonical_json(row["genui_json"])
        formatted.append(
            {
                "prompt": make_prompt(response_text, prompt_template),
                "response_text": response_text,
                "genui_json": reference_json,
            }
        )
    return Dataset.from_list(formatted)


def load_train_eval_datasets(
    train_jsonl: str | Path,
    eval_jsonl: Optional[str | Path] = None,
    validation_split: float = 0.05,
    seed: int = 42,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> Tuple[Dataset, Dataset]:
    train_rows = load_jsonl(train_jsonl)
    full_train = rows_to_dataset(train_rows, prompt_template)

    if eval_jsonl:
        eval_rows = load_jsonl(eval_jsonl)
        return full_train, rows_to_dataset(eval_rows, prompt_template)

    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be between 0 and 1 when eval_jsonl is not provided")

    split = full_train.train_test_split(test_size=validation_split, seed=seed)
    return split["train"], split["test"]
