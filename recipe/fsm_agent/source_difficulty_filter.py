# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_row_matches_difficulty(
    row: dict[str, Any],
    min_num_turns: int,
    *,
    correct_only: bool = False,
) -> tuple[bool, str]:
    qid = str(row.get("question_id") or "").strip()
    if not qid:
        return False, "skip_missing_question_id"

    if correct_only and not _as_bool(row.get("is_correct")):
        return False, "skip_incorrect"

    num_turns = _to_int(row.get("num_turns"))
    if num_turns is None:
        return False, "skip_missing_num_turns"
    if num_turns < min_num_turns:
        return False, "skip_below_min_num_turns"

    return True, "selected_rows"


def load_source_difficulty_question_ids(
    review_jsonl: str | Path,
    min_num_turns: int,
    *,
    correct_only: bool = False,
) -> tuple[set[str], dict[str, int]]:
    """Load question IDs whose source run took at least ``min_num_turns`` turns."""
    review_path = Path(review_jsonl)
    stats: Counter[str] = Counter()
    selected: set[str] = set()

    with review_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            stats["review_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["skip_json_decode_error"] += 1
                continue

            keep, reason = source_row_matches_difficulty(
                row,
                min_num_turns,
                correct_only=correct_only,
            )
            if not keep:
                stats[reason] += 1
                continue
            qid = str(row.get("question_id") or "").strip()
            selected.add(qid)
            stats[reason] += 1

    stats["selected_unique_question_ids"] = len(selected)
    return selected, dict(stats)
