# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REVIEW_DIR_NAME = "000_review"
REVIEW_FILE_NAME = "review.jsonl"


def review_jsonl_path(rollout_dir: str | Path) -> Path:
    return Path(rollout_dir) / REVIEW_DIR_NAME / REVIEW_FILE_NAME


def legacy_review_jsonl_path(rollout_dir: str | Path) -> Path:
    return Path(rollout_dir) / REVIEW_FILE_NAME


def review_jsonl_candidate_paths(rollout_dir: str | Path) -> tuple[Path, Path]:
    return review_jsonl_path(rollout_dir), legacy_review_jsonl_path(rollout_dir)


def existing_review_jsonl_paths(rollout_dir: str | Path) -> list[Path]:
    return [
        path
        for path in review_jsonl_candidate_paths(rollout_dir)
        if path.is_file()
    ]


def review_jsonl_read_paths(rollout_dir: str | Path) -> list[Path]:
    existing = existing_review_jsonl_paths(rollout_dir)
    if existing:
        return existing
    return [review_jsonl_path(rollout_dir)]
