# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataset.base import ParquetDataset, JsonlDataset
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BCReturn:
    question_id: int
    question: str
    ground_truth: str
    benchmark: str = "browsecomp"
    judge_metadata: dict[str, Any] = field(default_factory=dict)


class BrowseCompDataset(ParquetDataset):
    def __init__(
        self,
        path: str,
        limit: int | None = None,
        shuffle_buffer_size: int = 512,
        benchmark: str = "browsecomp",
    ) -> None:
        self.benchmark = benchmark
        super().__init__(
            path,
            limit=limit,
            shuffle_buffer_size=shuffle_buffer_size,
        )

    def _parse_row(self, row) -> BCReturn:
        # Parse extra information
        extra_info = row['extra_info']
        
        # Extract question text
        question = extra_info['question']
        
        # Parse ground-truth answers
        reward_model = row['reward_model']
        ground_truth = reward_model['ground_truth']
        if hasattr(ground_truth, 'tolist'):
            ground_truth = ground_truth.tolist()
        elif not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        
        # Question ID fallback
        question_id = extra_info['id']
        judge_metadata = extra_info.get("judge_metadata") or {}
        if hasattr(judge_metadata, "as_py"):
            judge_metadata = judge_metadata.as_py()
        if not isinstance(judge_metadata, dict):
            raise ValueError(
                f"judge_metadata for question {question_id!r} must be a mapping"
            )
        
        return BCReturn(
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
            benchmark=self.benchmark,
            judge_metadata=judge_metadata,
        )


class BrowseCompJsonl(JsonlDataset):
    def __init__(self, path: str, limit: int | None = None) -> None:
        super().__init__(path, limit=limit)

    def _parse_row(self, row) -> BCReturn:
        extra_info = row['extra_info']
        question = extra_info['question']
        ground_truth = row['reward_model']['ground_truth']
        if hasattr(ground_truth, 'tolist'):
            ground_truth = ground_truth.tolist()
        elif not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        question_id = extra_info['item_id']

        return BCReturn(
            question_id=question_id,
            question=question,
            ground_truth=ground_truth,
        )
