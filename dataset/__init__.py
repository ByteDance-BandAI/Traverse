# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataset.bc_dataset import BrowseCompDataset, BrowseCompJsonl


def get_bc_dataset(dataset_path: str, num_questions: int = None):
    if dataset_path.endswith(".jsonl"):
        return BrowseCompJsonl(dataset_path, num_questions)
    elif dataset_path.endswith(".parquet"):
        return BrowseCompDataset(dataset_path, num_questions)
    else:
        raise ValueError(f"Unknown dataset type {dataset_path}")
