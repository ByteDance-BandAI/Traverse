# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rollout.review_paths import review_jsonl_read_paths


def _finite_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if metric != metric or metric in (float("inf"), float("-inf")):
        return None
    return metric


def _read_latest_review_records(
    rollout_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Read review logs, keeping the latest row for each question ID."""

    records_by_question: dict[str, dict[str, Any]] = {}
    malformed_lines = 0
    for review_path in review_jsonl_read_paths(rollout_dir):
        if not review_path.is_file():
            continue
        with review_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(record, dict):
                    malformed_lines += 1
                    continue
                question_id = str(record.get("question_id", "")).strip()
                if not question_id:
                    malformed_lines += 1
                    continue
                records_by_question[question_id] = record
    return list(records_by_question.values()), malformed_lines


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful_judgments = 0
    exact_correct = 0
    empty_model_responses = 0
    valid_precision_sum = 0.0
    valid_recall_sum = 0.0
    valid_f1_sum = 0.0
    judge_models: set[str] = set()

    for record in records:
        answer = record.get("answer")
        if answer is None or not str(answer).strip():
            empty_model_responses += 1

        metrics = record.get("judge_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        judge_model = str(metrics.get("judge_model", "")).strip()
        if judge_model:
            judge_models.add(judge_model)

        precision = _finite_metric(metrics.get("precision"))
        recall = _finite_metric(metrics.get("recall"))
        f1 = _finite_metric(metrics.get("f1"))
        valid = (
            record.get("judge_complete_status") == "SUCCESS"
            and precision is not None
            and recall is not None
            and f1 is not None
        )
        if not valid:
            continue

        successful_judgments += 1
        valid_precision_sum += precision
        valid_recall_sum += recall
        valid_f1_sum += f1
        if record.get("is_correct") is True:
            exact_correct += 1

    total = len(records)
    invalid_judgments = total - successful_judgments

    # Benchmark scores use all recorded attempts as the denominator. A missing
    # or unparseable autorater output contributes zero, rather than silently
    # improving the score by being dropped.
    macro_precision = valid_precision_sum / total if total else 0.0
    macro_recall = valid_recall_sum / total if total else 0.0
    macro_f1 = valid_f1_sum / total if total else 0.0
    return {
        "records": total,
        "successful_judgments": successful_judgments,
        "invalid_judgments": invalid_judgments,
        "empty_model_responses": empty_model_responses,
        "exact_correct": exact_correct,
        "all_correct_rate": exact_correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "valid_only_macro_precision": (
            valid_precision_sum / successful_judgments
            if successful_judgments
            else 0.0
        ),
        "valid_only_macro_recall": (
            valid_recall_sum / successful_judgments
            if successful_judgments
            else 0.0
        ),
        "valid_only_macro_f1": (
            valid_f1_sum / successful_judgments
            if successful_judgments
            else 0.0
        ),
        "judge_models": sorted(judge_models),
    }


def summarize_dsqa_reviews(
    work_dir: str | Path,
    rollout_n: int,
    expected_num_rows: int = 900,
    *,
    benchmark: str = "deepsearchqa_900",
    judge_profile: str = "gemini-2.5-flash",
) -> dict[str, Any]:
    work_path = Path(work_dir)
    all_records: list[dict[str, Any]] = []
    per_rollout: list[dict[str, Any]] = []
    malformed_lines = 0

    for rollout_idx in range(rollout_n):
        records, rollout_malformed = _read_latest_review_records(
            work_path / f"rollout_{rollout_idx}"
        )
        rollout_summary = _summarize_records(records)
        rollout_summary["rollout_idx"] = rollout_idx
        rollout_summary["malformed_review_lines"] = rollout_malformed
        per_rollout.append(rollout_summary)
        all_records.extend(records)
        malformed_lines += rollout_malformed

    expected_records = expected_num_rows * rollout_n
    summary = {
        "benchmark": benchmark,
        "judge_profile": judge_profile,
        "metric_protocol": (
            "component precision/recall/F1 with excessive answers counted "
            "as false positives"
        ),
        "expected_records": expected_records,
        "missing_records": max(expected_records - len(all_records), 0),
        "coverage": (
            min(len(all_records) / expected_records, 1.0)
            if expected_records
            else 0.0
        ),
        "malformed_review_lines": malformed_lines,
        **_summarize_records(all_records),
        "per_rollout": per_rollout,
    }

    work_path.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".benchmark_summary.",
        suffix=".json",
        dir=str(work_path),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    output_path = work_path / "benchmark_summary.json"
    try:
        temporary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return summary


def _summarize_widesearch_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "row_precision",
        "row_recall",
        "row_f1",
        "item_precision",
        "item_recall",
        "item_f1",
    )
    metric_sums = {name: 0.0 for name in metric_names}
    successful_judgments = 0
    exact_successes = 0
    empty_model_responses = 0
    judge_models: set[str] = set()

    for record in records:
        answer = record.get("answer")
        if answer is None or not str(answer).strip():
            empty_model_responses += 1
        metrics = record.get("judge_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        judge_model = str(metrics.get("judge_model", "")).strip()
        if judge_model:
            judge_models.add(judge_model)
        values = {
            name: _finite_metric(metrics.get(name))
            for name in metric_names
        }
        valid = (
            record.get("judge_complete_status") == "SUCCESS"
            and all(value is not None for value in values.values())
        )
        if not valid:
            continue
        successful_judgments += 1
        for name, value in values.items():
            metric_sums[name] += float(value)
        if record.get("is_correct") is True:
            exact_successes += 1

    total = len(records)
    invalid_judgments = total - successful_judgments
    return {
        "records": total,
        "successful_judgments": successful_judgments,
        "invalid_judgments": invalid_judgments,
        "empty_model_responses": empty_model_responses,
        "exact_successes": exact_successes,
        "success_rate": exact_successes / total if total else 0.0,
        **{
            name: metric_sums[name] / total if total else 0.0
            for name in metric_names
        },
        **{
            f"valid_only_{name}": (
                metric_sums[name] / successful_judgments
                if successful_judgments
                else 0.0
            )
            for name in metric_names
        },
        "judge_models": sorted(judge_models),
    }


def summarize_widesearch_reviews(
    work_dir: str | Path,
    rollout_n: int,
    expected_num_rows: int = 200,
) -> dict[str, Any]:
    work_path = Path(work_dir)
    all_records: list[dict[str, Any]] = []
    per_rollout: list[dict[str, Any]] = []
    malformed_lines = 0

    for rollout_idx in range(rollout_n):
        records, rollout_malformed = _read_latest_review_records(
            work_path / f"rollout_{rollout_idx}"
        )
        rollout_summary = _summarize_widesearch_records(records)
        rollout_summary["rollout_idx"] = rollout_idx
        rollout_summary["malformed_review_lines"] = rollout_malformed
        per_rollout.append(rollout_summary)
        all_records.extend(records)
        malformed_lines += rollout_malformed

    records_by_question: dict[str, list[dict[str, Any]]] = {}
    for record in all_records:
        question_id = str(record.get("question_id", "")).strip()
        if question_id:
            records_by_question.setdefault(question_id, []).append(record)

    pass_values = []
    max_row_f1_values = []
    max_item_f1_values = []
    for question_records in records_by_question.values():
        pass_values.append(
            float(any(record.get("is_correct") is True for record in question_records))
        )
        row_values = []
        item_values = []
        for record in question_records:
            metrics = record.get("judge_metrics")
            if not isinstance(metrics, dict):
                metrics = {}
            row_values.append(_finite_metric(metrics.get("row_f1")) or 0.0)
            item_values.append(_finite_metric(metrics.get("item_f1")) or 0.0)
        max_row_f1_values.append(max(row_values, default=0.0))
        max_item_f1_values.append(max(item_values, default=0.0))

    expected_records = expected_num_rows * rollout_n
    aggregate = _summarize_widesearch_records(all_records)
    summary = {
        "benchmark": "widesearch_200",
        "judge_profile": "gpt-4.1-2025-04-14",
        "metric_protocol": (
            "official WideSearch Success Rate, row precision/recall/F1, "
            "and item precision/recall/F1"
        ),
        "expected_records": expected_records,
        "missing_records": max(expected_records - len(all_records), 0),
        "coverage": (
            min(len(all_records) / expected_records, 1.0)
            if expected_records
            else 0.0
        ),
        "unique_questions": len(records_by_question),
        "missing_questions": max(
            expected_num_rows - len(records_by_question),
            0,
        ),
        "malformed_review_lines": malformed_lines,
        **aggregate,
        # Official multi-trial reporting: average over trials and the best
        # observed trial for each task.
        "avg_at_n_success_rate": aggregate["success_rate"],
        "avg_at_n_row_f1": aggregate["row_f1"],
        "avg_at_n_item_f1": aggregate["item_f1"],
        "pass_at_n_success_rate": (
            sum(pass_values) / len(pass_values) if pass_values else 0.0
        ),
        "max_at_n_row_f1": (
            sum(max_row_f1_values) / len(max_row_f1_values)
            if max_row_f1_values
            else 0.0
        ),
        "max_at_n_item_f1": (
            sum(max_item_f1_values) / len(max_item_f1_values)
            if max_item_f1_values
            else 0.0
        ),
        "per_rollout": per_rollout,
    }

    work_path.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=".benchmark_summary.",
        suffix=".json",
        dir=str(work_path),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    output_path = work_path / "benchmark_summary.json"
    try:
        temporary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return summary
