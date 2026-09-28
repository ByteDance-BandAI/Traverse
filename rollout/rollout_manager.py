# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import logging
import json
import math

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from typing import Any
from dataset.bc_dataset import BCReturn
from constants import RunContext, trajectories_dir
from verifiers.base_qa_llm_judge import JudgeResult
from rollout.context_factory import BaseContextFactory
from rollout.review_paths import review_jsonl_read_paths
from dataset.base import ParquetDataset

logger = logging.getLogger(__name__)


def _iter_jsonl(path: Path) -> Iterator[Any]:
    """Yield valid, non-empty JSONL records and ignore malformed lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def _extract_completed_qids(review_file: Path) -> set[str]:
    """Safely parse question IDs from a JSONL file."""
    if not review_file.is_file():
        return set()

    qids = set()
    for row in _iter_jsonl(review_file):
        if isinstance(row, dict) and (qid := row.get("question_id")):
            qids.add(qid)
    return qids


def _clean_partial_dirs(rollout_dir: Path, completed_qids: set[str]) -> int:
    """Delete uncompleted question directories and return the deleted count."""
    cleaned = 0
    for qdir in trajectories_dir(rollout_dir).glob("question_*"):
        if qdir.is_dir() and qdir.name.removeprefix("question_") not in completed_qids:
            shutil.rmtree(qdir)
            cleaned += 1
    return cleaned


def _prepare_resume(work_dir: str, rollout_n: int) -> set[tuple[str, int]]:
    """Scan an existing *work_dir* and return the set of completed (question_id, rollout_idx).
    
    Side-effect: Deletes partial question directories not found in review logs.
    """
    completed: set[tuple[str, int]] = set()
    n_cleaned = 0

    for r_idx in range(rollout_n):
        r_dir = Path(work_dir) / f"rollout_{r_idx}"
        if not r_dir.exists():
            continue

        qids = set()
        for review_file in review_jsonl_read_paths(r_dir):
            qids.update(_extract_completed_qids(review_file))
        completed.update((qid, r_idx) for qid in qids)
        n_cleaned += _clean_partial_dirs(r_dir, qids)

    print(
        f"Resume: {len(completed)} (question, rollout) pair(s) already complete — will be skipped. "
        f"{n_cleaned} partial trajectory dir(s) deleted."
    )
    return completed


class RetryTrajHook:
    def __call__(self, context: RunContext) -> bool:
        return False


@dataclass
class DeferredRetry:
    raw: BCReturn
    rollout_idx: int
    attempt: int


class GenerationController:
    def __init__(
        self,
        args: Any,
        agent: Any,
        judge: Any,
        context_factory: BaseContextFactory,
        worker_func: Any,
        review_loggers: list[Any],
        should_judge: bool = True,
    ):
        self.args = args
        self.agent = agent
        self.judge = judge
        self.context_factory = context_factory
        self.worker = worker_func
        self.review_loggers = review_loggers
        self.should_judge = should_judge
        
        # State tracking
        self.correct = 0
        self.incorrect = 0
        self.total_tokens = 0
        self.metric_count = 0
        self.metric_sums: dict[str, float] = {}
        self.total_expected_rollouts = 0
        benchmark = args.benchmark.strip().lower()
        benchmark = benchmark.replace("-", "_")
        if benchmark in {
            "dsqa",
            "deepsearchqa",
            "deepsearch_qa",
            "deepsearchqa_900",
            "widesearch_v2_1",
            "widesearch_v2_1_hard_720",
            "widesearch_hard_720",
        }:
            self.live_metric_keys = ("f1",)
        elif benchmark in {"widesearch", "wide_search", "widesearch_200"}:
            self.live_metric_keys = ("row_f1", "item_f1")
        else:
            self.live_metric_keys = ()
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.pbar = None
        self.retry_traj_hooks: list[RetryTrajHook] = []

    def should_retry_whole_traj(self, context: RunContext) -> bool:
        return any(hook(context) for hook in self.retry_traj_hooks)

    def register_retry_hook(self, func: RetryTrajHook):
        self.retry_traj_hooks.append(func)

    async def _pre_run_prepare(self):
        if self.args.resume:
            completed = _prepare_resume(self.args.work_dir, self.args.rollout_n)
            completed_records = self._load_completed_review_records()
            self._seed_live_metrics_from_completed_records(completed_records)
            return completed
        return set()

    @staticmethod
    def _is_deferred_retry(result: Any) -> bool:
        return isinstance(result, DeferredRetry)

    async def _drain_one_finished_task(
        self,
        pending_tasks: set[asyncio.Task],
        retry_queue: deque[DeferredRetry],
    ) -> set[asyncio.Task]:
        done, pending_tasks = await asyncio.wait(
            pending_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if task.cancelled():
                continue
            result = task.result()
            if self._is_deferred_retry(result):
                retry_queue.append(result)
        return pending_tasks

    async def _schedule_rollout_task(
        self,
        pending_tasks: set[asyncio.Task],
        retry_queue: deque[DeferredRetry],
        raw: BCReturn,
        rollout_idx: int,
        attempt: int = 0,
    ) -> set[asyncio.Task]:
        while len(pending_tasks) >= self.args.concurrency:
            pending_tasks = await self._drain_one_finished_task(
                pending_tasks,
                retry_queue,
            )

        self._start_rollout_task(
            pending_tasks,
            raw,
            rollout_idx,
            attempt=attempt,
        )
        return pending_tasks

    def _start_rollout_task(
        self,
        pending_tasks: set[asyncio.Task],
        raw: BCReturn,
        rollout_idx: int,
        attempt: int = 0,
    ) -> None:
        task = asyncio.create_task(
            self._process_rollout(
                raw,
                rollout_idx,
                attempt=attempt,
            )
        )
        pending_tasks.add(task)

    async def _drain_pending_and_retries(
        self,
        pending_tasks: set[asyncio.Task],
        retry_queue: deque[DeferredRetry],
    ) -> set[asyncio.Task]:
        backfill_started = False
        while pending_tasks or retry_queue:
            if (
                retry_queue
                and len(pending_tasks) < self.args.concurrency
                and not backfill_started
            ):
                logger.warning(
                    "Backfilling whole-rollout retries after all first-pass "
                    "rollouts were submitted: %s queued, %s slot(s) available",
                    len(retry_queue),
                    self.args.concurrency - len(pending_tasks),
                )
                backfill_started = True
            while retry_queue and len(pending_tasks) < self.args.concurrency:
                retry = retry_queue.popleft()
                pending_tasks = await self._schedule_rollout_task(
                    pending_tasks,
                    retry_queue,
                    retry.raw,
                    retry.rollout_idx,
                    attempt=retry.attempt,
                )

            if pending_tasks:
                pending_tasks = await self._drain_one_finished_task(
                    pending_tasks,
                    retry_queue,
                )

        return pending_tasks

    async def run(self, dataset: ParquetDataset):
        # Keep a fixed denominator for live "all" metrics.  tqdm's total only
        # counts the remaining work on resume, while the seeded metric
        # numerators also include rollouts completed by an earlier process.
        self.total_expected_rollouts = len(dataset) * self.args.rollout_n
        completed_pairs = await self._pre_run_prepare()
            
        # We still need the total upfront to calculate the tqdm bar correctly
        total_tasks = len(dataset) * self.args.rollout_n - len(completed_pairs)
        self.pbar = tqdm(total=total_tasks, desc="Rollouts", unit="rollout")
        self._set_pbar_postfix()

        pending_tasks = set()
        retry_queue: deque[DeferredRetry] = deque()

        async for raw in dataset.iter_rows():
            for rollout_idx in range(self.args.rollout_n):
                if (raw.question_id, rollout_idx) in completed_pairs:
                    continue

                pending_tasks = await self._schedule_rollout_task(
                    pending_tasks,
                    retry_queue,
                    raw,
                    rollout_idx,
                )

        # Every first-pass rollout has now been submitted. Keep the remaining
        # first-pass tasks running, and immediately use any freed concurrency
        # slots for queued whole-rollout retries.
        pending_tasks = await self._drain_pending_and_retries(
            pending_tasks,
            retry_queue,
        )

        self.pbar.close()

    async def _process_rollout(
        self,
        raw: BCReturn,
        rollout_idx: int,
        attempt: int = 0,
    ):
        max_retry_limit = self.args.rollout_max_retry_limit

        async with self.semaphore:
            # The Controller uses the Factory. Zero parameter clutter.
            context, start_state = await self.context_factory.create(raw, rollout_idx)
            rollout_dir = Path(self.args.work_dir) / f"rollout_{rollout_idx}"
            whole_retry_enabled = (
                getattr(self.args, "over_turn_retry", -1) >= 0
                or getattr(self.args, "over_seal_retry", -1) >= 0
            )
            self._restore_retry_history(
                context=context,
                rollout_dir=rollout_dir,
                question_id=raw.question_id,
                attempt=attempt,
                whole_retry_enabled=whole_retry_enabled,
            )
            if attempt >= max_retry_limit and whole_retry_enabled:
                if getattr(context, "over_turn_retry", -1) >= 0:
                    context.over_turn_retry = -1
                    if hasattr(context, "over_turn_retry_disabled_final_attempt"):
                        context.over_turn_retry_disabled_final_attempt = True
                if getattr(context, "over_seal_retry", -1) >= 0:
                    context.over_seal_retry = -1
                    if hasattr(context, "over_seal_retry_disabled_final_attempt"):
                        context.over_seal_retry_disabled_final_attempt = True

            result: JudgeResult = await self.worker(
                raw,
                self.agent,
                context,
                self.judge if self.should_judge else None,
                self.review_loggers[rollout_idx],
                state=start_state,
            )

        used_tokens = context.trajectory.get_total_tokens()
        is_correct = getattr(result, "is_correct", False)
        judge_metrics = getattr(result, "metrics", {}) or {}

        should_retry = self.should_retry_whole_traj(context)
        if should_retry and attempt < max_retry_limit:
            logger.warning(
                "Retry whole rollout deferred for qid=%s rollout=%s attempt=%s",
                raw.question_id,
                rollout_idx,
                attempt,
            )
            return DeferredRetry(
                raw=raw,
                rollout_idx=rollout_idx,
                attempt=attempt + 1,
            )

        self._update_metrics(
            used_tokens,
            is_correct,
            delta=1,
            judge_metrics=judge_metrics,
        )
        return None

    def _restore_retry_history(
        self,
        context: RunContext,
        rollout_dir: Path,
        question_id: str,
        attempt: int,
        whole_retry_enabled: bool,
    ) -> int:
        """Restore prior-attempt metadata unless OverSeal clean-start is enabled."""
        if not whole_retry_enabled:
            return 0
        if getattr(self.args, "over_seal_retry_clean_start", False):
            return 0
        if not hasattr(context, "previous_retry_memories"):
            return 0

        loaded_retry_attempt_count = 0
        if hasattr(context, "previous_retry_attempt_count"):
            loaded_retry_attempt_count = self._load_retry_attempt_count(
                rollout_dir,
                question_id,
            )
            context.previous_retry_attempt_count = max(
                attempt,
                loaded_retry_attempt_count,
            )
        context.previous_retry_memories = self._load_retry_memories(
            rollout_dir,
            question_id,
        )
        return loaded_retry_attempt_count

    @staticmethod
    def _finite_metric(value: Any) -> float:
        if isinstance(value, bool):
            return 0.0
        try:
            metric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return metric if math.isfinite(metric) else 0.0

    def _update_metrics(
        self,
        tokens: int,
        is_correct: bool,
        delta: int,
        judge_metrics: dict[str, Any] | None = None,
    ):
        self.total_tokens += (tokens * delta)
        if is_correct:
            self.correct += delta
        else:
            self.incorrect += delta

        if self.live_metric_keys:
            metrics = judge_metrics if isinstance(judge_metrics, dict) else {}
            self.metric_count += delta
            for key in self.live_metric_keys:
                value = self._finite_metric(metrics.get(key))
                self.metric_sums[key] = self.metric_sums.get(key, 0.0) + value * delta
            
        self.pbar.update(delta)
        self._set_pbar_postfix()

    def _set_pbar_postfix(self):
        if self.pbar is None or not hasattr(self.pbar, "set_postfix"):
            return
        judged = self.correct + self.incorrect
        total = self.total_expected_rollouts
        if total <= 0:
            total = getattr(self.pbar, "total", 0) or judged
        total = max(int(total), 1)
        postfix = {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "accuracy": f"{self.correct / judged:.4f}" if judged else "0.0000",
            "accuracy_all": f"{self.correct / total:.4f}",
        }
        if self.live_metric_keys:
            denominator = self.metric_count if self.metric_count > 0 else 1
            for key in self.live_metric_keys:
                metric_sum = self.metric_sums.get(key, 0.0)
                postfix[key] = f"{metric_sum / denominator:.4f}"
                postfix[f"{key}_all"] = f"{metric_sum / total:.4f}"
        postfix["tokens"] = f"{self.total_tokens / 1e6:.1f}M"
        self.pbar.set_postfix(**postfix, refresh=True)

    def _load_completed_review_records(self) -> dict[str, dict[int, dict[str, Any]]]:
        records: dict[str, dict[int, dict[str, Any]]] = {}
        for rollout_idx in range(self.args.rollout_n):
            rollout_dir = Path(self.args.work_dir) / f"rollout_{rollout_idx}"
            for review_file in review_jsonl_read_paths(rollout_dir):
                if not review_file.is_file():
                    continue
                for row in _iter_jsonl(review_file):
                    if not isinstance(row, dict):
                        continue
                    qid = row.get("question_id")
                    if qid is None:
                        continue
                    records.setdefault(str(qid), {})[rollout_idx] = {
                        "is_correct": row.get("is_correct"),
                        "judge_metrics": row.get("judge_metrics") or {},
                    }
        return records

    def _seed_live_metrics_from_completed_records(
        self,
        completed_records: dict[str, dict[int, dict[str, Any]]],
    ) -> None:
        """Include resumed records in the live accuracy/F1 postfix."""

        self.correct = 0
        self.incorrect = 0
        self.metric_count = 0
        self.metric_sums = {}
        for rollout_records in completed_records.values():
            for record in rollout_records.values():
                if record.get("is_correct") is True:
                    self.correct += 1
                else:
                    self.incorrect += 1
                if not self.live_metric_keys:
                    continue
                self.metric_count += 1
                metrics = record.get("judge_metrics")
                if not isinstance(metrics, dict):
                    metrics = {}
                for key in self.live_metric_keys:
                    self.metric_sums[key] = (
                        self.metric_sums.get(key, 0.0)
                        + self._finite_metric(metrics.get(key))
                    )

    @staticmethod
    def _retry_memory_index_path(rollout_dir: Path, question_id: str) -> Path:
        return rollout_dir / "retry_history" / f"question_{question_id}" / "memory_index.jsonl"

    @classmethod
    def _load_retry_attempt_count(cls, rollout_dir: Path, question_id: str) -> int:
        attempts: set[int] = set()
        retry_root = rollout_dir / "retry_history" / f"question_{question_id}"
        if retry_root.is_dir():
            for attempt_dir in retry_root.glob("attempt_*"):
                if not attempt_dir.is_dir():
                    continue
                attempt_id = attempt_dir.name.removeprefix("attempt_").split("_", 1)[0]
                if attempt_id.isdigit():
                    attempts.add(int(attempt_id))

        index_path = cls._retry_memory_index_path(rollout_dir, question_id)
        if index_path.is_file():
            for row in _iter_jsonl(index_path):
                if not isinstance(row, dict):
                    continue
                try:
                    attempts.add(int(row.get("retry_attempt", 0)))
                except (TypeError, ValueError):
                    continue

        if not attempts:
            return 0
        return max(attempts) + 1

    @classmethod
    def _load_retry_memories(cls, rollout_dir: Path, question_id: str) -> list[dict[str, Any]]:
        index_path = cls._retry_memory_index_path(rollout_dir, question_id)
        if not index_path.is_file():
            return []

        rows = [
            row
            for row in _iter_jsonl(index_path)
            if isinstance(row, dict) and row.get("memory_id")
        ]

        return sorted(
            rows,
            key=lambda r: (
                int(r.get("retry_attempt", 0)),
                int(r.get("memory_order", 0)),
                str(r.get("memory_id", "")),
            ),
        )
