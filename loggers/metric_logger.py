# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union


# Set DISABLE_METRIC_LOGGER=1 (or true/yes/on) to turn the logger into a no-op:
# nothing is recorded and no report file is written.
_DISABLE_ENV_VAR = "DISABLE_METRIC_LOGGER"


def _metric_logging_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


class MetricType(Enum):
    """How a metric is typed and aggregated within a single unit."""

    INT = "int"      # SUM aggregation, reported as int
    FLOAT = "float"  # SUM aggregation, reported as float
    RATE = "rate"    # log 0/1 (or bool) per call -> reported as sum/count ratio
    MEAN = "mean"    # log a value per call -> reported as average (e.g. latency)


_SUM_TYPES = (MetricType.INT, MetricType.FLOAT)
_MEAN_TYPES = (MetricType.RATE, MetricType.MEAN)

UnitKey = Tuple[str, int]
Number = Union[int, float, bool]


@dataclass
class _Accumulator:
    """Accumulates the values added to a single (unit, metric) pair."""

    metric_type: MetricType
    count: int = 0
    total: float = 0.0
    min: float = math.inf
    max: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    @property
    def value(self) -> float:
        """The single scalar this accumulator contributes to aggregation."""
        if self.metric_type is MetricType.INT:
            return int(self.total)
        if self.metric_type is MetricType.FLOAT:
            return float(self.total)
        # RATE / MEAN -> average
        return self.total / self.count if self.count else 0.0


class MetricLogger:
    """Process-wide accumulator of per-(question_id, rollout_idx) metrics."""

    # Exposed for the ``metric_logger.INT`` style ergonomics.
    INT = MetricType.INT
    FLOAT = MetricType.FLOAT
    RATE = MetricType.RATE
    MEAN = MetricType.MEAN

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # (question_id, rollout_idx) -> metric_name -> _Accumulator
        self._units: Dict[UnitKey, Dict[str, _Accumulator]] = {}
        # metric_name -> MetricType, to detect conflicting type usage.
        self._types: Dict[str, MetricType] = {}
        # When disabled, recording and report writing become no-ops.
        self._disabled = _metric_logging_disabled()

    @property
    def enabled(self) -> bool:
        return not self._disabled

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def metric_add(
        self,
        question_id: str,
        metric_name: str,
        metric_type: MetricType,
        value: Number,
        rollout_idx: int = 0,
    ) -> None:
        """Add ``value`` to the accumulator for one unit/metric.

        Repeated calls accumulate (sum) into the same unit. ``RATE``/``MEAN``
        additionally track the call count so a ratio/average can be derived.
        """
        if self._disabled:
            return

        if not isinstance(metric_type, MetricType):
            raise TypeError(
                f"metric_type must be a MetricType, got {type(metric_type).__name__}"
            )

        numeric = float(value)
        key: UnitKey = (str(question_id), int(rollout_idx))

        with self._lock:
            registered = self._types.get(metric_name)
            if registered is None:
                self._types[metric_name] = metric_type
            elif registered is not metric_type:
                raise ValueError(
                    f"Metric '{metric_name}' already registered as "
                    f"{registered.name}, cannot re-log as {metric_type.name}."
                )

            unit = self._units.setdefault(key, {})
            acc = unit.get(metric_name)
            if acc is None:
                acc = _Accumulator(metric_type=metric_type)
                unit[metric_name] = acc
            acc.add(numeric)

    def incr(
        self,
        question_id: str,
        metric_name: str,
        value: Number = 1,
        rollout_idx: int = 0,
    ) -> None:
        """Increment an integer counter (e.g. number of tool calls)."""
        self.metric_add(question_id, metric_name, MetricType.INT, value, rollout_idx)

    def rate(
        self,
        question_id: str,
        metric_name: str,
        hit: bool,
        rollout_idx: int = 0,
    ) -> None:
        """Record one observation for a 0..1 rate metric (e.g. error rate)."""
        self.metric_add(
            question_id, metric_name, MetricType.RATE, 1.0 if hit else 0.0, rollout_idx
        )

    def observe(
        self,
        question_id: str,
        metric_name: str,
        value: Number,
        rollout_idx: int = 0,
    ) -> None:
        """Record one sample for an averaged metric (e.g. latency in ms)."""
        self.metric_add(question_id, metric_name, MetricType.MEAN, value, rollout_idx)

    @contextmanager
    def track_time(
        self,
        question_id: str,
        metric_name: str,
        rollout_idx: int = 0,
    ) -> Iterator[None]:
        """Context manager recording wall-clock elapsed milliseconds as MEAN."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe(question_id, metric_name, elapsed_ms, rollout_idx)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        """Build the hierarchical report dict (does not touch disk)."""
        with self._lock:
            # question_id -> metric_name -> list of per-unit values
            per_question_values: Dict[str, Dict[str, list]] = {}
            # question_id -> rollout_idx -> metric_name -> value
            per_question_rollouts: Dict[str, Dict[int, Dict[str, float]]] = {}

            for (qid, rollout_idx), metrics in self._units.items():
                q_values = per_question_values.setdefault(qid, {})
                q_rollouts = per_question_rollouts.setdefault(qid, {})
                rollout_view = q_rollouts.setdefault(rollout_idx, {})
                for name, acc in metrics.items():
                    value = acc.value
                    q_values.setdefault(name, []).append(value)
                    rollout_view[name] = value

            per_question: Dict[str, Any] = {}
            # metric_name -> list of per-question values (for global mean)
            global_values: Dict[str, list] = {}

            for qid in sorted(per_question_values):
                q_section: Dict[str, Any] = {}
                for name in sorted(per_question_values[qid]):
                    values = per_question_values[qid][name]
                    q_value = sum(values) / len(values) if values else 0.0
                    q_section[name] = q_value
                    global_values.setdefault(name, []).append(q_value)

                rollouts_section: Dict[str, Any] = {}
                for rollout_idx in sorted(per_question_rollouts[qid]):
                    rollouts_section[str(rollout_idx)] = per_question_rollouts[qid][rollout_idx]
                q_section["rollouts"] = rollouts_section
                per_question[qid] = q_section

            summary: Dict[str, Any] = {}
            for name in sorted(self._types):
                values = global_values.get(name, [])
                num_units = sum(
                    1 for metrics in self._units.values() if name in metrics
                )
                summary[name] = {
                    "type": self._types[name].name,
                    "global_mean": (sum(values) / len(values)) if values else 0.0,
                    "num_questions": len(values),
                    "num_units": num_units,
                }

            return {"summary": summary, "per_question": per_question}

    def write_report(self, path: Union[str, Path]) -> Path:
        """Write :meth:`report` as JSON to ``path`` and return the path.

        No file is written when the logger is disabled.
        """
        path = Path(path)
        if self._disabled:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.report()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def reset(self) -> None:
        """Clear all accumulated state."""
        with self._lock:
            self._units.clear()
            self._types.clear()


# Process-wide singleton and thin module-level helpers.
metric_logger = MetricLogger()


def metric_add(
    question_id: str,
    metric_name: str,
    metric_type: MetricType,
    value: Number,
    rollout_idx: int = 0,
) -> None:
    metric_logger.metric_add(question_id, metric_name, metric_type, value, rollout_idx)


def incr(question_id: str, metric_name: str, value: Number = 1, rollout_idx: int = 0) -> None:
    metric_logger.incr(question_id, metric_name, value, rollout_idx)


def rate(question_id: str, metric_name: str, hit: bool, rollout_idx: int = 0) -> None:
    metric_logger.rate(question_id, metric_name, hit, rollout_idx)


def observe(question_id: str, metric_name: str, value: Number, rollout_idx: int = 0) -> None:
    metric_logger.observe(question_id, metric_name, value, rollout_idx)


def report() -> Dict[str, Any]:
    return metric_logger.report()


def write_report(path: Union[str, Path]) -> Path:
    return metric_logger.write_report(path)


def reset() -> None:
    metric_logger.reset()
