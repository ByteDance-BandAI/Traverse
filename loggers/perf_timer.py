# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Tuple

# Pull in package-level setup_logging() via loggers/__init__.py when imported as loggers.perf_timer.
logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    total_s: float = 0.0
    count: int = 0
    min_s: float = math.inf
    max_s: float = -math.inf

    def add(self, duration_s: float) -> None:
        self.total_s += duration_s
        self.count += 1
        if duration_s < self.min_s:
            self.min_s = duration_s
        if duration_s > self.max_s:
            self.max_s = duration_s

    def stats_and_reset(self) -> Tuple[float, float, float]:
        mean = self.total_s / self.count
        min_s, max_s = self.min_s, self.max_s
        self.total_s = 0.0
        self.count = 0
        self.min_s = math.inf
        self.max_s = -math.inf
        return mean, min_s, max_s


class PerfTimer:
    WINDOW = 200

    def __init__(self) -> None:
        # Keeping the lock in case you share this instance across multiple OS threads.
        # If strictly single-threaded asyncio, this lock can be safely removed.
        self._lock = threading.Lock()
        self._buckets: Dict[str, _Bucket] = {}

    @contextlib.contextmanager
    def measure(self, name: str):
        """Context manager to safely time a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_s = time.perf_counter() - start
            with self._lock:
                bucket = self._buckets.setdefault(name, _Bucket())
                bucket.add(duration_s)

                if bucket.count >= self.WINDOW:
                    mean, min_s, max_s = bucket.stats_and_reset()
                    label = self._format_label(name)
                    logger.critical(
                        "[Perf] %s: %.3fs (min %.3fs, max %.3fs), averaged %d calls",
                        label,
                        mean,
                        min_s,
                        max_s,
                        self.WINDOW,
                    )

    @staticmethod
    def _format_label(name: str) -> str:
        if name == "api_call":
            return "API_Call"
        if name.startswith("tool."):
            return f"Tool {name[len('tool.'):]}"
        return name


perf_timer = PerfTimer()
