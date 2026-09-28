# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Global logging setup: colorful console output for every logger that propagates to root."""

from __future__ import annotations

import logging
import sys

_RESET = "\033[0m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",      # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[1;31m", # bold red
}

_CONFIGURED = False


class LevelColorFormatter(logging.Formatter):
    """Color the full log line by record level."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = _LEVEL_COLORS.get(record.levelno)
        if color is None:
            return message
        return f"{color}{message}{_RESET}"


def setup_logging(
    level: int = logging.WARNING,
    *,
    fmt: str = "[%(asctime)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    stream=None,
    force: bool = False,
) -> None:
    """Attach a colored StreamHandler to the root logger (idempotent).

    After this runs, every ``logging.getLogger(...)`` that propagates to root
    emits colored console lines prefixed with ``[YYYY-MM-DD HH:MM:SS]``.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(level)

    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(LevelColorFormatter(fmt, datefmt=datefmt))
    root.addHandler(handler)

    _CONFIGURED = True
