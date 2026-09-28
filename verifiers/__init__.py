# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto


class JudgeCompleteStatus(Enum):
    SUCCESS = auto()
    EXTRACTION_FAILED = auto()


@dataclass
class JudgeResult:
    """Structured verdict returned by every verifier."""

    is_correct: bool
    complete_status: JudgeCompleteStatus
    score: float = 0.0 # 0.0 – 1.0
    reasoning: str = ""
    raw_response: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class BaseVerifier(ABC):
    """Abstract base for all verifiers."""

    name: str = "base_verifier"

    @abstractmethod
    async def async_judge(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Registry  (mirrors tools/__init__.py)
# ---------------------------------------------------------------------------

_REGISTERED_VERIFIERS: dict[str, type[BaseVerifier]] = {}
_VERIFIER_MODULES_LOADED = False


def register_verifier(name: str | None = None):
    """Class decorator to register a verifier under a given name."""

    def decorator(cls: type[BaseVerifier]) -> type[BaseVerifier]:
        verifier_name = name or getattr(cls, "name", None) or cls.__name__.lower()
        if verifier_name in _REGISTERED_VERIFIERS:
            raise ValueError(f"Duplicate verifier name: {verifier_name!r}")
        _REGISTERED_VERIFIERS[verifier_name] = cls
        return cls

    return decorator


def _load_verifier_modules() -> None:
    """Lazily import all *_judge.py modules so their @register_verifier decorators run."""
    global _VERIFIER_MODULES_LOADED
    if _VERIFIER_MODULES_LOADED:
        return
    for module_info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        if not module_name.endswith("_judge"):
            continue
        importlib.import_module(f"{__name__}.{module_name}")
    _VERIFIER_MODULES_LOADED = True


def get_verifier_cls(name: str) -> type[BaseVerifier]:
    """Return the raw verifier class without requiring prior initialisation."""
    _load_verifier_modules()
    cls = _REGISTERED_VERIFIERS.get(name)
    if cls is None:
        available = ", ".join(_REGISTERED_VERIFIERS) or "none"
        raise ValueError(
            f"Verifier {name!r} not found. "
            f"Registered verifiers: {available}"
        )
    return cls
