# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import Any, Awaitable, Callable
import asyncio

DEFAULT_PROMPT_SET_NAME = "default"
SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}
_PROMPT_BUILDERS: dict[str, "BasePromptBuilder"] = {}


class BasePromptBuilder(ABC):
    _ROLE_TO_HANDLER: dict[str, str] = {
        "system": "build_system_prompt",
        "user": "build_user_prompt"
    }

    def build_prompt(
        self,
        role: str,
        **kwargs
    ) -> str | Awaitable[str]:
        handler_name = self._ROLE_TO_HANDLER.get(role)
        if handler_name is None:
            supported = ", ".join(sorted(self._ROLE_TO_HANDLER))
            raise ValueError(f"Unsupported role '{role}'. Supported roles: {supported}")

        handler = getattr(self, handler_name)
        return handler(**kwargs)

    def build_system_prompt(
        self,
        **kwargs
    ) -> str | Awaitable[str]:
        raise NotImplementedError(
            "build_system_prompt(...) must be implemented by the prompt builder"
        )

    def build_user_prompt(
        self,
        **kwargs
    ) -> str | Awaitable[str]:
        raise NotImplementedError(
            "build_user_prompt(...) is not implemented for this prompt builder"
        )


def register_prompt_set(
    prompt_set_name: str,
    overwrite: bool = False,
) -> Callable[[type[BasePromptBuilder]], type[BasePromptBuilder]]:
    if not prompt_set_name:
        raise ValueError("Prompt set name cannot be empty")

    def _register_builder(prompt_builder: BasePromptBuilder) -> None:
        if prompt_set_name in _PROMPT_BUILDERS and not overwrite:
            raise ValueError(
                f"Prompt set '{prompt_set_name}' already exists. "
                "Use overwrite=True to replace it."
            )
        _PROMPT_BUILDERS[prompt_set_name] = prompt_builder

    def decorator(target: type[BasePromptBuilder]) -> type[BasePromptBuilder]:
        if not issubclass(target, BasePromptBuilder):
            raise TypeError("Prompt builder class must inherit from BasePromptBuilder")
        _register_builder(target())
        return target

    return decorator


def available_prompt_sets() -> list[str]:
    return sorted(_PROMPT_BUILDERS.keys())

def get_prompt_builder(prompt_set) -> BasePromptBuilder:
    builder = _PROMPT_BUILDERS.get(prompt_set)
    if builder is not None:
        return builder

    available = ", ".join(available_prompt_sets()) or "none"
    raise ValueError(
        f"Prompt set '{prompt_set}' is not registered with a builder. "
        "Register a builder via @register_prompt_set(prompt_set_name). "
        f"Registered prompt sets: {available}"
    )

class PromptFactory:
    def __init__(self, prompt_set: str):
        self.prompt_set = prompt_set
        self.builder = get_prompt_builder(prompt_set)

    @classmethod
    def _validate_role(cls, role: str) -> None:
        if role not in SUPPORTED_ROLES:
            supported = ", ".join(sorted(SUPPORTED_ROLES))
            raise ValueError(f"Unsupported role '{role}'. Supported roles: {supported}")

    def get_prompt(
        self,
        role: str = "system",
        **kwargs
    ) -> str:
        self._validate_role(role)

        result = self.builder.build_prompt(
            role=role,
            **kwargs,
        )
        if not isinstance(result, str):
            raise TypeError("build_prompt(...) must return a string")
        return result
