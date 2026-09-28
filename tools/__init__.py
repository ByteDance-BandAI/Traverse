# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import logging
import pkgutil
from typing import Any

from tools.base import (
    BaseTool,
    MemoryTool,
    SubmitToolBase,
    ToolBackendUnavailable,
    ToolCallResult,
)
from constants import ToolCall, RunContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Owns tool registration, lazy module loading, instance initialization,
    schema/prompt lookup, and execution.
    """
    def __init__(self) -> None:
        self._tool_classes: dict[str, type[BaseTool]] = {}
        self._instances: dict[str, BaseTool] = {}
        self._unavailable: dict[str, str] = {}
        self._modules_loaded = False

    def register_tool(self, name: str | None = None):
        """Decorator to explicitly register a tool class."""
        def decorator(tool_cls: type[BaseTool]) -> type[BaseTool]:
            tool_name = name or getattr(tool_cls, "name", None) or tool_cls.__name__.lower()
            if tool_name in self._tool_classes:
                raise ValueError(f"Duplicated tool name detected: {tool_name}")
            self._tool_classes[tool_name] = tool_cls
            return tool_cls
        return decorator

    def _load_tool_modules(self) -> None:
        if self._modules_loaded:
            return

        for info in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
            if info.name != "base" and not info.name.startswith("_") and info.name.endswith("_tools"):
                try:
                    importlib.import_module(f"{__name__}.{info.name}")
                except ImportError as exc:
                    # A tool module whose optional SDK is absent must not take
                    # down every other tool in the package.
                    self._unavailable[info.name] = str(exc)
                    logger.warning("Skipped tool module %s: %s", info.name, exc)
        self._modules_loaded = True

    def init_tools(self, tool_config: dict[str, Any] | None = None) -> None:
        config = tool_config or {}
        self._load_tool_modules()
        for name, tool_cls in self._tool_classes.items():
            try:
                self._instances[name] = tool_cls(**config.get(name, {}))
            except ToolBackendUnavailable as exc:
                self._unavailable[name] = str(exc)
                logger.warning("Tool %s is unavailable: %s", name, exc)

    def _get_tools(self, names: list[str] | None = None) -> dict[str, BaseTool]:
        """Helper to fetch tools and validate they exist."""
        target_names = names if names is not None else list(self._instances.keys())
        if missing := set(target_names) - self._instances.keys():
            blocked = {n: self._unavailable[n] for n in missing if n in self._unavailable}
            if blocked:
                reasons = "; ".join(f"{n}: {why}" for n, why in blocked.items())
                raise ValueError(f"Requested tools are unavailable: {reasons}")
            raise ValueError(
                f"Tools not found: {missing}. Registered tools: {sorted(self._instances)}"
            )
        return {n: self._instances[n] for n in target_names}

    def get_prompt(self, tool_set: list[str] | None = None) -> str:
        return "\n".join(
            f"{i}. {name}: {tool.get_custom_tool_prompt()}"
            for i, (name, tool) in enumerate(self._get_tools(tool_set).items(), 1)
        )

    def get_schema(self, tool_set: list[str] | None = None) -> list:
        return [tool.schema for tool in self._get_tools(tool_set).values()]

    def get_instance(self, name: str | None) -> BaseTool | None:
        return self._instances.get(name.strip()) if name else None

    def is_memory_tool_call(self, tool_call: ToolCall) -> bool:
        return isinstance(self.get_instance(tool_call.name), MemoryTool)

    def is_submit_tool_call(self, tool_call: ToolCall) -> bool:
        return isinstance(self.get_instance(tool_call.name), SubmitToolBase)

    def has_submit_tool(self, tool_names: list[str] | None) -> bool:
        return any(isinstance(self.get_instance(n), SubmitToolBase) for n in (tool_names or []))

    async def _execute_single_tool_call(self, tool_call: ToolCall, context: RunContext) -> ToolCallResult:
        name = (tool_call.name or "").strip()
        if not name:
            return ToolCallResult(ok=False, error="Received a tool call with an empty tool name.")
        if isinstance(tool_call.arguments, str):
            return ToolCallResult(ok=False, error="Tool arguments can't be loaded, please check.")

        allowed = context.allowed_tool_names or sorted(self._instances.keys())
        if name not in allowed:
            return ToolCallResult(ok=False, error=f"Unavailable tool {name}, available tools\n{allowed}")

        if not (tool := self._instances.get(name)):
            raise RuntimeError("Tools are not initialized, call init_tools first!")

        return await tool.execute(tool_call, context)

    async def execute(self, tool_calls: list[ToolCall], context: RunContext) -> list[ToolCallResult]:
        """Execute a batch of tool calls concurrently, isolating terminal tools."""
        is_mixed_batch = len(tool_calls) > 1

        async def _resolve(call: ToolCall) -> ToolCallResult:
            is_mem = self.is_memory_tool_call(call)
            is_sub = self.is_submit_tool_call(call)

            if is_mixed_batch and (is_mem or is_sub):
                t_type = "Memory" if is_mem else "Submit"
                return ToolCallResult(
                    ok=False,
                    error=f"{t_type} tool must be called alone. Please retry with only the "
                          f"{t_type.lower()} tool call, or call non-{t_type.lower()} tools in a separate turn."
                )
            return await self._execute_single_tool_call(call, context)

        return await asyncio.gather(*(_resolve(call) for call in tool_calls))


tool_registry = ToolRegistry()
