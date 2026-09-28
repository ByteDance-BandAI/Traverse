# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Adapters for user-provided search and link-summary functions.

The framework owns the Agent-facing schemas and result normalization. Deployments
own the actual network integrations and provide them as importable callables.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agents.base_agent_suit import RunContext
from constants import ToolCall
from tools import tool_registry
from tools.base import (
    BaseTool,
    ReturnableToolError,
    ToolBackendUnavailable,
    ToolSchema,
)

logger = logging.getLogger(__name__)


def load_function(function_spec: str) -> Callable[..., Any]:
    """Load ``module:function`` or ``/path/to/file.py:function``.

    Attribute paths may be nested, for example
    ``my_package.integrations:search.client.search``.
    """
    module_ref, separator, attribute_path = str(function_spec or "").rpartition(":")
    if not separator or not module_ref.strip() or not attribute_path.strip():
        raise ValueError(
            "Function references must use 'module:function' or "
            "'/path/to/file.py:function'."
        )

    module_ref = module_ref.strip()
    attribute_path = attribute_path.strip()
    is_file_reference = (
        module_ref.endswith(".py")
        or "/" in module_ref
        or "\\" in module_ref
    )

    if is_file_reference:
        source_path = Path(module_ref).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Custom tool module does not exist: {source_path}")
        digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:16]
        module_name = f"_custom_tool_{digest}"
        module = sys.modules.get(module_name)
        if module is None:
            module_spec = importlib.util.spec_from_file_location(module_name, source_path)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"Cannot import custom tool module: {source_path}")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            try:
                module_spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
    else:
        module = importlib.import_module(module_ref)

    loaded: Any = module
    for part in attribute_path.split("."):
        loaded = getattr(loaded, part)
    if not callable(loaded):
        raise TypeError(f"Loaded object is not callable: {function_spec}")
    return loaded


def _decode_function_output(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class LoadedFunctionTool(BaseTool):
    """Base class that loads and safely invokes a user-owned callable."""

    def __init__(
        self,
        *,
        name: str,
        function_path: str | None,
        function_cli_flag: str,
        max_attempts: int,
        request_timeout_seconds: float | None,
        function_kwargs: dict[str, Any] | None,
        custom_tool_prompt: str,
    ) -> None:
        super().__init__(name=name, custom_tool_prompt=custom_tool_prompt)
        if not function_path:
            raise ToolBackendUnavailable(
                f"{name} needs a custom function. Provide {function_cli_flag} "
                "with module:function or /path/to/file.py:function."
            )
        try:
            self.function = load_function(function_path)
        except Exception as exc:
            raise ToolBackendUnavailable(
                f"Could not load {name} function {function_path!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self.max_attempts = max(1, int(max_attempts))
        self.request_timeout_seconds = request_timeout_seconds
        self.function_kwargs = dict(function_kwargs or {})

    async def _invoke(self, **request_kwargs: Any) -> Any:
        call_kwargs = {**self.function_kwargs, **request_kwargs}
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await self._call_once(call_kwargs)
                return _decode_function_output(result)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    logger.warning(
                        "%s custom function failed; retrying (%d/%d): %s",
                        self.name,
                        attempt,
                        self.max_attempts,
                        exc,
                    )

        assert last_error is not None
        raise ReturnableToolError(
            f"{self.name} failed after {self.max_attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    async def _call_once(self, call_kwargs: dict[str, Any]) -> Any:
        async def invoke() -> Any:
            if inspect.iscoroutinefunction(self.function):
                return await self.function(**call_kwargs)
            result = await asyncio.to_thread(self.function, **call_kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

        timeout = self.request_timeout_seconds
        if timeout is None or timeout <= 0:
            return await invoke()
        return await asyncio.wait_for(invoke(), timeout=timeout)


@tool_registry.register_tool(name="search_api")
class FunctionSearchTool(LoadedFunctionTool):
    """Agent-facing search tool backed by a user-provided callable."""

    def __init__(
        self,
        function_path: str | None = None,
        function_kwargs: dict[str, Any] | None = None,
        keys_to_keep: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        num: int = 10,
        max_attempts: int = 3,
        request_timeout_seconds: float | None = 120.0,
        drop_query_echo: bool = True,
        echo_title_overlap: float = 0.9,
        echo_max_desc_len: int = 30,
    ) -> None:
        super().__init__(
            name="search_api",
            function_path=function_path,
            function_cli_flag="--search_tool_function",
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
            function_kwargs=function_kwargs,
            custom_tool_prompt="search the web for up-to-date, accurate information",
        )
        self.keys_to_keep = (
            ["title", "url", "description"]
            if keys_to_keep is None
            else keys_to_keep
        )
        self.blocked_domains = [
            domain.lower().strip().strip(".")
            for domain in (blocked_domains or [])
            if domain and domain.strip()
        ]
        self.num = max(1, int(num))
        self.drop_query_echo = drop_query_echo
        self.echo_title_overlap = echo_title_overlap
        self.echo_max_desc_len = echo_max_desc_len
        self.schema = (
            ToolSchema(
                name="search_api",
                description="Search the web and return relevant results.",
            )
            .add_string(
                name="query",
                description="Search query keywords or question",
                required=True,
            )
            .add_number(
                name="return_n",
                description="Number of results to return",
                is_integer=True,
                default=self.num,
                required=False,
            )
            .build(wrap_in_function=True)
        )

    async def run(self, payload: ToolCall, context: RunContext) -> str:
        query = payload.arguments["query"]
        requested_num = max(1, int(payload.arguments.get("return_n") or self.num))
        raw_result = await self._invoke(query=query, num=requested_num)
        response = self._normalize_search_response(raw_result)
        items = response["return"]

        if isinstance(items, list):
            normalized_items = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                normalized = dict(item)
                if "description" not in normalized and "snippet" in normalized:
                    normalized["description"] = normalized["snippet"]
                if self._is_blocked_url(normalized.get("url")):
                    continue
                if self.drop_query_echo and self._is_query_echo(normalized, query):
                    continue
                normalized_items.append(
                    {
                        key: normalized[key]
                        for key in self.keys_to_keep
                        if key in normalized
                    }
                    if self.keys_to_keep
                    else normalized
                )
            response["return"] = normalized_items[:requested_num]
            if not response["return"]:
                response["status"] = "NO_RESULT"

        return json.dumps(response, ensure_ascii=False, default=str)

    @staticmethod
    def _normalize_search_response(result: Any) -> dict[str, Any]:
        if isinstance(result, Mapping):
            response = dict(result)
            if "return" not in response:
                if "results" in response:
                    response["return"] = response.pop("results")
                else:
                    status = response.pop("status", None)
                    response = {"return": response}
                    if status is not None:
                        response["status"] = status
        else:
            response = {"return": result}
        response.setdefault(
            "status",
            "SUCCESS" if response.get("return") else "NO_RESULT",
        )
        return response

    def _is_blocked_url(self, url: Any) -> bool:
        if not isinstance(url, str) or not self.blocked_domains:
            return False
        host = (urlparse(url).hostname or "").lower().strip(".")
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.blocked_domains
        )

    @staticmethod
    def _tokenize(text: Any) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(text).lower()))

    def _is_query_echo(self, item: Mapping[str, Any], query: str) -> bool:
        description = str(item.get("description") or item.get("snippet") or "")
        if len(description.strip()) >= self.echo_max_desc_len:
            return False
        query_tokens = self._tokenize(query)
        title_tokens = self._tokenize(item.get("title", ""))
        if not query_tokens or not title_tokens:
            return False
        overlap = len(query_tokens & title_tokens) / len(query_tokens)
        return overlap >= self.echo_title_overlap


@tool_registry.register_tool(name="link_summary_tool")
class FunctionLinkSummaryTool(LoadedFunctionTool):
    """Agent-facing URL summarizer backed by a user-provided callable."""

    def __init__(
        self,
        function_path: str | None = None,
        function_kwargs: dict[str, Any] | None = None,
        max_attempts: int = 3,
        request_timeout_seconds: float | None = 120.0,
    ) -> None:
        super().__init__(
            name="link_summary_tool",
            function_path=function_path,
            function_cli_flag="--link_summary_tool_function",
            max_attempts=max_attempts,
            request_timeout_seconds=request_timeout_seconds,
            function_kwargs=function_kwargs,
            custom_tool_prompt="summarize URL content relevant to the current task",
        )
        self.schema = (
            ToolSchema(
                name="link_summary_tool",
                description="Read and summarize one or more URLs for a question.",
            )
            .add_string(
                name="question",
                description="Question or summarization requirement",
                required=True,
            )
            .add_one_of(
                name="url",
                description="A URL or list of URLs to summarize",
                required=True,
                options=[
                    {"type": "string", "description": "A single URL"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Multiple URLs",
                    },
                ],
            )
            .build(wrap_in_function=True)
        )

    async def run(self, payload: ToolCall, context: RunContext) -> str:
        result = await self._invoke(
            question=payload.arguments["question"],
            url=payload.arguments["url"],
        )
        if result is None:
            raise ReturnableToolError("link_summary_tool returned no result.")
        if isinstance(result, Mapping):
            response = dict(result)
            if "return" not in response:
                status = response.pop("status", None)
                response = {"return": response}
                if status is not None:
                    response["status"] = status
        else:
            response = {"return": result}
        response.setdefault("status", "SUCCESS")
        return json.dumps(response, ensure_ascii=False, default=str)
