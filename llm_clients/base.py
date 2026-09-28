# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


@dataclass
class TokenUsage:
    """Per-call token accounting from an OpenAI-compatible response."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class LLMCallResult:
    """Normalized result returned by `BaseLLMClient.call_llm`.

    `completion` is the raw OpenAI SDK object.
    `raw_message` is the dict form of the assistant message, populated only
    when we successfully dumped it (used for reasoning-content fallbacks).
    """
    completion: Any = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    reasoning_content: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw_message: Optional[Dict[str, Any]] = None


def get_first_env(*keys: str, default: Any = None) -> Any:
    """
    Return the first truthy env value in keys, else default.
    """
    return next((value for key in keys if (value := os.getenv(key))), default)


_RETRY_STATS_LOCK = threading.Lock()
_RETRY_RATE_LIMIT_HITS = 0
_RETRY_TOTAL = 0

_RATE_LIMIT_MARKERS = (
    "qpm limit",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "quota",
    "throttl",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def log_retry_attempt(retry_state: Any) -> None:
    """tenacity ``before_sleep`` hook: surface retryable failures, esp. 429/QPM.

    Gated by env ``LLM_RETRY_LOG`` (set it to 1 in the run script). Maintains a
    process-wide tally so you can gauge throttling pressure live: a high and
    growing ``rate_limit_hits`` means concurrency is above the API ceiling.
    """
    if os.getenv("LLM_RETRY_LOG", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }:
        return
    outcome = getattr(retry_state, "outcome", None)
    exc = outcome.exception() if outcome is not None else None
    if exc is None:
        return
    is_rate_limit = _is_rate_limit_error(exc)
    global _RETRY_RATE_LIMIT_HITS, _RETRY_TOTAL
    with _RETRY_STATS_LOCK:
        _RETRY_TOTAL += 1
        if is_rate_limit:
            _RETRY_RATE_LIMIT_HITS += 1
        rate_hits, total = _RETRY_RATE_LIMIT_HITS, _RETRY_TOTAL
    next_action = getattr(retry_state, "next_action", None)
    wait_s = getattr(next_action, "sleep", 0.0) if next_action is not None else 0.0
    tag = "RATE_LIMIT/429" if is_rate_limit else "retryable"
    logging.getLogger("llm_clients.retry").warning(
        "[%s] attempt=%d wait=%.1fs rate_limit_hits=%d total_retries=%d :: %s",
        tag,
        getattr(retry_state, "attempt_number", -1),
        wait_s,
        rate_hits,
        total,
        str(exc).replace("\n", " ")[:200],
    )


class BaseLLMClient(ABC):
    """
    Shared request, response, and retry handling for OpenAI Chat Completions.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        extra_body_overrides: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.model_name = (
            model_name or os.getenv("OPENAI_MODEL") or os.getenv("MODEL_NAME")
        )
        if not self.model_name:
            raise ValueError(
                "model_name is required; pass it explicitly or set OPENAI_MODEL."
            )
        self._client_timeout = timeout
        self.max_tokens_limit: Optional[int] = None
        self.client: Any = None

        self.extra_body_defaults = self._load_extra_body_defaults()
        if extra_body_overrides:
            self.extra_body_defaults.update(extra_body_overrides)

        self._init_client()
        self.logger.info(
            "Initialized OpenAI-compatible client with model=%s", self.model_name
        )

    @abstractmethod
    def _init_client(self) -> None:
        """Initialize the OpenAI SDK client and server limits."""

    def _prepare_extra_body(
        self,
        *,
        extra_body: Dict[str, Any],
        include_thoughts: bool,
        chat_id: Optional[str],
    ) -> Dict[str, Any]:
        """Add OpenAI-compatible server extensions to ``extra_body``."""
        return extra_body

    @staticmethod
    def _parse_env_value(raw_value: str) -> Any:
        if raw_value is None:
            return raw_value
        value = raw_value.strip()
        if value == "":
            return raw_value
        try:
            return json.loads(value)
        except Exception:
            return raw_value

    def _load_extra_body_defaults(self) -> Dict[str, Any]:
        extra_body: Dict[str, Any] = {}
        prefix = "LLM_EXTRA_BODY_"
        for key, raw_value in os.environ.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix) :].strip().lower()
            if not suffix:
                continue
            extra_body[suffix] = self._parse_env_value(raw_value)
        return extra_body

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        msg = str(exc).lower()
        exc_type = type(exc).__name__.lower()
        non_retry_keywords = [
            "maximum context length",
            "context length",
            "context window",
            "prompt too long",
            "too many tokens",
            "input is too long",
            "request too large",
            "token limit",
            "data_inspection_failed",
            "inappropriate content",
            "input data may contain inappropriate content",
            "-4316",
        ]
        if any(k in msg for k in non_retry_keywords):
            return False

        retry_keywords = [
            "qpm limit",
            "quota",
            "rate limit",
            "rate_limit",
            "too many requests",
            "429",
            "temporarily unavailable",
            "service unavailable",
            "503",
            "502",
            "504",
            "500",
            "connection aborted",
            "connection reset",
            "connection refused",
            "connection error",
            "connection timeout",
            "timeout",
            "timed out",
            "read timeout",
            "connect timeout",
            "request timeout",
            "network error",
            "network unreachable",
            "ssl error",
            "eof occurred",
            "internal server error",
            "bad gateway",
            "gateway timeout",
        ]
        retry_exception_types = [
            "timeout",
            "connectionerror",
            "connecttimeout",
            "readtimeout",
        ]
        return any(k in msg for k in retry_keywords) or any(
            t in exc_type for t in retry_exception_types
        )

    def _capped_max_tokens(self, max_tokens: int) -> int:
        if self.max_tokens_limit is not None and max_tokens > self.max_tokens_limit:
            self.logger.debug(
                "Requested max_tokens=%s exceeds the configured limit %s; capping.",
                max_tokens,
                self.max_tokens_limit,
            )
            return self.max_tokens_limit
        return max_tokens

    @classmethod
    def _extract_reasoning_tokens(cls, usage: Any) -> Optional[int]:
        """
        Extract reasoning token count from OpenAI-compatible usage schemas.
        """
        if not usage:
            return None

        direct = cls._read_field(usage, "reasoning_tokens")
        if direct is not None:
            return direct

        details_candidates = [
            cls._read_field(usage, "completion_tokens_details"),
            cls._read_field(usage, "output_tokens_details"),
            cls._read_field(usage, "reasoning_tokens_details"),
        ]
        for details in details_candidates:
            if not details:
                continue
            nested = cls._read_field(details, "reasoning_tokens")
            if nested is not None:
                return nested

        return None

    @staticmethod
    def _object_to_dict(obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            try:
                dumped = obj.model_dump()
                return dumped if isinstance(dumped, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _read_field(obj: Any, field: str) -> Any:
        if obj is None:
            return None
        value = getattr(obj, field, None)
        if value is not None:
            return value
        if isinstance(obj, dict):
            return obj.get(field)
        return None

    @staticmethod
    def _extract_reasoning_from_dict(payload: Dict[str, Any]) -> Any:
        return payload.get("reasoning_content") or payload.get("reasoning")

    @staticmethod
    def _build_usage(usage: Any) -> TokenUsage:
        if not usage:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=BaseLLMClient._extract_reasoning_tokens(usage),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def _build_result(self, completion: Any) -> LLMCallResult:
        message = completion.choices[0].message if completion.choices else None
        content = self._read_field(message, "content")
        tool_calls = self._read_field(message, "tool_calls")
        reasoning_content = self._read_field(message, "reasoning_content")
        raw_message_dict = self._object_to_dict(message)

        if reasoning_content is None and raw_message_dict:
            reasoning_content = self._extract_reasoning_from_dict(raw_message_dict)

        if reasoning_content is None and hasattr(completion, "model_dump"):
            try:
                completion_dict = completion.model_dump()
                if isinstance(completion_dict, dict):
                    first_choice = completion_dict.get("choices", [{}])[0] or {}
                    msg_dict = first_choice.get("message", {}) or {}
                    raw_message_dict = raw_message_dict or msg_dict
                    reasoning_content = self._extract_reasoning_from_dict(msg_dict)
            except Exception:
                pass

        finish_reason = (
            completion.choices[0].finish_reason if completion.choices else None
        )
        return LLMCallResult(
            completion=completion,
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            usage=self._build_usage(getattr(completion, "usage", None)),
            raw_message=raw_message_dict or None,
        )

    def _build_request_params(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 65536,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        include_thoughts: bool = False,
        chat_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        request_params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._capped_max_tokens(max_tokens),
        }
        if top_p is not None:
            request_params["top_p"] = top_p
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if kwargs.get("stop") is not None:
            request_params["stop"] = kwargs["stop"]

        if tools:
            request_params["tools"] = tools
            if tool_choice:
                request_params["tool_choice"] = tool_choice

        extra_body = dict(self.extra_body_defaults)
        extra_body.update(kwargs.get("extra_body", {}))
        # Non-OpenAI sampling extensions. The OpenAI Python SDK sends
        # `extra_body` keys as extra top-level JSON fields, which is accepted
        # by both vLLM and SGLang OpenAI-compatible servers. Explicit function
        # arguments intentionally override env/default extra_body values.
        if top_k is not None:
            extra_body["top_k"] = top_k
        if min_p is not None:
            extra_body["min_p"] = min_p
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty
        extra_body = self._prepare_extra_body(
            extra_body=extra_body,
            include_thoughts=include_thoughts,
            chat_id=chat_id,
        )
        if extra_body:
            request_params["extra_body"] = extra_body

        extra_headers = dict(kwargs.get("extra_headers", {}))
        if extra_headers:
            request_params["extra_headers"] = extra_headers
        return request_params

    @retry(
        retry=retry_if_exception(_should_retry),
        wait=wait_random_exponential(
            multiplier=float(os.getenv("LLM_RETRY_EXP_MULTIPLIER", "1")),
            max=float(os.getenv("LLM_RETRY_EXP_MAX_SECONDS", "60")),
        ),
        stop=stop_after_attempt(int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "10"))),
        before_sleep=log_retry_attempt,
    )
    async def _invoke_completion(
        self,
        request_params: Dict[str, Any],
    ) -> LLMCallResult:
        completion = await self.client.chat.completions.create(**request_params)
        return self._build_result(completion)

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 65536,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        include_thoughts: bool = False,
        chat_id: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMCallResult:
        request_params = self._build_request_params(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            tools=tools,
            tool_choice=tool_choice,
            include_thoughts=include_thoughts,
            chat_id=chat_id,
            **kwargs,
        )
        return await self._invoke_completion(request_params)

    async def aclose(self) -> None:
        if self.client is None:
            return
        close = getattr(self.client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result

    def get_max_tokens_limit(self) -> Optional[int]:
        return self.max_tokens_limit
