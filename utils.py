# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from functools import wraps
from logging import getLogger
import time
from constants import RunContext
from typing import Any, Awaitable, Callable, Union
import json

logger = getLogger(__name__)


def naive_token_estimator(messages: list[dict[str, Any]], **kwargs) -> int:
    """Rough token count using the ~4-chars-per-token heuristic.

    Sums the character length of every text-bearing field across all messages
    (content, reasoning_content, and any tool-call name + JSON-serialized
    arguments), then returns total_chars // 4.
    """
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get("content") or "")
        if msg.get("reasoning_content"):
            total_chars += len(msg["reasoning_content"])
        for tc in msg.get("tool_calls") or []:
            # Support both the flat internal form ({"name", "arguments"}) and the
            # serialized OpenAI form ({"function": {"name", "arguments"}}).
            fn = tc.get("function", tc)
            total_chars += len(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                total_chars += len(args)
            elif args is not None:
                total_chars += len(json.dumps(args, ensure_ascii=False))
    return total_chars // 4

def api_response_usage(context: RunContext, **kwargs):
    return context.trajectory.current_num_tokens_api

def tokenizer_estimator(tokenizer_path: str):
    from transformers import AutoTokenizer, PreTrainedTokenizer
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def parse_args(args: Any) -> dict[str, Any]:
        """Helper to safely parse tool call arguments into a dictionary."""
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            # Add fallback here, we don't want the token estimation fail
            # but in the main logic, we record the raw string without fallback
            try:
                parsed = json.loads(args)
                return parsed if isinstance(parsed, dict) else {"_raw": args}
            except (json.JSONDecodeError, ValueError):
                return {"_raw": args}
        return {}

    def estimate(messages: list[dict[str, Any]], **kwargs) -> int:
        sanitized_messages: list[dict[str, Any]] = []

        for msg in messages:
            if not isinstance(msg, dict):
                continue

            msg_copy = dict(msg)
            
            # Use the walrus operator (:=) to assign and check in one step
            if isinstance(tool_calls := msg_copy.get("tool_calls"), list):
                normalized_calls: list[dict[str, Any]] = []
                
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue

                    tc_copy = dict(tc)
                    
                    if isinstance(fn := tc_copy.get("function"), dict):
                        # Dictionary unpacking creates a clean shallow copy with overrides
                        tc_copy["function"] = {
                            **fn, 
                            "arguments": parse_args(fn.get("arguments"))
                        }

                    normalized_calls.append(tc_copy)

                msg_copy["tool_calls"] = normalized_calls

            sanitized_messages.append(msg_copy)

        result = tokenizer.apply_chat_template(
            sanitized_messages, 
            add_generation_prompt=False,
            tools=kwargs.get("tools", None)
        )
        # For different Transformers version
        return len(result) if isinstance(result, list) else len(result["input_ids"])

    return estimate


def retry_on_exception(retry_times: int, delay_seconds: float = 0):
    if retry_times < 1:
        raise ValueError("retry_times must be at least 1")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be greater than or equal to 0")

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapped(*args, **kwargs):
                last_exception = None
                for attempt in range(retry_times):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:  # pylint: disable=broad-except
                        last_exception = exc
                        logger.warning(
                            f"[{func.__name__}] Failed:\n{exc}\nRetry {attempt}/{retry_times}"
                        )
                        await asyncio.sleep(delay_seconds)
                raise last_exception

            return async_wrapped

        @wraps(func)
        def wrapped(*args, **kwargs):
            last_exception = None
            for attempt in range(retry_times):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # pylint: disable=broad-except
                    last_exception = exc
                    logger.warning(
                        f"[{func.__name__}] Failed:\n{exc}\nRetry {attempt}/{retry_times}"
                    )
                    time.sleep(delay_seconds)
            raise last_exception

        return wrapped

    return decorator
