# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, TYPE_CHECKING
from constants import RuntimeConfig, AgentState, ToolCallResult, ToolCall
import json
from constants import Message, MessageRole
from llm_clients.base import LLMCallResult
import asyncio
from functools import wraps
import inspect
from contextlib import contextmanager
from agents.events import Observation
from logging import getLogger


logger = getLogger(__name__)

if TYPE_CHECKING:
    from agents.self_verify_agent import FSMContext


def handles(state: AgentState):
    """Decorator to mark a method as the handler for a specific AgentState."""
    def decorator(func):
        setattr(func, "_handles_state", state)
        return func
    return decorator


def serialize_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    elif hasattr(item, "dict"):
        return item.dict()
    else:
        return item if isinstance(item, dict) else {}

def _tool_result_base_content(result: ToolCallResult) -> str:
    if result.ok:
        out = result.output
        if isinstance(out, str):
            return out
        return json.dumps(out, ensure_ascii=False)
    return json.dumps({"succeed": False, "error": str(result.error)}, ensure_ascii=False)

def tool_result_to_content(result: ToolCallResult) -> str:
    base = _tool_result_base_content(result)
    if result.extra_info:
        return f"{base}\n{result.extra_info}"
    return base

def _parse_tool_call_arguments(raw_args: Any) -> dict[str, Any] | str:
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError:
            # Preserve the original API string so downstream tool execution
            # can detect the parse failure and surface a returnable error.
            return raw_args
    return {}


def _normalize_tool_call_entry(tool_call: Any) -> tuple[str, str, dict[str, Any] | str]:
    """
    OpenAI-compatible APIs return tool_calls as {id, type, function: {name, arguments}}.
    Some payloads put name/arguments at the top level; support both.
    """
    d = serialize_to_dict(tool_call)
    fn = d.get("function")
    if isinstance(fn, dict):
        name = (fn.get("name") or d.get("name") or "").strip()
        raw_args = fn.get("arguments", d.get("arguments", "{}"))
    else:
        name = (d.get("name") or "").strip()
        raw_args = d.get("arguments", "{}")
    tc_id = str(d.get("id", "") or "")
    return tc_id, name, _parse_tool_call_arguments(raw_args)


def api_response_handler(api_response: LLMCallResult) -> Message:
    # 1. Build Message
    message = Message(
        role=MessageRole.ASSISTANT,
        content=api_response.content or "",
        reasoning_content=api_response.reasoning_content or "",
        tool_calls=[],
    )

    # 2. Build Tool Calls
    tool_calls = api_response.tool_calls or []
    if not tool_calls:
        return message

    for tool_call in tool_calls:
        tc_id, name, parsed_args = _normalize_tool_call_entry(tool_call)
        tool_call_message = ToolCall(
            id=tc_id,
            name=name,
            arguments=parsed_args,
            metadata={},
        )
        message.tool_calls.append(tool_call_message)

    return message


class FSMStateRetryHook:
    def __call__(self, context: "FSMContext", error: Exception):
        raise NotImplementedError("Subclass should implement the hook!")


def with_retry_hook(hook_attr: str, max_retries: int = 3):
    """Wraps an async-generator handler with retries driven by a hook object.

    On exception, reads ``getattr(self, hook_attr)`` and expects an instance of
    ``FSMStateRetryHook``. The hook is called with ``(context, error)``;
    returning ``True`` retries from scratch. Any other return value (or missing
    hook) re-raises. Retries are capped at ``max_retries``.
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(self, context: "FSMContext"):
            last_scope: tuple[str, ...] = ("FSMStateRetryHook",)
            for attempt in range(max_retries + 1):
                attempt_start_turn = context.trajectory.turn
                attempt_start_answer_turn_count = getattr(context, "answer_turn_count", None)
                attempt_start_answer_attempt_count = getattr(context, "answer_attempt_count", None)
                attempt_start_answer_attempt_turn_count = getattr(
                    context, "answer_attempt_turn_count", None
                )
                attempt_start_verify_turn_count = getattr(context, "verify_turn_count", None)
                attempt_start_verify_attempt_turn_count = getattr(
                    context, "verify_attempt_turn_count", None
                )
                attempt_start_verify_cnt = getattr(context, "verify_cnt", None)
                try:
                    async for ev in fn(self, context):
                        if isinstance(ev, Observation):
                            last_scope = ev.scope
                        yield ev
                    return
                except Exception as e:
                    hook = getattr(self, hook_attr, None)
                    if hook is None or attempt >= max_retries:
                        raise
                    if not isinstance(hook, FSMStateRetryHook):
                        raise TypeError(
                            f"'{hook_attr}' must be an instance of FSMStateRetryHook, "
                            f"got {type(hook).__name__}"
                        )
                    result = (
                        await hook(context, e)
                        if inspect.iscoroutinefunction(hook.__call__)
                        else await asyncio.to_thread(hook, context, e)
                    )
                    if result:
                        logger.warning(
                            "Retryable exception in FSM state handler '%s' at attempt %d/%d: %s",
                            hook_attr,
                            attempt + 1,
                            max_retries + 1,
                            e,
                        )
                        # Roll back turns consumed by this failed state attempt.
                        context.trajectory.turn = attempt_start_turn
                        if attempt_start_answer_turn_count is not None:
                            context.answer_turn_count = attempt_start_answer_turn_count
                        if attempt_start_answer_attempt_count is not None:
                            context.answer_attempt_count = attempt_start_answer_attempt_count
                        if attempt_start_answer_attempt_turn_count is not None:
                            context.answer_attempt_turn_count = (
                                attempt_start_answer_attempt_turn_count
                            )
                        if attempt_start_verify_turn_count is not None:
                            context.verify_turn_count = attempt_start_verify_turn_count
                        if attempt_start_verify_attempt_turn_count is not None:
                            context.verify_attempt_turn_count = (
                                attempt_start_verify_attempt_turn_count
                            )
                        if attempt_start_verify_cnt is not None:
                            context.verify_cnt = attempt_start_verify_cnt
                        # Local import avoids circular import with tools/base.py.
                        from agents.self_verify_agent import FSMAgentState
                        # Inform the outer loop that we encountered an error
                        yield Observation(
                            scope=last_scope,
                            state=FSMAgentState.ERROR,
                            context=context,
                        )
                        # Clear for retrying this state
                        context.clear_for_next_state()
                    else:
                        raise
            raise RuntimeError(f"Retried {max_retries} but failed for {hook_attr}")
        return wrapper
    return decorator


@contextmanager
def attribute_overide(obj: Any, attribute: str, value: Any):
    """Temporarily set ``obj.attribute`` to ``value``, restoring it on exit.

    Mutates ``obj`` in place, so only use it on an object you exclusively own for
    the duration of the block. To override an attribute on an object shared with
    other concurrent tasks (e.g. a shared ``RuntimeConfig``), pass a copy so the
    change does not leak across ``await`` boundaries.
    """
    original = getattr(obj, attribute)
    setattr(obj, attribute, value)
    try:
        yield obj
    finally:
        setattr(obj, attribute, original)
