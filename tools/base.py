# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
import json
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional

from constants import ToolCall, ToolCallResult, RunContext
from loggers.metric_logger import metric_logger
from loggers.perf_timer import perf_timer
from tools.validation import (
    build_validator,
    expected_shape_hint,
    format_validation_errors,
)
from agents.utils import _tool_result_base_content


class ToolSchema:
    """A fluent builder for generating LLM function calling JSON schemas."""
    
    def __init__(self, name: str = "", description: str = ""):
        self.name = name
        self.description = description
        self.properties: Dict[str, Any] = {}
        self.required_fields: List[str] = []

    def add_string(self, name: str, description: str, required: bool = True, enum: Optional[List[str]] = None, default: Optional[str] = None):
        """Adds a string parameter to the schema."""
        prop = {"type": "string", "description": description}
        if enum:
            prop["enum"] = enum
        if default is not None:
            prop["default"] = default
        return self._add_property(name, prop, required)

    def add_number(self, name: str, description: str, required: bool = True, is_integer: bool = False, default: Optional[float] = None):
        """Adds a numeric parameter (float or integer)."""
        prop = {
            "type": "integer" if is_integer else "number", 
            "description": description
        }
        if default is not None:
            prop["default"] = default
        return self._add_property(name, prop, required)

    def add_boolean(self, name: str, description: str, required: bool = True, default: Optional[bool] = None):
        """Adds a boolean parameter."""
        prop = {"type": "boolean", "description": description}
        if default is not None:
            prop["default"] = default
        return self._add_property(name, prop, required)

    def add_array(self, name: str, description: str, item_type: str = "string", required: bool = True, default: Optional[list] = None):
        """Adds a basic array parameter with a single primitive item type."""
        prop = {
            "type": "array",
            "description": description,
            "items": {"type": item_type}
        }
        if default is not None:
            prop["default"] = default
        return self._add_property(name, prop, required)

    def add_one_of(self, name: str, description: str, options: List[dict], required: bool = True):
        """Adds a polymorphic parameter (e.g., can be a string OR an array of strings)."""
        prop = {
            "oneOf": options,
            "description": description
        }
        return self._add_property(name, prop, required)

    def add_object(self, name: str, description: str, schema: 'ToolSchema', required: bool = True):
        """Injects a nested object defined by another ToolSchema instance."""
        prop = {
            "type": "object",
            "description": description,
            "properties": schema.properties,
        }
        if schema.required_fields:
            prop["required"] = schema.required_fields
        return self._add_property(name, prop, required)

    def add_array_of_objects(self, name: str, description: str, schema: 'ToolSchema', required: bool = True):
        """Injects an array where each item is an object defined by another ToolSchema instance."""
        prop = {
            "type": "array",
            "description": description,
            "items": {
                "type": "object",
                "properties": schema.properties,
            }
        }
        if schema.required_fields:
            prop["items"]["required"] = schema.required_fields
        return self._add_property(name, prop, required)

    def _add_property(self, name: str, prop_dict: dict, required: bool):
        """Internal helper to attach properties and track required fields."""
        self.properties[name] = prop_dict
        if required and name not in self.required_fields:
            self.required_fields.append(name)
        return self 

    def build(self, wrap_in_function: bool = False) -> dict:
        """Compiles the builder into the final JSON schema expected by LLMs."""
        schema = {}
        
        # Only add name and description if they were provided 
        # (useful for inner schemas that don't need top-level names)
        if self.name:
            schema["name"] = self.name
        if self.description:
            schema["description"] = self.description
            
        schema["parameters"] = {
            "type": "object",
            "properties": self.properties,
            "additionalProperties": False # Enforces strict schema matching for LLMs
        }
        
        # Only attach the required array if there are required fields
        if self.required_fields:
            schema["parameters"]["required"] = self.required_fields
            
        # Wrap in the standard OpenAI-compatible tool envelope if requested
        if wrap_in_function:
            return {
                "type": "function",
                "function": schema
            }
            
        return schema

async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class ReturnableToolError(Exception):
    """Exception raised by tools when the error message is safe and useful to
    surface back to the Agent as the tool's output.

    `BaseTool.execute` catches only this exception and converts it into a
    structured `ToolCallResult`. Any other exception is considered a bug or
    an unrecoverable failure and is re-raised so it can be handled (or crash)
    at a higher level.
    """


class ToolBackendUnavailable(Exception):
    """Raised from a tool constructor when its backend cannot serve requests.

    Signals a deployment gap (missing SDK, unset credentials) rather than a
    bug. `ToolRegistry.init_tools` records these instead of failing, so a
    deployment that only uses some of the registered tools still starts, and
    the error surfaces with its original reason if such a tool is requested.
    """


class BaseTool(ABC):
    """
    Candidate base template for tool execution.

    Execution flow:
    pre_tool_call -> run -> post_tool_call
                      | error
                  on_tool_error

    Argument validation:
        `pre_tool_call` strict-validates `payload.arguments` against `self.schema`
        using a cached `jsonschema` validator. Type/required/enum/extra-field
        violations are raised as `ReturnableToolError` so the agent receives an
        actionable message and can retry. Set `validate_arguments = False` on a
        subclass to opt out (e.g. tools that intentionally accept free-form payloads).
    """

    name: str = "base_tool"
    validate_arguments: bool = True

    def __init__(self, name: str, custom_tool_prompt: str = None) -> None:
        # Adapters use explicit names so schemas, metrics, and dispatch stay aligned.
        self.name = name
        self.custom_tool_prompt = custom_tool_prompt
    
    def get_custom_tool_prompt(self):
        return self.custom_tool_prompt

    @cached_property
    def _args_validator(self):
        """Compiled once per tool instance; None if the tool has no schema."""
        return build_validator(getattr(self, "schema", None))

    async def execute(self, payload: ToolCall, context: RunContext) -> ToolCallResult:
        start = time.perf_counter()
        prepared_payload = payload

        with perf_timer.measure(f"tool.{self.name}"):
            try:
                prepared_payload = await _maybe_await(self.pre_tool_call(prepared_payload, context))
                raw_output = await _maybe_await(self.run(prepared_payload, context))

                result = ToolCallResult(ok=True, output=raw_output)
                result = await _maybe_await(self.post_tool_call(result, context))
                result.duration_ms = (time.perf_counter() - start) * 1000
                self._log_metrics(payload, result, context)
                return result
            except ReturnableToolError as exc:
                # Only errors explicitly marked as "returnable" are surfaced to the
                # Agent as structured tool output. All other exceptions propagate.
                result = ToolCallResult(ok=False, error=exc, duration_ms=(time.perf_counter() - start) * 1000)
                result = await _maybe_await(self.on_tool_error(result, context))
                self._log_metrics(payload, result, context)
                return result

    def _log_metrics(self, payload: ToolCall, result: ToolCallResult, context: RunContext) -> None:
        """Record per-call count, error rate and latency, broken down by tool name."""
        qid, rollout_idx = context.question_id, context.rollout_idx
        name = payload.name
        metric_logger.incr(qid, f"tool_calls.{name}", 1, rollout_idx)
        metric_logger.rate(qid, f"tool_error_rate.{name}", not result.ok, rollout_idx)
        metric_logger.observe(qid, f"tool_latency_ms.{name}", result.duration_ms, rollout_idx)

    async def pre_tool_call(self, payload: ToolCall, context: RunContext) -> ToolCall:
        """Hook: validate/enrich payload before run().

        Default implementation: strict-validate `payload.arguments` against
        `self.schema`. Subclasses that override this should call
        `super().pre_tool_call(payload, context)` first to keep validation.
        """
        self._validate_arguments(payload)
        return payload

    def _validate_arguments(self, payload: ToolCall) -> None:
        if not self.validate_arguments or self._args_validator is None:
            return

        args = payload.arguments
        if not isinstance(args, dict):
            # Pydantic should already reject this at ToolCall construction, but
            # guard anyway for callers that bypass the model (e.g. tests).
            raise ReturnableToolError(
                f"{self.name}: `arguments` must be a JSON object, "
                f"got {type(args).__name__}. "
                f"Expected shape: {expected_shape_hint(self.schema)}"
            )

        errors = list(self._args_validator.iter_errors(args))
        if errors:
            raise ReturnableToolError(
                format_validation_errors(errors, self.name, self.schema)
            )

    @abstractmethod
    async def run(self, payload: ToolCall, context: RunContext) -> str:
        """Tool implementation."""
        raise NotImplementedError

    async def post_tool_call(self, result: ToolCallResult, context: RunContext) -> ToolCallResult:
        """Hook: normalize output, attach metadata, or transform result."""
        runtime_config = context.runtime_config
        if not (runtime_config.add_token_budget or runtime_config.add_turn_budget):
            return result

        parts: List[str] = []

        if runtime_config.add_token_budget:
            estimator = runtime_config.token_estimator

            projected_messages = list(context.trajectory.serialized_messages)
            projected_messages.append(
                {"role": "tool", "content": _tool_result_base_content(result)}
            )
            projected_token_estimate = int(
                await asyncio.to_thread(
                    estimator,
                    messages=projected_messages,
                    context=context,
                    tools=context.tools
                )
            )

            token_used = projected_token_estimate
            token_total = int(runtime_config.main_llm.max_tokens) - 4000

            token_total = max(token_total, 0)
            token_remain = max(token_total - token_used, 0)
            parts.append(
                f"<token_budget> Used: {token_used} / Total: {token_total}; Remain: {token_remain} </token_budget>"
            )

        if runtime_config.add_turn_budget:
            # `_state_turn_budget` rebinds these when a bounded FSM state is
            # running, so a state-local ceiling must win over the global one.
            active_counter = getattr(context, "active_turn_counter", None)
            active_limit = getattr(context, "active_turn_limit", None)
            if isinstance(active_counter, str) and isinstance(active_limit, int):
                turn_used = int(getattr(context, active_counter))
                turn_total = int(active_limit)
            else:
                turn_used = int(context.trajectory.turn)
                turn_total = int(context.max_turns)
            turn_remain = max(turn_total - turn_used, 0)
            parts.append(
                f"<turn_budget> Used: {turn_used} / Total: {turn_total}; Remain: {turn_remain} </turn_budget>"
            )

        result.extra_info = "\n".join(parts)
        return result

    async def on_tool_error(self, result: ToolCallResult, context: RunContext) -> ToolCallResult:
        """Hook: map exceptions to structured error output."""
        return result


class TerminateTool(BaseTool):
    async def post_tool_call(self, result: ToolCallResult, context: RunContext) -> ToolCallResult:
        # No post tool call behavior for termination tools
        return result


class MemoryTool(TerminateTool):
    pass

class SubmitToolBase(TerminateTool):
    pass
