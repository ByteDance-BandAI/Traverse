# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from prompt.base import PromptFactory
from constants import (
    MessageRole, 
    EndReason, 
    AgentState, 
    Message, 
    ToolCall, 
    ToolCallResult,
    RunContext,
)
from tools import tool_registry
from agents.events import Transition
from agents.state_machine import ExceptionAction, HandlerRegistryMixin, StateMachineRunner
from llm_clients.base import LLMCallResult
from loggers.perf_timer import perf_timer
from typing import Any
from agents.utils import (
    handles,
    tool_result_to_content,
    api_response_handler,
)
from agents.tool_loop_guard import (
    MAX_CYCLE_LENGTH_CAP,
    detect_repeat_cycle,
    record_signature,
    repeated_call_notice,
)


class BaseAgentTurnRetryHook:
    def __call__(
        self,
        api_response: LLMCallResult,
        context: RunContext,
        attempt: int
    ) -> bool:
        return False


class BaseAgentHandler(HandlerRegistryMixin, ABC):
    def __init__(self):
        self.turn_retry_hooks: list[BaseAgentTurnRetryHook] | None = None

    def prepare_llm_payload(self, context: RunContext) -> dict[str, Any]:
        """Override to format the exact payload sent to the LLM."""
        llm_cfg = context.runtime_config.main_llm
        completion_cap = context.runtime_config.completion_cap
        if getattr(context, "prompt_set", None) in {
            "fsm_rubric_gen",
            "fsm_rubric_refine",
        }:
            rubric_completion_cap = getattr(
                context.runtime_config,
                "rubric_completion_cap",
                None,
            )
            if rubric_completion_cap is not None:
                completion_cap = rubric_completion_cap
        max_tokens = min(llm_cfg.max_tokens, completion_cap)

        return {
            k: v for k, v in {
                "max_tokens": max_tokens,
                "temperature": llm_cfg.temperature,
                "tools": context.tools,
                "top_p": llm_cfg.top_p,
                "top_k": llm_cfg.top_k,
                "min_p": llm_cfg.min_p,
                "presence_penalty": llm_cfg.presence_penalty,
                "repetition_penalty": llm_cfg.repetition_penalty,
                "chat_id": context.trajectory.chat_id, # Randomly generated for kv caching
                "include_thoughts": llm_cfg.include_thoughts,
            }.items() if v is not None
        }

    def register_turn_retry_hook(self, hooks: list[BaseAgentTurnRetryHook]):
        self.turn_retry_hooks = hooks

    def should_retry_turn(self, api_response: LLMCallResult, context: RunContext, attempt: int) -> bool:
        if not self.turn_retry_hooks:
            return False
        return any(hook(api_response, context, attempt) for hook in self.turn_retry_hooks)

    async def execute_tool_call(self, tool_calls: list[ToolCall], context: RunContext) -> list[ToolCallResult]:
        """Override to define how tools are actually executed."""
        return await tool_registry.execute(tool_calls=tool_calls, context=context)

    @handles(AgentState.INITIAL)
    async def handle_initial(self, context: RunContext):
        factory = PromptFactory(prompt_set=context.prompt_set)

        context.trajectory.add_messages(
            Message(
                role=MessageRole.SYSTEM, 
                content=factory.get_prompt(role="system", **context.prompt_payload.system)
            ),
            Message(
                role=MessageRole.USER, 
                content=factory.get_prompt(role="user", **context.prompt_payload.user)
            )
        )
        yield Transition(AgentState.API_CALL)

    def blocked_repeated_calls(
        self, tool_calls: list[ToolCall], context: RunContext
    ) -> dict[int, ToolCallResult]:
        """Locally answer looping tool calls, keyed by index in `tool_calls`.

        A blocked call never reaches the tool layer, so a model stuck in a cycle
        spends no MCP request on results it already has.
        """
        if not getattr(context.runtime_config, "tool_loop_guard", False):
            return {}

        max_cycle_length = getattr(
            context.runtime_config, "tool_loop_guard_max_cycle", MAX_CYCLE_LENGTH_CAP
        )
        prompt_language = getattr(context.runtime_config, "prompt_language", "en")
        signatures = context.trajectory.recent_tool_signatures

        blocked: dict[int, ToolCallResult] = {}
        for index, tool_call in enumerate(tool_calls):
            record_signature(signatures, tool_call)
            cycle_length = detect_repeat_cycle(signatures, max_cycle_length)
            if cycle_length is None:
                continue
            context.tool_loop_blocked_count += 1
            blocked[index] = ToolCallResult(
                ok=False,
                error=repeated_call_notice(cycle_length, prompt_language),
            )
        return blocked

    @handles(AgentState.TOOL)
    async def handle_tool(self, context: RunContext):
        pending = list(context.trajectory.pending_tool_calls)
        if not pending:
            raise ValueError("No tool calls while entering TOOL state!")

        pending_tool_calls = context.trajectory.pending_tool_calls
        has_memory_tool = any(
            tool_registry.is_memory_tool_call(tc) for tc in pending_tool_calls
        )
        # Memory batches bypass the guard: sealing the same state twice is
        # legitimate, a blocked seal would hand the seal handler a warning
        # instead of a memory snapshot, and dropping a sibling call would change
        # how the tool layer treats a mixed batch.
        blocked = (
            {}
            if has_memory_tool
            else self.blocked_repeated_calls(pending_tool_calls, context)
        )
        responses: list[ToolCallResult | None] = [
            blocked.get(index) for index in range(len(pending_tool_calls))
        ]
        runnable = [
            index for index, response in enumerate(responses) if response is None
        ]
        if runnable:
            executed = await self.execute_tool_call(
                [pending_tool_calls[index] for index in runnable], context
            )
            for index, response in zip(runnable, executed):
                responses[index] = response

        terminal_tool_seen = False
        for tc, response in zip(pending_tool_calls, responses):
            if tool_registry.is_memory_tool_call(tc):
                terminal_tool_seen = True

            context.trajectory.add_messages(
                Message(
                    role=MessageRole.TOOL, 
                    content=tool_result_to_content(response), 
                    tool_call_id=tc.id, 
                    tool_call_succeed=response.ok
                )
            )

        context.trajectory.pending_tool_calls.clear()

        if terminal_tool_seen:
            succeeded = len(responses) == 1 and responses[0].ok
            yield Transition(AgentState.END if succeeded else AgentState.API_CALL)
            return

        yield Transition(AgentState.API_CALL)

    @handles(AgentState.API_CALL)
    async def handle_api_call(self, context: RunContext):
        payload = self.prepare_llm_payload(context)
        max_retries = max(0, context.max_retry_limit)

        for attempt in range(max_retries + 1):
            with perf_timer.measure("api_call"):
                api_response = await context.runtime_config.main_llm.llm_client.call_llm(
                    messages=context.trajectory.serialized_messages,
                    **payload,
                )

            if attempt >= max_retries or not self.should_retry_turn(api_response, context, attempt):
                break

        msg = api_response_handler(api_response)
        context.trajectory.add_messages(msg)
        context.trajectory.turn += 1
        context.trajectory.record_tokens_api(api_response.usage.total_tokens)
        context.last_api_finish_reason = api_response.finish_reason
        context.last_api_prompt_tokens = api_response.usage.prompt_tokens
        context.last_api_completion_tokens = api_response.usage.completion_tokens
        context.last_api_reasoning_tokens = api_response.usage.reasoning_tokens
        # In case we can't set context length limit of the backend, raise error here.
        max_tokens = context.runtime_config.main_llm.max_tokens
        if context.trajectory.current_num_tokens_api > max_tokens:
            raise RuntimeError(
                f"Context length exceeded: total tokens used ({context.trajectory.current_num_tokens_api}) "
                f"surpasses the configured max_tokens limit ({max_tokens})."
            )

        context.trajectory.pending_tool_calls = list(msg.tool_calls or [])
        yield Transition(AgentState.TOOL if context.trajectory.pending_tool_calls else AgentState.END)


class BaseAgent:
    name = "BaseAgent"

    def __init__(self, handler: BaseAgentHandler):
        self.handler = self._validate_handler(handler)

    def _validate_handler(self, handler: BaseAgentHandler | None) -> BaseAgentHandler:
        if handler is None:
            raise ValueError("Subclasses must pass a handler when initializing BaseAgent")
        if not isinstance(handler, BaseAgentHandler):
            raise TypeError(f"handler must be a BaseAgentHandler, got {type(handler).__name__}")
        return handler

    @staticmethod
    def _should_stop(context: RunContext, state: AgentState) -> bool:
        if context.trajectory.turn >= context.max_turns:
            context.trajectory.end_reason = EndReason.TURN_LIMIT_REACHED
            return True
        return False

    @staticmethod
    def _on_exception(exc: Exception, context: RunContext, state: AgentState) -> ExceptionAction:
        if "context" in str(exc).lower():
            context.trajectory.end_reason = EndReason.CONTEXT_EXCEED
            return ExceptionAction.BREAK
        context.trajectory.end_reason = EndReason.ERROR
        return ExceptionAction.RAISE

    @staticmethod
    def _on_natural_end(context: RunContext, state: AgentState) -> None:
        context.trajectory.end_reason = EndReason.AGENT_EXIT

    async def stream_run(
        self,
        context: RunContext,
        state: AgentState = AgentState.INITIAL,
    ):
        """
        Executes the agent loop as an async generator of `Observation`s.
        """
        runner = StateMachineRunner(
            name=self.name,
            handler=self.handler,
            end_state=AgentState.END,
            should_stop=self._should_stop,
            on_exception=self._on_exception,
            on_natural_end=self._on_natural_end,
        )
        async for ev in runner.stream(context, state):
            yield ev
