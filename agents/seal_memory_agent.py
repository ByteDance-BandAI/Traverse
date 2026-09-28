# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum, auto
from pathlib import Path
from typing import Any

from constants import EndReason, Message, MessageRole
from agents.base_agent_suit import BaseAgent, RunContext
from agents.events import Transition
from agents.state_machine import HandlerRegistryMixin, StateMachineRunner
from tools import tool_registry


class SealMemoryAgentState(Enum):
    INITIAL = auto()
    SEAL = auto()
    RUN_QUESTION = auto()
    END = auto()


class SealMemoryContext(RunContext):
    seal_count: int
    rollout_dir: "Path | None"  # set per-rollout; used by memory tools for path resolution


def _should_over_seal_retry(context: "SealMemoryContext") -> bool:
    threshold = getattr(context, "over_seal_retry", -1)
    if threshold < 0:
        return False
    return context.seal_count >= threshold


def _build_seal_summary_text(
    tool_result: dict[str, Any],
    prompt_language: str = "en",
) -> str:
    """
    Build a concise system note after sealing memory so new context can reload key info.
    """
    nav = tool_result.get("navigation_state", {})

    progress = tool_result.get("task_progress")
    plan = tool_result.get("next_step_plan")
    meta = tool_result.get("meta_learnings") or []
    visited = nav.get("visited_summary")
    frontier = nav.get("frontier_queue", [])
    dead_ends = nav.get("dead_ends", [])

    facts: list[str] = []
    for item in tool_result.get("knowledge_graph", []):
        if isinstance(item, dict):
            if fact := item.get("fact"):
                status = f" [{item['status']}]" if item.get("status") else ""
                facts.append(f"{fact}{status}")
        else:
            facts.append(str(item))

    def fmt_frontier(item: Any) -> str:
        if isinstance(item, dict):
            target, reason = item.get("target", ""), item.get("reason", "")
        else:
            target, reason = str(item), ""
        return f"- {target or '(unknown target)'}" + (f" ({reason})" if reason else "")

    parts: list[str] = []

    if memory_id := tool_result.get("memory_id"):
        parts.append(f"Sealed memory id: {memory_id}")
    if progress:
        parts.append(f"Task progress: {progress}")
    if plan:
        parts.append(f"Next step plan: {plan}")

    # Carry the full state forward (no truncation): dropping dead_ends/facts here
    # caused already-excluded leads to "resurrect" across seals and triggered
    # redundant re-searching. Keep everything so the next segment sees it.
    if facts:
        parts.append("Key facts:\n" + "\n".join(f"- {f}" for f in facts))
    if visited:
        parts.append(f"Visited path: {visited}")
    if frontier:
        parts.append("Frontier leads:\n" + "\n".join(fmt_frontier(i) for i in frontier))
    if dead_ends:
        parts.append("Dead ends to avoid (DO NOT re-investigate these):\n" + "\n".join(f"- {d}" for d in dead_ends))
    if meta:
        parts.append("Meta learnings:\n" + "\n".join(f"- {m}" for m in meta))

    if prompt_language == "zh":
        parts.extend([
            "发起新搜索前请先复用这些信息。",
            "请继续使用中文作答。"
        ])
    else:
        parts.extend([
            "Reuse these before launching new searches.",
            "Continue in English unless the user requested another language."
        ])

    return "\n".join(parts)


def _rebuild_context(
    tool_response: dict[str, Any],
    prompt_language: str = "en",
):
    """Build the full, untruncated carryover for the next segment."""
    if isinstance(tool_response, str):
        tool_response = json.loads(tool_response)

    return _build_seal_summary_text(
        tool_response,
        prompt_language=prompt_language,
    )



def handles(state: SealMemoryAgentState):
    """Decorator to mark a method as the handler for a specific AgentState."""
    def decorator(func):
        setattr(func, "_handles_state", state)
        return func
    return decorator


class SealAgentHandler(HandlerRegistryMixin):
    def __init__(self, base_agent: BaseAgent):
        self.base_agent = base_agent

    def is_memory_tool_call_end(self, message: Message) -> bool:
        return (message.role == MessageRole.ASSISTANT and
                bool(message.tool_calls) and
                message.tool_calls[0].name == "seal_memory_tool")

    @handles(SealMemoryAgentState.INITIAL)
    async def handle_initial(self, context: SealMemoryContext):
        yield Transition(SealMemoryAgentState.RUN_QUESTION)

    @handles(SealMemoryAgentState.RUN_QUESTION)
    async def handle_run_question(self, context: SealMemoryContext):
        async for ev in self.base_agent.stream_run(context):
            yield ev

        # messages[-1] is the tool response; messages[-2] is the assistant turn
        # that triggered the seal_memory_tool call we care about.
        if (
            len(context.trajectory.messages) >= 2
            and self.is_memory_tool_call_end(context.trajectory.messages[-2])
        ):
            yield Transition(SealMemoryAgentState.SEAL)
        else:
            yield Transition(SealMemoryAgentState.END)

    @handles(SealMemoryAgentState.SEAL)
    async def handle_seal_memory(self, context: SealMemoryContext):
        tool_response = context.trajectory.messages[-1].content
        context.seal_count += 1
        if _should_over_seal_retry(context):
            if hasattr(context, "over_seal_retry_triggered"):
                context.over_seal_retry_triggered = True
            context.trajectory.end_reason = EndReason.OVER_SEAL_RETRY
            yield Transition(SealMemoryAgentState.END)
            return

        lang = context.runtime_config.prompt_language
        carryover_text = _rebuild_context(tool_response, prompt_language=lang)

        if getattr(context.runtime_config, "add_seal_budget", False):
            seal_tool = tool_registry.get_instance("seal_memory_tool")
            if not seal_tool:
                raise RuntimeError("Cannot find seal_memory_tool when running seal_memory_agent!")
            if seal_tool.max_seal_times > 0:
                carryover_text += (
                    f"\n<seal_budget> Used: {context.seal_count} / "
                    f"Total: {seal_tool.max_seal_times}; "
                    f"Remain: {seal_tool.max_seal_times - context.seal_count} </seal_budget>"
                )

        context.trajectory.clear()
        context.prompt_payload.user.update({"carryover_text": carryover_text})

        yield Transition(SealMemoryAgentState.RUN_QUESTION)


class SealMemoryAgent:
    name = "SealMemoryAgent"

    def __init__(self, seal_handler: SealAgentHandler) -> None:
        self.seal_handler = seal_handler

    @property
    def base_agent(self) -> BaseAgent:
        return self.seal_handler.base_agent

    async def stream_run(
        self,
        context: SealMemoryContext,
        state: SealMemoryAgentState = SealMemoryAgentState.INITIAL,
    ):
        runner = StateMachineRunner(
            name=self.name,
            handler=self.seal_handler,
            end_state=SealMemoryAgentState.END,
        )
        async for ev in runner.stream(context, state):
            yield ev
