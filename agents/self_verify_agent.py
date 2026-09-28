# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import re
import json
import os
from contextlib import contextmanager
from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from constants import AgentState, EndReason, Message, MessageRole, PromptPayload
from agents.base_agent_suit import handles
from agents.seal_memory_agent import SealMemoryAgent, SealMemoryContext
from agents.events import Observation, Transition
from agents.state_machine import HandlerRegistryMixin, StateMachineRunner
from verifiers.utils import answer_parser, submit_result_parser, submit_tool_enabled
from agents.utils import (
    FSMStateRetryHook,
    api_response_handler,
    with_retry_hook,
    attribute_overide,
)

class RubricGenerationError(RuntimeError):
    pass

class VerifyGenerationError(RuntimeError):
    pass


class FSMAgentState(Enum):
    INITIAL = auto()
    RUBRIC = auto()
    ANSWER = auto()
    VERIFY = auto()
    ERROR = auto()
    END = auto()

class VerifyDecision(Enum):
    PASS = auto()
    REVISE_RUBRIC = auto()
    REVISE_ANSWER = auto()


@dataclass
class Rubric:
    id: int
    description: str
    satisfied: bool = False

    def __str__(self):
        return f"{self.id}: {self.description}"


@dataclass
class FSMContext(SealMemoryContext):
    transition_cnt: int = 0
    verify_cnt: int = 0
    question: str = ""
    benchmark: str = "browsecomp"
    # rubrics
    rubrics: list[Rubric] = field(default_factory=list)
    rubric_critique: str = ""
    rubric_suggestion: str = ""
    skip_rubrics: bool = False
    
    # answers
    rejected_answers: list[dict] = field(default_factory=list)
    answer_suggestion: str = ""
    answer_critique: str = ""
    current_answer: str = ""
    current_reason: str = ""
    over_turn_retry: int = -1
    over_seal_retry: int = -1
    max_answer_turns: int | None = None
    answer_attempt_count: int = 0
    answer_attempt_turn_count: int = 0
    answer_turn_count: int = 0
    answer_attempt_forced_final_triggered: bool = False
    over_turn_retry_triggered: bool = False
    over_seal_retry_triggered: bool = False
    over_turn_retry_disabled_final_attempt: bool = False
    over_seal_retry_disabled_final_attempt: bool = False
    turn_limit_final_answer_triggered: bool = False
    context_limit_final_answer_triggered: bool = False
    previous_retry_attempt_count: int = 0
    previous_retry_memories: list[dict] = field(default_factory=list)
    # verify
    skip_verify: bool = False
    max_verify: int = 3
    max_verify_turns: int = 128
    # Total Verify turns across all attempts, retained for aggregate metrics.
    verify_turn_count: int = 0
    # Verify turns consumed by the current attempt. Reset on every VERIFY entry.
    verify_attempt_turn_count: int = 0
    verify_limit_reached: bool = False
    verify_turn_limit_reached: bool = False
    final_answer_after_verify_limit: bool = False
    disable_rubric_reivision: bool = False
    # If True, the verifier judges answer correctness directly from the question
    # (no rubrics injected), matching rubric-free verify training.
    no_rubric_verify: bool = False
    # The tool budget display reads these while an Answer/Verify state is active.
    active_turn_counter: str | None = None
    active_turn_limit: int | None = None

    def __post_init__(self):
        # Direct FSMContext users (for example rubric pre-generation) only set
        # the inherited max_turns field. Preserve that behavior unless the FSM
        # runner explicitly supplies a separate Answer budget.
        if self.max_answer_turns is None:
            self.max_answer_turns = self.max_turns

    def clear_for_next_state(self):
        self.trajectory.clear()


@contextmanager
def _state_turn_budget(
    context: FSMContext,
    *,
    counter_attr: str | None,
    limit: int,
):
    """Expose a state-local budget through the shared BaseAgent context.

    ``trajectory.turn`` remains the end-to-end total for logging. BaseAgent,
    however, compares it with an absolute ceiling, so translate the remaining
    state-local allowance into that absolute ceiling for the duration of the
    nested run.
    """
    previous = (
        context.max_turns,
        context.active_turn_counter,
        context.active_turn_limit,
    )
    used = int(getattr(context, counter_attr, 0)) if counter_attr else 0
    remaining = max(int(limit) - used, 0)
    context.max_turns = context.trajectory.turn + remaining
    context.active_turn_counter = counter_attr
    context.active_turn_limit = int(limit) if counter_attr else None
    try:
        yield remaining
    finally:
        (
            context.max_turns,
            context.active_turn_counter,
            context.active_turn_limit,
        ) = previous


def _mark_final_answer_after_verify_limit(context: FSMContext) -> None:
    count_limit_reached = context.verify_cnt >= context.max_verify
    if count_limit_reached:
        context.verify_limit_reached = True
    if count_limit_reached:
        context.final_answer_after_verify_limit = True


def _record_automatic_verify_failure(
    context: FSMContext,
    *,
    critique: str,
    suggestion: str,
) -> None:
    """Record a failed Verify cycle that did not produce a model decision."""
    context.answer_critique = critique
    context.answer_suggestion = suggestion
    if context.current_answer:
        context.rejected_answers.append(
            {"answer": context.current_answer, "reason": critique}
        )
    _mark_final_answer_after_verify_limit(context)


def _next_state_after_verify_execution_failure(context: FSMContext) -> FSMAgentState:
    """Retry Verify on the same answer unless its attempt count is exhausted."""
    if context.verify_cnt >= context.max_verify:
        context.verify_limit_reached = True
        return FSMAgentState.END
    return FSMAgentState.VERIFY


def _parse_rubrics(content: str) -> List[Rubric]:
    """Parse rubrics from model output."""
    # 1. Extract JSON string (prioritize markdown blocks, fallback to outermost braces)
    match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    json_str = (
        match.group(1) if match else content[content.find("{") : content.rfind("}") + 1]
    )

    # 2. Attempt JSON parsing
    try:
        data = json.loads(json_str)
        if isinstance(data, dict) and "rubrics" in data:
            return [
                Rubric(id=item.get("id", i), description=item.get("description", ""), satisfied=bool(item.get("satisfied", False)))
                for i, item in enumerate(data["rubrics"], start=1) if isinstance(item, dict)
            ]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass  # Fall through to line-by-line parsing

    # 3. Fallback: Parse line-by-line using regex and the walrus operator
    pattern = re.compile(r"^\s*(\d+)\.\s*(.+)$")
    return [
        Rubric(id=int(m.group(1)), description=m.group(2).strip())
        for line in content.splitlines() if (m := pattern.match(line))
    ]


def _parse_reason(content: str) -> Optional[str]:
    if not content:
        return None

    # Isolate content after the last </think> tag
    content = content.split("</think>")[-1]

    # Patterns ordered by extraction priority
    patterns = (
        r"Reasoning:\s*(.+?)(?:\n\s*\n|\n\s*Answer:)",
        r"<reasoning>\s*(.+?)\s*</reasoning>",
        r"Summary:\s*(.+?)(?:\n\s*\n|\n\s*Answer:)",
    )

    for pattern in patterns:
        if match := re.search(pattern, content, re.IGNORECASE | re.DOTALL):
            return match.group(1).strip()

    return None


def _parse_verify_result(
    content: str,
) -> Tuple[Optional[VerifyDecision], Optional[str], Optional[str], Dict[int, bool]]:
    """Parse verification result from model output."""
    if not content:
        return VerifyDecision.REVISE_ANSWER, None, None, {}

    # 1. Strip reasoning tags
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    # 2. Extract JSON payload
    json_str = None
    if match := re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL):
        json_str = match.group(1)
    elif (start := content.find("{")) != -1 and (end := content.rfind("}")) > start:
        json_str = content[start : end + 1]

    if json_str:
        try:
            data = json.loads(json_str)
            d_str = data.get("decision", "").upper()

            decision = (
                getattr(VerifyDecision, d_str)
                if d_str in VerifyDecision.__members__
                else VerifyDecision.REVISE_ANSWER
            )

            rubric_results = {
                check["id"]: bool(check.get("satisfied"))
                for check in data.get("rubric_checks", [])
                if check.get("id") is not None
            }

            # If we successfully parsed the critique, return early and skip the fallback
            if critique := data.get("critique"):
                return decision, critique, data.get("suggestion"), rubric_results

        except json.JSONDecodeError:
            pass

    content_upper = content.upper()
    if "REVISE_RUBRIC" in content_upper or "REVISE RUBRIC" in content_upper:
        return VerifyDecision.REVISE_RUBRIC, content[:500], None, {}
    if "REVISE_ANSWER" in content_upper or "REVISE ANSWER" in content_upper:
        return VerifyDecision.REVISE_ANSWER, content[:500], None, {}
    if "PASS" in content_upper and "REVISE" not in content_upper:
        return VerifyDecision.PASS, content[:500], None, {}

    return None, content[:500], None, {}



def _is_valid_answer(answer: Optional[str]) -> Tuple[bool, str]:
    """
    Check if the answer is valid (not empty, no obvious refusals).

    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    if not answer or not (clean_answer := answer.strip()):
        return False, "empty_answer"

    lower_ans = clean_answer.lower()

    # Check if answer is just "Answer:"
    if lower_ans == "answer:":
        return False, "empty_answer_prefix"

    # Check for obvious error/refusal patterns
    refusal_patterns = (
        "i cannot provide",
        "i'm unable to",
        "i don't have access",
        "i could not find",
        "unable to determine",
        "cannot be determined",
    )

    if matched_refusal := next(
        (p for p in refusal_patterns if lower_ans.startswith(p)), None
    ):
        return False, f"refusal_response:{matched_refusal[:20]}"

    return True, ""


def _latest_assistant_message(context: FSMContext):
    for msg in reversed(context.trajectory.messages):
        if msg.role == MessageRole.ASSISTANT:
            return msg
    return None


def _should_over_turn_retry(context: FSMContext) -> bool:
    if context.over_turn_retry < 0:
        return False
    if context.answer_attempt_turn_count <= context.over_turn_retry:
        return False

    last_assistant_msg = _latest_assistant_message(context)
    if last_assistant_msg is None:
        return False

    return answer_parser(
        last_assistant_msg,
        strict=True,
        prefer_markdown_table=context.benchmark == "widesearch_200",
    ) is None


TURN_LIMIT_FINAL_ANSWER_PROMPT = """You have reached the maximum turn limit for this question.

You cannot use any more tools, search, link summaries, or memory operations. Based only on the information already gathered in this conversation, provide your best final answer now.

Absolutely do not write or simulate any tool call. Your response must not contain:
- <tool_call>, </tool_call>, <function=...>, </function>, <parameter=...>, or </parameter>
- search_api, link_summary_tool, seal_memory_tool, read_memory_tool, or any tool name
- JSON, XML, or pseudo-code for a tool call

You must still follow the required answer format:
Reasoning: <1-3 sentences explaining the key evidence you are relying on>

Answer: <your best final answer>

If the evidence is incomplete, choose the most likely answer from the gathered information rather than continuing to investigate."""

CONTEXT_LIMIT_FINAL_ANSWER_PROMPT = """The conversation has reached the context limit for this question.

You cannot use any more tools, search, link summaries, or memory operations. Based only on the information already gathered in this conversation, provide your best final answer now.

Absolutely do not write or simulate any tool call. Your response must not contain:
- <tool_call>, </tool_call>, <function=...>, </function>, <parameter=...>, or </parameter>
- search_api, link_summary_tool, seal_memory_tool, read_memory_tool, or any tool name
- JSON, XML, or pseudo-code for a tool call

You must still follow the required answer format:
Reasoning: <1-3 sentences explaining the key evidence you are relying on>

Answer: <your best final answer>

If the evidence is incomplete, choose the most likely answer from the gathered information rather than continuing to investigate."""

TURN_LIMIT_FINAL_ANSWER_REPAIR_PROMPT = """Your previous response was invalid because it attempted to use or describe a tool call after the maximum turn limit.

Tools are unavailable. Do not search. Do not mention or write any tool syntax, XML tags, function calls, tool names, or JSON.

Reply again using only this exact format:
Reasoning: <1-3 sentences using only evidence already gathered>

Answer: <your best final answer>"""

TURN_LIMIT_FINAL_ANSWER_SYSTEM_SUFFIX = """

FINAL ANSWER MODE OVERRIDE:
The research phase for this question is over. External operations are permanently
unavailable for the rest of this attempt, regardless of any earlier instruction
that described them. Select one concrete candidate from the evidence already
gathered. Your visible response must contain exactly:
Reasoning: <1-3 concise sentences>
Answer: <one concrete answer>
Do not continue investigating and do not return an abstention."""

_TURN_LIMIT_FORBIDDEN_FINAL_PATTERNS = (
    r"</?tool_call\b[^>]*>",
    r"</?function\b[^>]*>",
    r"<function\s*=",
    r"</?parameter\b[^>]*>",
    r"<parameter\s*=",
    r"\bsearch_api\b",
    r"\blink_summary_tool\b",
    r"\bseal_memory_tool\b",
    r"\bread_memory_tool\b",
    r"\btool_calls?\b",
    r"\btool\s+calls?\b",
)

_TURN_LIMIT_FORBIDDEN_FINAL_RE = re.compile(
    "|".join(_TURN_LIMIT_FORBIDDEN_FINAL_PATTERNS),
    re.IGNORECASE,
)


def _has_forbidden_turn_limit_final_output(
    message: Message,
    *,
    had_structured_tool_calls: bool = False,
) -> bool:
    if had_structured_tool_calls:
        return True
    # Hidden reasoning may legitimately mention the earlier research process.
    # Only reject forbidden syntax that leaks into the visible response.
    text = message.content or ""
    return bool(_TURN_LIMIT_FORBIDDEN_FINAL_RE.search(text))


def _is_usable_turn_limit_final_output(
    message: Message,
    *,
    had_structured_tool_calls: bool = False,
    prefer_markdown_table: bool = False,
) -> bool:
    if _has_forbidden_turn_limit_final_output(
        message,
        had_structured_tool_calls=had_structured_tool_calls,
    ):
        return False
    parsed = answer_parser(
        message,
        strict=True,
        prefer_markdown_table=prefer_markdown_table,
    )
    valid, _ = _is_valid_answer(parsed)
    return valid


def _activate_turn_limit_final_answer_mode(context: FSMContext) -> None:
    """Override the original tool-enabled system prompt for finalization."""
    suffix = TURN_LIMIT_FINAL_ANSWER_SYSTEM_SUFFIX.strip()
    for idx, message in enumerate(context.trajectory.messages):
        if message.role != MessageRole.SYSTEM:
            continue
        if suffix not in message.content:
            message.content = f"{message.content.rstrip()}\n\n{suffix}"
            context.trajectory.serialized_messages[idx]["content"] = message.content
        return


def _append_turn_limit_tool_stubs(
    context: FSMContext,
    *,
    reason: str = "the maximum turn limit was reached",
) -> None:
    pending_tool_calls = list(context.trajectory.pending_tool_calls or [])
    if not pending_tool_calls and context.trajectory.messages:
        last_msg = context.trajectory.messages[-1]
        if last_msg.role == MessageRole.ASSISTANT and last_msg.tool_calls:
            pending_tool_calls = list(last_msg.tool_calls)

    for tool_call in pending_tool_calls:
        context.trajectory.add_messages(
            Message(
                role=MessageRole.TOOL,
                content=json.dumps(
                    {
                        "succeed": False,
                        "error": (
                            f"Tool call skipped because {reason}. "
                            "No more tool use is allowed; answer from existing evidence."
                        ),
                    },
                    ensure_ascii=False,
                ),
                tool_call_id=tool_call.id,
                tool_call_succeed=False,
            )
        )
    context.trajectory.pending_tool_calls.clear()


def _needs_forced_final_answer(context: FSMContext) -> bool:
    return context.trajectory.end_reason == EndReason.TURN_LIMIT_REACHED


class FSMAgentHandler(HandlerRegistryMixin):
    def __init__(
        self,
        inner_agent: SealMemoryAgent,
        rubric_retry_hook: Optional[FSMStateRetryHook] = None,
        verify_retry_hook: Optional[FSMStateRetryHook] = None,
        answer_retry_hook: Optional[FSMStateRetryHook] = None,
    ):
        self.inner_agent = inner_agent
        self.rubric_retry_hook = rubric_retry_hook
        self.verify_retry_hook = verify_retry_hook
        self.answer_retry_hook = answer_retry_hook

    async def _run_forced_final_answer(self, context: FSMContext):
        if not _needs_forced_final_answer(context):
            return

        end_reason = context.trajectory.end_reason
        if end_reason != EndReason.TURN_LIMIT_REACHED:
            return

        context.turn_limit_final_answer_triggered = True
        context.answer_attempt_forced_final_triggered = True
        prompt = TURN_LIMIT_FINAL_ANSWER_PROMPT
        stub_reason = "the maximum turn limit was reached"

        _append_turn_limit_tool_stubs(context, reason=stub_reason)
        _activate_turn_limit_final_answer_mode(context)

        max_attempts = max(
            2,
            int(os.getenv("TURN_LIMIT_FINAL_ANSWER_MAX_ATTEMPTS", "4")),
        )
        for attempt in range(max_attempts):
            attempt_prompt = (
                prompt
                if attempt == 0
                else TURN_LIMIT_FINAL_ANSWER_REPAIR_PROMPT
            )
            message, had_tool_calls = await self._call_turn_limit_final_answer(
                context,
                attempt_prompt,
            )
            if message is None:
                context.current_answer = ""
                context.current_reason = ""
                context.trajectory.add_messages(Message(role=MessageRole.ASSISTANT, content=""))
                return
            yield Observation(
                scope=("BaseAgent",), state=AgentState.API_CALL, context=context
            )

            if _is_usable_turn_limit_final_output(
                message,
                had_structured_tool_calls=had_tool_calls,
                prefer_markdown_table=context.benchmark == "widesearch_200",
            ):
                context.trajectory.end_reason = EndReason.AGENT_EXIT
                return

        context.current_answer = ""
        context.current_reason = ""
        context.trajectory.add_messages(Message(role=MessageRole.ASSISTANT, content=""))
        context.trajectory.end_reason = EndReason.ERROR

    async def _call_turn_limit_final_answer(
        self,
        context: FSMContext,
        prompt: str,
    ) -> tuple[Message | None, bool]:
        context.trajectory.add_messages(Message(role=MessageRole.USER, content=prompt))

        base_handler = self.inner_agent.base_agent.handler
        payload = base_handler.prepare_llm_payload(context)
        # This is an extra answer turn after the search budget is exhausted.
        # Do not expose tools here; the model must answer from gathered evidence.
        payload["tools"] = []
        payload["tool_choice"] = "none"

        try:
            api_response = await context.runtime_config.main_llm.llm_client.call_llm(
                messages=context.trajectory.serialized_messages,
                **payload,
            )
        except Exception as exc:
            if "context" in str(exc).lower():
                context.trajectory.end_reason = EndReason.CONTEXT_EXCEED
                return None, False
            context.trajectory.end_reason = EndReason.ERROR
            raise

        message = api_response_handler(api_response)
        had_tool_calls = bool(message.tool_calls)
        if message.tool_calls:
            message.tool_calls = []
        context.trajectory.add_messages(message)
        context.trajectory.turn += 1
        context.answer_turn_count += 1
        context.answer_attempt_turn_count += 1
        context.trajectory.record_tokens_api(api_response.usage.total_tokens or 0)
        context.last_api_finish_reason = api_response.finish_reason
        context.last_api_prompt_tokens = api_response.usage.prompt_tokens
        context.last_api_completion_tokens = api_response.usage.completion_tokens
        context.last_api_reasoning_tokens = api_response.usage.reasoning_tokens
        return message, had_tool_calls

    @handles(FSMAgentState.INITIAL)
    async def handle_initial(self, context: FSMContext):
        yield Transition(FSMAgentState.RUBRIC)

    @handles(FSMAgentState.RUBRIC)
    @with_retry_hook("rubric_retry_hook")
    async def handle_rubric(self, context: FSMContext):
        if context.skip_rubrics:
            yield Transition(FSMAgentState.ANSWER)
            return
            
        lang = context.runtime_config.prompt_language
        if context.transition_cnt < 2:
            context.prompt_set = "fsm_rubric_gen"
            context.prompt_payload = PromptPayload(
                system={"prompt_language": lang},
                user={
                    "question": context.question,
                    "prompt_language": lang,
                },
            )
        else:
            context.prompt_set = "fsm_rubric_refine"
            context.prompt_payload = PromptPayload(
                system={"prompt_language": lang},
                user={
                    "suggestion": context.rubric_suggestion,
                    "critique": context.rubric_critique,
                    "question": context.question,
                    "current_rubrics": context.rubrics,
                    "prompt_language": lang,
                }
            )

        # Rubric generation is not charged to either the Answer or Verify pool.
        # Give this bounded state its own local ceiling so prior Verify turns do
        # not accidentally exhaust it through the global trajectory counter.
        with _state_turn_budget(
            context,
            counter_attr=None,
            limit=context.max_answer_turns,
        ):
            with attribute_overide(context, "tools", []):
                context.clear_for_next_state()
                async for ev in self.inner_agent.base_agent.stream_run(context):
                    yield ev

        last_assistant_msg = context.trajectory.messages[-1]
        if not last_assistant_msg.role == MessageRole.ASSISTANT:
            raise RubricGenerationError(
                f"Rubric generation ended with {last_assistant_msg.role}!"
            )
        rubrics = _parse_rubrics(last_assistant_msg.content)
        if not rubrics:
            raise RubricGenerationError(
                f"Unable to parse rubrics, raw content\n{last_assistant_msg.content}"
            )

        context.rubrics = rubrics
        yield Transition(FSMAgentState.ANSWER)

    @handles(FSMAgentState.ANSWER)
    @with_retry_hook("answer_retry_hook")
    async def handle_answer(self, context: FSMContext):
        context.answer_attempt_count += 1
        context.answer_attempt_turn_count = 0
        context.answer_attempt_forced_final_triggered = False
        context.current_answer = ""
        context.current_reason = ""

        # 1. Prepare prompt
        context.prompt_set = "fsm_answer"

        cfg = context.runtime_config
        system_payload = {
            "tool_set": context.allowed_tool_names,
            "skip_rubrics": context.skip_rubrics,
            "add_turn_budget": cfg.add_turn_budget,
            "add_token_budget": cfg.add_token_budget,
            "add_seal_budget": cfg.add_seal_budget,
            "enable_budget_prompt": cfg.enable_budget_prompt,
            "prompt_language": cfg.prompt_language,
            "benchmark": context.benchmark,
        }
        if context.rejected_answers:
            system_payload["rejected_answers"] = context.rejected_answers

        context.prompt_payload = PromptPayload(
            system=system_payload,
            user={
                k: v
                for k, v in {
                    "question": context.question,
                    "previous_answer": context.rejected_answers[-1]["answer"]
                    if context.rejected_answers
                    else None,
                    "rubrics": context.rubrics,
                    "failed_rubrics": [
                        str(r) for r in context.rubrics if not r.satisfied
                    ],
                    "critique": context.answer_critique,
                    "suggestion": context.answer_suggestion,
                    "previous_retry_attempt_count": context.previous_retry_attempt_count,
                    "previous_retry_memories": context.previous_retry_memories,
                    "prompt_language": cfg.prompt_language,
                }.items()
                if v
            },  # Automatically drops any keys where the value is None, [], or ""
        )
        context.clear_for_next_state()

        # 2. Generate answer
        with _state_turn_budget(
            context,
            counter_attr="answer_attempt_turn_count",
            limit=context.max_answer_turns,
        ) as remaining_answer_turns:
            if remaining_answer_turns <= 0:
                context.trajectory.end_reason = EndReason.TURN_LIMIT_REACHED
            else:
                last_seen_turn = context.trajectory.turn
                async for ev in self.inner_agent.stream_run(context):
                    yield ev
                    if context.trajectory.end_reason == EndReason.OVER_SEAL_RETRY:
                        yield Transition(FSMAgentState.END)
                        return
                    current_turn = context.trajectory.turn
                    if current_turn == last_seen_turn:
                        continue
                    if current_turn > last_seen_turn:
                        consumed_turns = current_turn - last_seen_turn
                        context.answer_turn_count += consumed_turns
                        context.answer_attempt_turn_count += consumed_turns
                    last_seen_turn = current_turn
                    if _should_over_turn_retry(context):
                        context.over_turn_retry_triggered = True
                        context.trajectory.end_reason = EndReason.OVER_TURN_RETRY
                        yield Transition(FSMAgentState.END)
                        return

        async for ev in self._run_forced_final_answer(context):
            yield ev
        if context.trajectory.end_reason == EndReason.CONTEXT_EXCEED:
            # No parseable answer was produced. With Verify enabled, count this
            # as an automatic failed Verify cycle and give the next Answer
            # attempt a fresh budget instead of ending the whole rollout.
            context.current_answer = ""
            context.current_reason = ""
            yield Transition(
                FSMAgentState.END
                if (
                    context.skip_verify
                    or context.final_answer_after_verify_limit
                    or context.max_verify <= 0
                )
                else FSMAgentState.VERIFY
            )
            return
        if (
            context.context_limit_final_answer_triggered
            and context.trajectory.end_reason != EndReason.AGENT_EXIT
        ):
            yield Transition(FSMAgentState.END)
            return

        # 3. Extract answer
        messages = context.trajectory.messages
        if submit_tool_enabled(context):
            submit_result = submit_result_parser(messages) or {}
            context.current_answer = str(submit_result.get("answer", "")).strip()
            context.current_reason = str(submit_result.get("reason", "")).strip() or None
        else:
            last_msg = messages[-1]
            assert last_msg.role == MessageRole.ASSISTANT, (
                f"Got last message role {last_msg.role} in answer!"
            )
            context.current_answer = answer_parser(
                last_msg,
                strict=True,
                prefer_markdown_table=context.benchmark == "widesearch_200",
            )
            context.current_reason = _parse_reason(last_msg.content)

        # 4. Inline transition
        yield Transition(
            FSMAgentState.END
            if (
                context.skip_verify
                or context.context_limit_final_answer_triggered
                or context.final_answer_after_verify_limit
                or context.max_verify <= 0
            )
            else FSMAgentState.VERIFY
        )

    @handles(FSMAgentState.VERIFY)
    @with_retry_hook("verify_retry_hook")
    async def handle_verify(self, context: FSMContext):
        if context.verify_cnt >= context.max_verify:
            context.verify_limit_reached = True
            yield Transition(FSMAgentState.END)
            return
        context.verify_cnt += 1
        context.verify_attempt_turn_count = 0

        # 1. Rule-based fast path
        validity, reason = _is_valid_answer(context.current_answer)
        
        if not validity:
            _record_automatic_verify_failure(
                context,
                critique=(
                    f"Answer validation failed: {reason}. The previous Answer attempt did "
                    "not produce a valid answer (possibly because it exceeded its context "
                    "window, emitted a malformed tool call, or returned an error message)."
                ),
                suggestion=(
                    "Start a fresh Answer attempt with a different, more focused search path "
                    "and provide a proper parseable final answer."
                ),
            )
            yield Transition(FSMAgentState.ANSWER)
            return

        lang = context.runtime_config.prompt_language
        context.prompt_set = "fsm_verify"
        context.prompt_payload = PromptPayload(
            system={
                "tool_set": context.allowed_tool_names,
                "disable_rubric_reivision": context.disable_rubric_reivision,
                "no_rubric_verify": context.no_rubric_verify,
                "prompt_language": lang,
            },
            user={
                "question": context.question,
                "rubrics": context.rubrics,
                "answer": context.current_answer,
                "reasoning": context.current_reason,
                "disable_rubric_reivision": context.disable_rubric_reivision,
                "no_rubric_verify": context.no_rubric_verify,
                "prompt_language": lang,
            },
        )
        context.clear_for_next_state()

        with _state_turn_budget(
            context,
            counter_attr="verify_attempt_turn_count",
            limit=context.max_verify_turns,
        ) as remaining_verify_turns:
            if remaining_verify_turns <= 0:
                context.verify_turn_limit_reached = True
                context.trajectory.end_reason = EndReason.TURN_LIMIT_REACHED
                yield Transition(_next_state_after_verify_execution_failure(context))
                return

            last_seen_turn = context.trajectory.turn
            async for ev in self.inner_agent.stream_run(context):
                yield ev
                current_turn = context.trajectory.turn
                if current_turn > last_seen_turn:
                    consumed_turns = current_turn - last_seen_turn
                    context.verify_turn_count += consumed_turns
                    context.verify_attempt_turn_count += consumed_turns
                last_seen_turn = current_turn

        if context.trajectory.end_reason == EndReason.TURN_LIMIT_REACHED:
            context.verify_turn_limit_reached = True
            yield Transition(_next_state_after_verify_execution_failure(context))
            return
        if context.trajectory.end_reason == EndReason.CONTEXT_EXCEED:
            yield Transition(_next_state_after_verify_execution_failure(context))
            return
        if context.trajectory.end_reason == EndReason.OVER_SEAL_RETRY:
            yield Transition(FSMAgentState.END)
            return

        last_msg = context.trajectory.messages[-1]
        if last_msg.role != MessageRole.ASSISTANT:
            if context.trajectory.end_reason == EndReason.CONTEXT_EXCEED:
                yield Transition(FSMAgentState.END)
                return
            raise VerifyGenerationError(
                f"Got last message role {last_msg.role} in Verification!"
            )

        # 3. Extract and route verification result
        decision, critique, suggestion, rubric_results = _parse_verify_result(
            last_msg.content
        )

        match decision:
            case VerifyDecision.PASS:
                yield Transition(FSMAgentState.END)

            case VerifyDecision.REVISE_ANSWER:
                context.answer_critique, context.answer_suggestion = (
                    critique,
                    suggestion,
                )
                context.rejected_answers.append(
                    {"answer": context.current_answer, "reason": critique}
                )

                for rubric in context.rubrics:
                    rubric.satisfied = (rubric_results or {}).get(rubric.id, False)

                _mark_final_answer_after_verify_limit(context)
                yield Transition(FSMAgentState.ANSWER)

            case VerifyDecision.REVISE_RUBRIC:
                context.rubric_critique, context.rubric_suggestion = (
                    critique,
                    suggestion,
                )
                _mark_final_answer_after_verify_limit(context)
                yield Transition(FSMAgentState.RUBRIC)

            case None:
                raise VerifyGenerationError("No decision found.")

            case _:
                raise VerifyGenerationError(
                    f"Verifier generated {last_msg.content} and cannot be parsed"
                )


class FSMAgent:
    name = "FSMAgent"

    def __init__(self, fsm_handler) -> None:
        self.fsm_handler = fsm_handler

    async def stream_run(
        self,
        context: FSMContext,
        state: FSMAgentState = FSMAgentState.INITIAL,
    ):
        # No `should_stop` guard here: `_state_turn_budget` rebinds
        # `context.max_turns` to a state-local ceiling, so a turn-based stop at
        # this level would cut the machine off before VERIFY or the forced
        # final answer ever runs. Per-state budgets own that decision.
        runner = StateMachineRunner(
            name=self.name,
            handler=self.fsm_handler,
            end_state=FSMAgentState.END,
        )
        async for ev in runner.stream(context, state):
            yield ev
