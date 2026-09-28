# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from agents.base_agent_suit import BaseAgentTurnRetryHook
from agents.utils import FSMStateRetryHook
from rollout.rollout_manager import RetryTrajHook
from verifiers.utils import answer_parser
from llm_clients.base import LLMCallResult
from constants import RunContext, Message, MessageRole, EndReason
from loggers.metric_logger import metric_logger
from logging import getLogger
import re
from agents.self_verify_agent import (
    RubricGenerationError,
    VerifyGenerationError,
    FSMContext,
    _parse_rubrics,
    _parse_verify_result,
)

logger = getLogger(__name__)


def _record_retry(context: RunContext, reason: str) -> None:
    """Count one retry for the current (question_id, rollout_idx) unit.

    Retries are tracked per cause via ``num_retries.{reason}``. The metric
    report then exposes both the per-question averages and the global
    aggregates for each reason.
    """
    qid, rollout_idx = context.question_id, context.rollout_idx
    metric_logger.incr(qid, f"num_retries.{reason}", 1, rollout_idx)


_MALFORMED_TOOL_CALL_PATTERNS = [
    r"<parameter=[^>]*>",
    r"</?tool_call>",
    r"</?function\b[^>]*>",
    r"<parameter>[^<]*>",
]
_MALFORMED_TOOL_CALL_RE = re.compile(
    "|".join(_MALFORMED_TOOL_CALL_PATTERNS),
    re.IGNORECASE,
)


def _prefer_markdown_table_answer(context: RunContext) -> bool:
    return getattr(context, "benchmark", "") == "widesearch_200"


def has_malformed_tool_call(text: str) -> bool:
    return bool(_MALFORMED_TOOL_CALL_RE.search(text))

class LengthOrMalformedRetry(BaseAgentTurnRetryHook):
    @staticmethod
    def _hit_output_token_limit(
        api_response: LLMCallResult,
        context: RunContext,
    ) -> bool:
        """True when the response was cut off at the per-call output token limit.

        Checks finish_reason first, then falls back to comparing completion_tokens
        against the effective output cap — needed because sglang (and some other
        backends) sometimes returns finish_reason="" or None instead of "length".
        """
        hit = False
        if api_response.finish_reason == "length":
            hit = True
        completion_tokens = api_response.usage.completion_tokens
        if completion_tokens is not None:
            per_call_max = min(
                context.runtime_config.main_llm.max_tokens,
                context.runtime_config.completion_cap,
            )
            hit = (completion_tokens >= per_call_max)
        return hit

    def __call__(
        self,
        api_response: LLMCallResult,
        context: RunContext,
        attempt: int,
    ) -> bool:
        if (
            self._hit_output_token_limit(api_response, context)
            and context.trajectory.current_num_tokens_api < context.runtime_config.main_llm.max_tokens
        ):
            logger.warning(f"Attempt {attempt}: Retry on token limit!")
            # logger.warning(f"Token limit: {api_response.reasoning_content}")
            # logger.warning(f"Token limit: {api_response.content}")
            _record_retry(context, "token_limit")
            return True

        llm_response_full = (api_response.content or "") + (
            api_response.reasoning_content or ""
        )
        if not api_response.tool_calls and has_malformed_tool_call(llm_response_full):
            logger.warning(f"Attempt {attempt}: Retry on malformed tool call!")
            logger.warning(f"Malformed tool call: {llm_response_full}")
            _record_retry(context, "malformed_tool_call")
            return True

        return False


class NoToolAndAnswerRetry(LengthOrMalformedRetry):
    def __call__(
        self,
        api_response: LLMCallResult,
        context: RunContext,
        attempt: int,
    ) -> bool:
        if super().__call__(api_response, context, attempt):
            return True

        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=api_response.content or "",
            reasoning_content=api_response.reasoning_content or ""
        )
        if (
            answer_parser(
                assistant_msg,
                strict=True,
                prefer_markdown_table=_prefer_markdown_table_answer(context),
            )
            is None
            and not api_response.tool_calls
        ):
            logger.warning(f"Attempt {attempt}: Retry on no answer!")
            _record_retry(context, "no_answer_no_tool")
            return True
        return False


class FSMTurnParseRetry(BaseAgentTurnRetryHook):
    def __init__(
        self,
        *,
        retry_rubric_parse: bool = False,
        retry_answer_parse: bool = False,
        retry_verify_parse: bool = False,
    ) -> None:
        self.retry_rubric_parse = retry_rubric_parse
        self.retry_answer_parse = retry_answer_parse
        self.retry_verify_parse = retry_verify_parse

    @staticmethod
    def _current_fsm_state(context: RunContext) -> str | None:
        for frame in context.state_stack:
            if frame.agent == "FSMAgent":
                return frame.state
        return None

    def __call__(
        self,
        api_response: LLMCallResult,
        context: RunContext,
        attempt: int,
    ) -> bool:
        fsm_state = self._current_fsm_state(context)
        if fsm_state is None or api_response.tool_calls:
            return False

        content = api_response.content or ""
        if fsm_state == "RUBRIC":
            if not self.retry_rubric_parse:
                return False
            should_retry = not bool(_parse_rubrics(content))
        elif fsm_state == "ANSWER":
            if not self.retry_answer_parse:
                return False
            assistant_msg = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                reasoning_content=api_response.reasoning_content or ""
            )
            should_retry = answer_parser(
                assistant_msg,
                strict=True,
                prefer_markdown_table=_prefer_markdown_table_answer(context),
            ) is None
        elif fsm_state == "VERIFY":
            if not self.retry_verify_parse:
                return False
            decision, _, _, _ = _parse_verify_result(content)
            should_retry = decision is None
        else:
            should_retry = False

        if should_retry:
            logger.warning(
                "Attempt %s: FSM turn parse retry for state=%s",
                attempt,
                fsm_state,
            )
            _record_retry(context, f"fsm_parse.{fsm_state}")
        return should_retry


class FSMNoToolAndAnswerRetry(LengthOrMalformedRetry):
    def __call__(
        self,
        api_response: LLMCallResult,
        context: RunContext,
        attempt: int,
    ) -> bool:
        if super().__call__(api_response, context, attempt):
            return True

        fsm_state = FSMTurnParseRetry._current_fsm_state(context)
        if fsm_state != "ANSWER" or api_response.tool_calls:
            return False

        assistant_msg = Message(
            role=MessageRole.ASSISTANT,
            content=api_response.content or "",
            reasoning_content=api_response.reasoning_content or "",
        )
        if answer_parser(
            assistant_msg,
            strict=True,
            prefer_markdown_table=_prefer_markdown_table_answer(context),
        ) is None:
            logger.warning(
                "Attempt %s: Retry on no tool and no strict answer in FSM ANSWER state!",
                attempt,
            )
            return True
        return False


class RetryLengthyResponse(RetryTrajHook):
    def __call__(self, context: RunContext) -> bool:
        if context.trajectory.end_reason is not EndReason.AGENT_EXIT:
            return False
        if not context.trajectory.messages:
            return False

        last_msg = context.trajectory.messages[-1]
        if last_msg.role is not MessageRole.ASSISTANT:
            return False
        if answer_parser(
            last_msg,
            strict=True,
            prefer_markdown_table=_prefer_markdown_table_answer(context),
        ) is not None:
            return False
        if last_msg.tool_calls:
            return False

        within_max_tokens = (
            context.trajectory.current_num_tokens_api
            < context.runtime_config.main_llm.max_tokens
        )
        hit_length_finish_reason = context.last_api_finish_reason == "length"
        per_call_max = min(
            context.runtime_config.main_llm.max_tokens,
            context.runtime_config.completion_cap,
        )
        hit_completion_cap = (
            context.last_api_completion_tokens is not None
            and context.last_api_completion_tokens >= per_call_max
        )
        if within_max_tokens and (hit_length_finish_reason or hit_completion_cap):
            return True

        llm_response_full = (last_msg.content or "") + (last_msg.reasoning_content or "")
        if has_malformed_tool_call(llm_response_full):
            return True

        return False


class RetryUnparseableFinalAnswer(RetryTrajHook):
    def __call__(self, context: RunContext) -> bool:
        if context.trajectory.end_reason is not EndReason.AGENT_EXIT:
            return False
        if not context.trajectory.messages:
            return False

        last_msg = context.trajectory.messages[-1]
        if last_msg.role is not MessageRole.ASSISTANT:
            return False
        if last_msg.tool_calls:
            return False

        return answer_parser(
            last_msg,
            strict=True,
            prefer_markdown_table=_prefer_markdown_table_answer(context),
        ) is None


class OverTurnRetry(RetryTrajHook):
    def __call__(self, context: RunContext) -> bool:
        return bool(getattr(context, "over_turn_retry_triggered", False))


class OverSealRetry(RetryTrajHook):
    def __call__(self, context: RunContext) -> bool:
        return bool(getattr(context, "over_seal_retry_triggered", False))


class RubricParseRetry(FSMStateRetryHook):
    def __call__(self, context: FSMContext, error: Exception):
        if isinstance(error, RubricGenerationError):
            _record_retry(context, "rubric_generation_error")
            return True

class VerifyParseRetry(FSMStateRetryHook):
    def __call__(self, context: FSMContext, error: Exception):
        if isinstance(error, VerifyGenerationError):
            _record_retry(context, "verify_generation_error")
            return True
