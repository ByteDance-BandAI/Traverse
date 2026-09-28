# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import traceback
from loggers.logging_config import setup_logging

setup_logging()

import recipe.seal_memory_agent.seal_prompt  # DON'T delete!

from pathlib import Path
from typing import Tuple
from tools import tool_registry
from agents.base_agent_suit import BaseAgentHandler, BaseAgent
from agents.seal_memory_agent import (
    SealAgentHandler,
    SealMemoryAgent,
    SealMemoryContext,
    SealMemoryAgentState,
)
from agents.events import Observation
from constants import AgentState, AgentTrajectory, LLMBackend, PromptPayload, question_dir
from config import RuntimeConfig
from dataset import get_bc_dataset
from recipe.seal_memory_agent.arguments import build_args
from dataset.bc_dataset import BCReturn
from verifiers import JudgeCompleteStatus, get_verifier_cls, JudgeResult
from verifiers.base_qa_llm_judge import BaseQALLMJudge
from verifiers.utils import parse_answer
from loggers.file_logger import JsonlFileLogger
from loggers.segment_logger import SegmentFileLogger
from loggers.metric_logger import metric_logger
from rollout.context_factory import BaseContextFactory
from rollout.rollout_manager import GenerationController
from rollout.review_paths import review_jsonl_path
from rollout.retries import LengthOrMalformedRetry, NoToolAndAnswerRetry
from rollout.utils import _prepare_tools, _get_llm_config
from utils import tokenizer_estimator


class SealContextFactory(BaseContextFactory):
    def __init__(
        self,
        runtime: RuntimeConfig,
        tools: list,
        allowed_tools: list,
        max_turns: int,
        backend_parser: LLMBackend | None,
    ):
        self.runtime = runtime
        self.tools = tools
        self.allowed_tools = allowed_tools
        self.max_turns = max_turns
        self.backend_parser = backend_parser

    async def create(self, question: BCReturn, rollout_idx: int) -> Tuple[SealMemoryContext, SealMemoryAgentState]:
        context = SealMemoryContext(
            runtime_config=self.runtime,
            trajectory=AgentTrajectory(
                messages=[],
                backend_parser=self.backend_parser,
            ),
            tools=self.tools,
            prompt_payload=PromptPayload(
                system={
                    "tool_set": self.allowed_tools,
                    "prompt_language": self.runtime.prompt_language,
                },
                user={
                    "question": question.question,
                    "prompt_language": self.runtime.prompt_language,
                },
            ),
        )
        context.seal_count = 0
        context.question_id = str(question.question_id)
        context.rollout_dir = Path(self.runtime.work_dir) / f"rollout_{rollout_idx}"
        context.rollout_idx = rollout_idx
        context.prompt_set = "seal"
        context.max_turns = self.max_turns
        return context, SealMemoryAgentState.INITIAL


async def worker(
    question: BCReturn,
    agent: SealMemoryAgent,
    context: SealMemoryContext,
    judge: BaseQALLMJudge | None,
    review_logger: JsonlFileLogger,
    state: SealMemoryAgentState = SealMemoryAgentState.INITIAL,
) -> JudgeResult | None:
    """Process a single rollout end-to-end and return the judge verdict.

    Shareability notes:
    - `agent`         : safe to share — stream_run only mutates the passed context.
    - `judge`         : safe to share — async_judge builds local state and only reads self.*.
    - `review_logger` : safe to share — JsonlFileLogger serialises writes with an asyncio.Lock.
    - `context`       : must NOT be shared — owns all mutable per-question state.

    Streaming segment writes
    ------------------------
    New messages are flushed to the current segment_{k}.jsonl as soon as each
    Observation arrives.  On a SEAL event SegmentFileLogger.rotate() closes the
    current file and opens segment_{k+1}.jsonl so the file reflects live progress.
    The final close() in the finally block flushes any tail messages regardless
    of whether the run succeeded or raised an exception.
    """
    answer: str | None = None
    judge_result: JudgeResult | None = None
    error: str | None = None

    q_dir = question_dir(context.rollout_dir, context.question_id)
    seg_logger = SegmentFileLogger(q_dir)

    try:
        async for ev in agent.stream_run(context):
            if not isinstance(ev, Observation):
                continue

            if ev.state == AgentState.INITIAL and ev.scope[-1] == "BaseAgent":
                continue

            if ev.scope[-1] == "SealMemoryAgent":
                # Save the segment when next state is SEAL, save before SEAL state clears the trajectory
                if ev.state == SealMemoryAgentState.SEAL:
                    # Flush remaining messages from this segment, then rotate.
                    # The trajectory is still intact here — handle_seal_memory has
                    # not run yet — so we can catch any messages the last BaseAgent
                    # event may not have surfaced.
                    await seg_logger.rotate(context.trajectory.serialized_messages)
                if ev.state == SealMemoryAgentState.END:
                    break
                continue

            # BaseAgent event: stream any messages added since the last flush.
            await seg_logger.write(context.trajectory.serialized_messages)

        # Flush any messages that arrived after the last BaseAgent event
        # (e.g. the final tool-result message before END).
        await seg_logger.write(context.trajectory.serialized_messages)

        answer = parse_answer(context)
        if not answer:
            error = "Answer extraction failed."
            judge_result = JudgeResult(is_correct=False, complete_status=JudgeCompleteStatus.EXTRACTION_FAILED)
        elif judge is not None:
            judge_result = await judge.async_judge(
                question=question.question,
                ground_truth=question.ground_truth,
                answer=answer,
            )


    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    finally:
        await seg_logger.close(context.trajectory.serialized_messages)

    # --- Persist review record ---
    metric_logger.rate(
        context.question_id, 
        "is_correct", 
        judge_result.is_correct if judge_result is not None else False, 
        context.rollout_idx
    )
    end_reason = context.trajectory.end_reason
    review_record = {
        "question_id": context.question_id,
        "question": question.question,
        "ground_truth": question.ground_truth,
        "answer": answer,
        "is_correct": judge_result.is_correct if judge_result is not None else None,
        "score": judge_result.score if judge_result is not None else None,
        "judge_reasoning": judge_result.reasoning if judge_result is not None else None,
        "judge_raw_response": judge_result.raw_response if judge_result is not None else None,
        "judge_complete_status": (
            judge_result.complete_status.name if judge_result is not None else None
        ),
        "exit_reason": end_reason.name if end_reason is not None else None,
        "error": error,
    }
    await review_logger.write_log(review_record)

    return judge_result


async def _main() -> None:
    args = build_args()
    backend_parser = (
        LLMBackend[args.backend_parser.strip().upper()]
        if args.backend_parser
        else None
    )

    _prepare_tools(
        args.tool_config,
        search_tool_function=args.search_tool_function,
        link_summary_tool_function=args.link_summary_tool_function,
    )
    tools = tool_registry.get_schema(args.allowed_tools)

    runtime = RuntimeConfig(
        main_llm=_get_llm_config(args, "agent"),
        judge_llm=_get_llm_config(args, "judge"),
        completion_cap=10_000,
        work_dir=args.work_dir,
        tool_loop_guard=args.tool_loop_guard,
        tool_loop_guard_max_cycle=args.tool_loop_guard_max_cycle,
        add_turn_budget=args.add_turn_budget,
        add_token_budget=args.add_token_budget,
        add_seal_budget=args.add_seal_budget,
        enable_budget_prompt=args.enable_budget_prompt,
        prompt_language=args.prompt_language,
    )
    token_estimator = (
        tokenizer_estimator(args.tokenizer_path)
        if args.tokenizer_path
        else None
    )
    runtime.token_estimator = token_estimator

    retry_on_length_or_malformed = bool(getattr(args, "retry_on_length_or_malformed", False))
    base_handler = BaseAgentHandler()
    if retry_on_length_or_malformed:
        # Fall back to the legacy no-answer retry only when no SubmitTool is used.
        retry_hook = (
            LengthOrMalformedRetry()
            if tool_registry.has_submit_tool(args.allowed_tools)
            else NoToolAndAnswerRetry()
        )
        base_handler.register_turn_retry_hook([retry_hook])

    base_agent = BaseAgent(handler=base_handler)
    seal_handler = SealAgentHandler(base_agent=base_agent)
    agent = SealMemoryAgent(seal_handler=seal_handler)

    dataset = get_bc_dataset(args.dataset_path, getattr(args, "num_questions"))

    judge_payload = {
        "prompt_set": "bc_judge",
        "client_config": runtime.judge_llm,
        "max_retry_times": args.judge_max_retry,
    }
    judge_cls = get_verifier_cls("bc_qa_llm_judge")
    judge = judge_cls(**judge_payload)

    context_builder = SealContextFactory(
        runtime=runtime,
        tools=tools,
        allowed_tools=args.allowed_tools,
        max_turns=args.max_turns,
        backend_parser=backend_parser,
    )
    review_loggers = [
        JsonlFileLogger(
            str(review_jsonl_path(Path(args.work_dir) / f"rollout_{i}"))
        )
        for i in range(args.rollout_n)
    ]
    controller = GenerationController(
        args=args,
        agent=agent,
        judge=judge,
        context_factory=context_builder,
        worker_func=worker,
        review_loggers=review_loggers,
        should_judge=not args.disable_judge,
    )

    await controller.run(dataset)

    metric_logger.write_report(Path(args.work_dir) / "metric_report.json")

    for logger in review_loggers:
        await logger.close()


if __name__ == "__main__":
    asyncio.run(_main())
