# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from logging import getLogger

from loggers.logging_config import setup_logging

setup_logging()

import recipe.fsm_agent.fsm_prompt  # DON'T delete — registers all prompt sets
from tools import tool_registry
from agents.base_agent_suit import BaseAgentHandler, BaseAgent
from agents.seal_memory_agent import SealAgentHandler, SealMemoryAgent
from agents.self_verify_agent import FSMAgent, FSMAgentHandler, FSMAgentState, FSMContext, Rubric
from agents.events import Observation
from constants import (
    AgentState,
    AgentTrajectory,
    LLMBackend,
    MessageRole,
    RunContext,
    question_dir,
)
from config import RuntimeConfig
from dataset.bc_dataset import BCReturn, BrowseCompDataset
from dataset.benchmark_registry import create_benchmark_dataset
from recipe.fsm_agent.arguments import build_args
from verifiers import JudgeCompleteStatus, get_verifier_cls, JudgeResult
from verifiers.base_qa_llm_judge import BaseQALLMJudge
from verifiers.utils import answer_parser
from loggers.file_logger import JsonlFileLogger
from loggers.fsm_state_logger import FSMStateLogger
from loggers.metric_logger import metric_logger
from utils import tokenizer_estimator
from rollout.context_factory import BaseContextFactory
from rollout.utils import _prepare_tools, _get_llm_config
from rollout.retries import (
    FSMNoToolAndAnswerRetry,
    LengthOrMalformedRetry,
    FSMTurnParseRetry,
    RubricParseRetry,
    VerifyParseRetry,
    OverTurnRetry,
    OverSealRetry,
)
from rollout.rollout_manager import GenerationController
from rollout.review_paths import review_jsonl_path
from recipe.fsm_agent.source_difficulty_filter import (
    load_source_difficulty_question_ids,
)
from recipe.fsm_agent.benchmark_summary import (
    summarize_dsqa_reviews,
    summarize_widesearch_reviews,
)

logger = getLogger(__name__)

RubricsMap = Dict[str, List[Rubric]]


class QuestionIdFilteredDataset:
    def __init__(
        self,
        dataset: BrowseCompDataset,
        allowed_question_ids: set[str],
        limit: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.allowed_question_ids = allowed_question_ids
        self.limit = limit

    def __len__(self) -> int:
        total = min(len(self.dataset), len(self.allowed_question_ids))
        if self.limit is not None:
            total = min(total, self.limit)
        return total

    async def iter_rows(self):
        yielded = 0
        async for row in self.dataset.iter_rows():
            if str(row.question_id) not in self.allowed_question_ids:
                continue
            yield row
            yielded += 1
            if self.limit is not None and yielded >= self.limit:
                return


def _configure_default_executor(concurrency: int) -> None:
    raw_workers = os.getenv("LLM_THREAD_POOL_WORKERS", str(concurrency)).strip()
    try:
        workers = int(raw_workers)
    except ValueError as exc:
        raise ValueError(f"Invalid LLM_THREAD_POOL_WORKERS={raw_workers!r}") from exc
    if workers <= 0:
        raise ValueError(f"LLM_THREAD_POOL_WORKERS must be positive, got {workers}")

    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm")
    )
    print(f"Configured asyncio default executor max_workers={workers}")


def _clone_runtime_config(runtime: RuntimeConfig) -> RuntimeConfig:
    return replace(
        runtime,
        main_llm=replace(runtime.main_llm) if runtime.main_llm else None,
        judge_llm=replace(runtime.judge_llm) if runtime.judge_llm else None,
    )


def _has_material_han_text(text: str) -> bool:
    han_count = sum(
        1
        for char in text
        if "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
    )
    non_space_count = sum(1 for char in text if not char.isspace())
    if non_space_count == 0:
        return False
    # Avoid switching templates for an English question that only quotes a
    # short Chinese name/title.
    return han_count >= 8 or (han_count >= 4 and han_count / non_space_count >= 0.05)


def _resolve_prompt_language(prompt_language: str, question: str) -> str:
    if prompt_language != "auto":
        return prompt_language
    return "zh" if _has_material_han_text(question) else "en"


def load_rubrics(path: str) -> RubricsMap:
    """Load a rubrics JSON/JSONL file produced by save_rubrics."""
    def parse_rubrics(data: list) -> List[Rubric]:
        return [Rubric(id=r["id"], description=r["description"], satisfied=False) for r in data]

    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return {
                row["question_id"]: parse_rubrics(row["rubrics"])
                for line in f if (row := json.loads(line))
            }
        return {qid: parse_rubrics(entry["rubrics"]) for qid, entry in json.load(f).items()}


def _build_turn_retry_hooks(args) -> list:
    """Constructs the appropriate retry hooks based on arguments."""
    hooks = []
    if args.rubric_parse_retry or args.answer_parse_retry or args.verify_parse_retry:
        hooks.append(FSMTurnParseRetry(
            retry_rubric_parse=args.rubric_parse_retry,
            retry_answer_parse=args.answer_parse_retry,
            retry_verify_parse=args.verify_parse_retry,
        ))
    
    if args.retry_on_tool_answer:
        if tool_registry.has_submit_tool(args.allowed_tools):
            raise RuntimeError("Cannot retry on tool & answer when using submit tool!")
        hooks.append(FSMNoToolAndAnswerRetry())
    elif args.retry_on_length_or_malformed:
        hooks.append(LengthOrMalformedRetry())

    return hooks


class FSMContextFactory(BaseContextFactory):
    def __init__(
        self,
        runtime: RuntimeConfig,
        tools: list,
        allowed_tools: list,
        max_turns: int,
        skip_rubrics: bool,
        skip_verify: bool,
        disable_rubric_reivision: bool,
        args=None,
        no_rubric_verify: bool = False,
        max_verify: int = 3,
        max_verify_turns: int = 128,
        over_turn_retry: int = -1,
        over_seal_retry: int = -1,
        token_estimator=None,
        pregenerated_rubrics: RubricsMap | None = None,
    ):
        self.runtime = runtime
        self.tools = tools
        self.args = args
        self.allowed_tools = allowed_tools
        self.max_turns = max_turns
        self.skip_rubrics = skip_rubrics
        self.skip_verify = skip_verify
        self.disable_rubric_reivision = disable_rubric_reivision
        self.no_rubric_verify = no_rubric_verify
        self.max_verify = max_verify
        self.max_verify_turns = max_verify_turns
        self.over_turn_retry = over_turn_retry
        self.over_seal_retry = over_seal_retry
        self.token_estimator = token_estimator
        self.pregenerated_rubrics = pregenerated_rubrics
        backend_parser = getattr(args, "backend_parser", None)
        self.backend_parser = (
            LLMBackend[backend_parser.strip().upper()] if backend_parser else None
        )

    async def create(self, question: BCReturn, rollout_idx: int) -> Tuple[RunContext, AgentState]:
        pregened = self.pregenerated_rubrics.get(question.question_id) if self.pregenerated_rubrics else None
        runtime_config = _clone_runtime_config(self.runtime)
        runtime_config.prompt_language = _resolve_prompt_language(
            runtime_config.prompt_language,
            question.question,
        )
        # Important: Create fresh context here
        context = FSMContext(
            runtime_config=runtime_config,
            trajectory=AgentTrajectory(messages=[], backend_parser=self.backend_parser),
            tools=self.tools,
            allowed_tool_names=self.allowed_tools,
            benchmark=question.benchmark,
            skip_rubrics=self.skip_rubrics,
            skip_verify=self.skip_verify,
            disable_rubric_reivision=self.disable_rubric_reivision,
            no_rubric_verify=self.no_rubric_verify,
            max_answer_turns=self.max_turns,
            max_verify=self.max_verify,
            max_verify_turns=self.max_verify_turns,
            over_turn_retry=self.over_turn_retry,
            over_seal_retry=self.over_seal_retry,
        )
        
        if pregened:
            context.rubrics = pregened
            
        # Extended Context Initialization
        context.seal_count = 0
        context.compaction_token_threshold = getattr(
            self.args, "compaction_token_threshold", 0
        )
        context.question = question.question
        context.question_id = question.question_id
        context.rollout_dir = Path(self.runtime.work_dir) / f"rollout_{rollout_idx}"
        context.rollout_idx = rollout_idx
        context.max_turns = self.max_turns
        context.prompt_set = "fsm_answer"
        
        if self.token_estimator:
            context.runtime_config.token_estimator = self.token_estimator

        return context, FSMAgentState.ANSWER if pregened else FSMAgentState.INITIAL


async def worker(
    question: BCReturn,
    agent: FSMAgent,
    context: FSMContext,
    judge: BaseQALLMJudge | None,
    review_logger: JsonlFileLogger,
    state: FSMAgentState = FSMAgentState.INITIAL,
) -> JudgeResult | None:
    """Process a single rollout end-to-end and return the judge verdict.

    Shareability notes:
    - `agent`         : safe to share — stream_run only mutates the passed context.
    - `judge`         : safe to share — async_judge builds local state and reads self.* only.
    - `review_logger` : safe to share — JsonlFileLogger serialises writes with asyncio.Lock.
    - `context`       : must NOT be shared — owns all mutable per-question state.

    Event-scope hierarchy emitted by FSMAgent
    -----------------------------------------
    ("FSMAgent",)                               → FSM state boundary
    ("FSMAgent", "BaseAgent")                   → BaseAgent events during RUBRIC or max-turn final answer
    ("FSMAgent", "SealMemoryAgent")             → SealMemory state during ANSWER/VERIFY
    ("FSMAgent", "SealMemoryAgent", "BaseAgent") → BaseAgent events during ANSWER/VERIFY

    File naming under question_{qid}/
    ----------------------------------
    rubric_t{tc}_attempt_{i}_{k}.jsonl — one file per seal-segment in each rubric attempt
    answer_t{tc}_attempt_{i}_{k}.jsonl — one file per seal-segment in each answer attempt
    verify_t{tc}_attempt_{i}_{k}.jsonl — one file per seal-segment in each verify attempt
    i=0 when no exception happened, i increments after each exception/retry
    memory/                     — written directly by seal_memory_tool
    """
    answer: str | None = None
    judge_result: JudgeResult | None = None
    error: str | None = None
    last_active_fsm_state: FSMAgentState | None = None

    state_logger = FSMStateLogger(question_dir(context.rollout_dir, context.question_id))

    try:
        async for ev in agent.stream_run(context, state=state):
            if not isinstance(ev, Observation):
                continue

            if ev.state == FSMAgentState.ERROR:
                await state_logger.error(context.trajectory.serialized_messages)
                continue

            match ev.scope:
                case ("FSMAgent",):
                    if ev.state in (FSMAgentState.RUBRIC, FSMAgentState.ANSWER, FSMAgentState.VERIFY):
                        last_active_fsm_state = ev.state
                        context.transition_cnt += 1
                        await state_logger.transition(ev.state, context.transition_cnt, context.trajectory.serialized_messages)
                    elif ev.state == FSMAgentState.END:
                        break
                        
                case (*_, agent_name) if agent_name in (
                    SealMemoryAgent.name,
                    "ExternalSealMemoryAgent",
                    "PeriodicCompactionAgent",
                ):
                    if ev.state.name in ("SEAL", "COMPACT"):
                        await state_logger.seal(context.trajectory.serialized_messages)
                        
                case (*_, "BaseAgent"):
                    if ev.state != AgentState.INITIAL:
                        await state_logger.write(context.trajectory.serialized_messages)

        await state_logger.write(context.trajectory.serialized_messages)

        # Prefer the structured answer extracted by FSMAgent; fall back to parsing
        # the last assistant message for robustness, but only if the last active
        # FSM state before loop exit was ANSWER.
        answer = context.current_answer or None
        if (
            not answer
            and last_active_fsm_state == FSMAgentState.ANSWER
            and context.trajectory.messages
            and context.trajectory.messages[-1].role == MessageRole.ASSISTANT
        ):
            answer = answer_parser(
                context.trajectory.messages[-1],
                strict=True,
                prefer_markdown_table=question.benchmark == "widesearch_200",
            )

        if not answer:
            error = "Answer extraction failed."
            judge_result = JudgeResult(
                is_correct=False,
                complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
            )
        elif judge is not None:
            if question.judge_metadata:
                judge_result = await judge.async_judge(
                    question=question.question,
                    ground_truth=question.ground_truth,
                    answer=answer,
                    metadata=question.judge_metadata,
                )
            else:
                judge_result = await judge.async_judge(
                    question=question.question,
                    ground_truth=question.ground_truth,
                    answer=answer,
                )

    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    finally:
        await state_logger.close(context.trajectory.serialized_messages)

    end_reason = context.trajectory.end_reason
    is_correct = judge_result.is_correct if judge_result is not None else False
    metric_logger.rate(context.question_id, "is_correct", is_correct, context.rollout_idx)

    await review_logger.write_log({
        "benchmark": question.benchmark,
        "question_id": context.question_id,
        "question": question.question,
        "ground_truth": question.ground_truth,
        "answer": answer,
        "is_correct": is_correct if judge_result is not None else None,
        "score": judge_result.score if judge_result is not None else None,
        "judge_metrics": judge_result.metrics if judge_result is not None else {},
        "judge_reasoning": judge_result.reasoning if judge_result is not None else None,
        "judge_raw_response": judge_result.raw_response if judge_result is not None else None,
        "judge_complete_status": (
            judge_result.complete_status.name if judge_result is not None else None
        ),
        "exit_reason": end_reason.name if end_reason is not None else None,
        "verify_cnt": context.verify_cnt,
        "transition_cnt": context.transition_cnt,
        "seal_count": context.seal_count,
        "rubrics_count": len(context.rubrics),
        "num_turns": context.trajectory.turn,
        "answer_turn_count": context.answer_turn_count,
        "answer_attempt_count": context.answer_attempt_count,
        "answer_attempt_turn_count": context.answer_attempt_turn_count,
        "verify_turn_count": context.verify_turn_count,
        "verify_attempt_turn_count": context.verify_attempt_turn_count,
        "max_answer_turns": context.max_answer_turns,
        "max_verify": context.max_verify,
        "max_verify_turns": context.max_verify_turns,
        "verify_limit_reached": context.verify_limit_reached,
        "verify_turn_limit_reached": context.verify_turn_limit_reached,
        "final_answer_after_verify_limit": context.final_answer_after_verify_limit,
        "over_turn_retry_triggered": context.over_turn_retry_triggered,
        "over_seal_retry_triggered": context.over_seal_retry_triggered,
        "over_turn_retry_disabled_final_attempt": context.over_turn_retry_disabled_final_attempt,
        "over_seal_retry_disabled_final_attempt": context.over_seal_retry_disabled_final_attempt,
        "turn_limit_final_answer_triggered": context.turn_limit_final_answer_triggered,
        "context_limit_final_answer_triggered": context.context_limit_final_answer_triggered,
        "prompt_language": context.runtime_config.prompt_language,
        "tool_loop_guard": context.runtime_config.tool_loop_guard,
        "tool_loop_guard_max_cycle": context.runtime_config.tool_loop_guard_max_cycle,
        "tool_loop_blocked_count": context.tool_loop_blocked_count,
        "agent_temperature": (
            context.runtime_config.main_llm.temperature
            if context.runtime_config.main_llm is not None
            else None
        ),
        "error": error,
    })

    return judge_result


def _build_agent_hierarchy(args, token_estimator, summary_llm) -> FSMAgent:
    """Assembles the inner memory agent and FSM handler based on arguments."""
    base_agent = BaseAgent(handler=BaseAgentHandler())
    
    if hooks := _build_turn_retry_hooks(args):
        base_agent.handler.register_turn_retry_hook(hooks)

    if args.inner_agent == "compaction":
        from agents.periodic_compaction_agent import (
            PeriodicCompactionAgent,
            PeriodicCompactionHandler,
        )

        if not token_estimator:
            raise ValueError("inner_agent='compaction' requires a token estimator (--tokenizer_path).")
        inner_agent = PeriodicCompactionAgent(handler=PeriodicCompactionHandler(base_agent=base_agent))
        
    elif args.inner_agent == "external_seal":
        from agents.external_seal_memory_agent import (
            ExternalSealAgentHandler,
            ExternalSealMemoryAgent,
        )

        if not summary_llm:
            raise ValueError("inner_agent='external_seal' requires an external summarizer (--summary_config).")
        inner_agent = ExternalSealMemoryAgent(seal_handler=ExternalSealAgentHandler(base_agent=base_agent))
        
    else:
        inner_agent = SealMemoryAgent(seal_handler=SealAgentHandler(base_agent=base_agent))

    return FSMAgent(fsm_handler=FSMAgentHandler(
        inner_agent=inner_agent,
        rubric_retry_hook=RubricParseRetry() if args.rubric_parse_retry else None,
        verify_retry_hook=VerifyParseRetry() if args.verify_parse_retry else None,
    ))


async def _main() -> None:
    args = build_args()
    _configure_default_executor(args.concurrency)

    # 2. Tools
    _prepare_tools(
        args.tool_config,
        search_tool_function=args.search_tool_function,
        link_summary_tool_function=args.link_summary_tool_function,
    )
    
    token_estimator = tokenizer_estimator(args.tokenizer_path) if args.tokenizer_path else None
    summary_llm = _get_llm_config(args, "summary") if args.inner_agent == "external_seal" else None

    runtime = RuntimeConfig(
        main_llm=_get_llm_config(args, "agent"),
        judge_llm=_get_llm_config(args, "judge"),
        summary_llm=summary_llm,
        token_estimator=token_estimator,
        completion_cap=10_000,
        work_dir=args.work_dir,
        add_turn_budget=args.add_turn_budget,
        add_token_budget=args.add_token_budget,
        add_seal_budget=args.add_seal_budget,
        enable_budget_prompt=args.enable_budget_prompt,
        prompt_language=args.prompt_language,
        rubric_completion_cap=args.rubric_max_tokens,
        tool_loop_guard=args.tool_loop_guard,
        tool_loop_guard_max_cycle=args.tool_loop_guard_max_cycle,
    )

    agent = _build_agent_hierarchy(args, token_estimator, summary_llm)

    if args.source_difficulty_review_jsonl:
        source_question_ids, source_filter_stats = load_source_difficulty_question_ids(
            args.source_difficulty_review_jsonl,
            args.source_min_num_turns,
            correct_only=args.source_difficulty_correct_only,
        )
        if not source_question_ids:
            raise ValueError(
                "Source difficulty filter selected no questions: "
                f"review_jsonl={args.source_difficulty_review_jsonl}, "
                f"min_num_turns={args.source_min_num_turns}, "
                f"correct_only={args.source_difficulty_correct_only}, "
                f"stats={source_filter_stats}"
            )
        print(
            "Loaded source difficulty filter: "
            f"{len(source_question_ids)} question ids from "
            f"{args.source_difficulty_review_jsonl}; "
            f"min_num_turns>={args.source_min_num_turns}; "
            f"correct_only={args.source_difficulty_correct_only}; "
            f"stats={source_filter_stats}"
        )
        benchmark_spec, base_dataset = create_benchmark_dataset(
            args.benchmark,
            args.dataset_path,
            limit=None,
            shuffle_buffer_size=args.dataset_shuffle_buffer_size,
        )
        dataset = QuestionIdFilteredDataset(
            base_dataset,
            source_question_ids,
            limit=getattr(args, "num_questions"),
        )
    else:
        benchmark_spec, dataset = create_benchmark_dataset(
            args.benchmark,
            args.dataset_path,
            limit=getattr(args, "num_questions"),
            shuffle_buffer_size=args.dataset_shuffle_buffer_size,
        )
    print(
        f"Benchmark: {benchmark_spec.name} | "
        f"Dataset: {base_dataset.path if args.source_difficulty_review_jsonl else dataset.path}"
    )

    pregenerated_rubrics = load_rubrics(args.pregenerated_rubrics) if args.pregenerated_rubrics else None
    if pregenerated_rubrics:
        logger.info(f"Loaded pre-generated rubrics for {len(pregenerated_rubrics)} questions.")

    judge_payload = {
        "prompt_set": benchmark_spec.judge_prompt_set,
        "client_config": runtime.judge_llm,
        "max_retry_times": args.judge_max_retry,
    }
    judge_cls = get_verifier_cls(benchmark_spec.verifier_name)
    judge = judge_cls(**judge_payload)
    recommended_judge_model = benchmark_spec.recommended_judge_model
    actual_judge_model = getattr(runtime.judge_llm.llm_client, "model_name", "")
    if (
        recommended_judge_model
        and recommended_judge_model.lower() not in str(actual_judge_model).lower()
    ):
        print(
            "WARNING: "
            f"{benchmark_spec.name} configured evaluation recommends judge model "
            f"{recommended_judge_model!r}, but this run uses "
            f"{actual_judge_model!r}. Results will record the actual model."
        )

    context_builder = FSMContextFactory(
        runtime=runtime,
        tools=tool_registry.get_schema(args.allowed_tools),
        args=args,
        allowed_tools=args.allowed_tools,
        max_turns=args.max_turns,
        skip_rubrics=args.skip_rubrics,
        skip_verify=args.skip_verify,
        disable_rubric_reivision=args.disable_rubric_reivision,
        no_rubric_verify=args.no_rubric_verify,
        max_verify=args.max_verify,
        max_verify_turns=args.max_verify_turns,
        over_turn_retry=args.over_turn_retry,
        over_seal_retry=args.over_seal_retry,
        token_estimator=token_estimator,
        pregenerated_rubrics=pregenerated_rubrics,
    )
    
    review_loggers = [
        JsonlFileLogger(
            str(review_jsonl_path(Path(args.work_dir) / f"rollout_{i}"))
        )
        for i in range(args.rollout_n)
    ]
    
    controller = GenerationController(
        args=args, agent=agent, judge=judge,
        context_factory=context_builder, worker_func=worker,
        review_loggers=review_loggers,
        should_judge=not args.disable_judge,
    )
    if args.over_turn_retry >= 0:
        controller.register_retry_hook(OverTurnRetry())
    if args.over_seal_retry >= 0:
        controller.register_retry_hook(OverSealRetry())

    await controller.run(dataset)
    metric_logger.write_report(Path(args.work_dir) / "metric_report.json")

    for logger_inst in review_loggers:
        await logger_inst.close()

    if benchmark_spec.name in {
        "deepsearchqa_900",
        "widesearch_v2_1_hard_720",
    }:
        summary = summarize_dsqa_reviews(
            args.work_dir,
            args.rollout_n,
            expected_num_rows=len(dataset),
            benchmark=benchmark_spec.name,
            judge_profile=str(actual_judge_model),
        )
        print(
            f"{benchmark_spec.name} DSQA-style summary: "
            f"all_correct_rate={summary['all_correct_rate']:.6f}, "
            f"macro_precision={summary['macro_precision']:.6f}, "
            f"macro_recall={summary['macro_recall']:.6f}, "
            f"macro_f1={summary['macro_f1']:.6f}, "
            f"coverage={summary['coverage']:.6f}"
        )
    elif benchmark_spec.name == "widesearch_200":
        summary = summarize_widesearch_reviews(
            args.work_dir,
            args.rollout_n,
            expected_num_rows=len(dataset),
        )
        print(
            "WideSearch summary: "
            f"SR={summary['success_rate']:.6f}, "
            f"Row-F1={summary['row_f1']:.6f}, "
            f"Item-F1={summary['item_f1']:.6f}, "
            f"coverage={summary['coverage']:.6f}"
        )


if __name__ == "__main__":
    asyncio.run(_main())
