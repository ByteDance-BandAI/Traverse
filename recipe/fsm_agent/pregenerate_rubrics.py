# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import recipe.fsm_agent.fsm_prompt  # DON'T delete — registers all prompt sets
import traceback

from pathlib import Path
from agents.base_agent_suit import BaseAgent, BaseAgentHandler
from agents.events import Observation
from agents.seal_memory_agent import SealAgentHandler, SealMemoryAgent
from agents.self_verify_agent import FSMAgent, FSMAgentHandler, FSMContext, FSMAgentState, Rubric
from config import RuntimeConfig
from constants import AgentState, AgentTrajectory, LLMBackend, RunContext, question_dir
from dataset import get_bc_dataset
from dataset.bc_dataset import BCReturn
from loggers.fsm_state_logger import FSMStateLogger
from recipe.fsm_agent.arguments import build_args
from recipe.fsm_agent.main import _configure_default_executor
from rollout.retries import LengthOrMalformedRetry, RubricParseRetry
from rollout.utils import _get_llm_config, _prepare_tools
from rollout.context_factory import BaseContextFactory
from rollout.rollout_manager import GenerationController
from utils import tokenizer_estimator
from dataclasses import dataclass
from typing import Tuple


RubricsMap = dict[str, list[Rubric]]


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def save_rubrics(rubrics_map: RubricsMap, path: str) -> None:
    """Persist a {question_id -> [Rubric, ...]} map as a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        qid: {"rubrics": [{"id": r.id, "description": r.description} for r in rubrics]}
        for qid, rubrics in rubrics_map.items()
    }
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


def save_failures_path(path: str) -> Path:
    return Path(path).with_suffix(Path(path).suffix + ".errors.json")


def save_failures(failures: dict[str, dict[str, str]], path: str) -> None:
    """Persist per-question rubric generation failures without aborting the batch."""
    output_path = save_failures_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


def load_rubrics(path: str, valid_question_ids: set[str]) -> RubricsMap:
    """Load existing rubrics for resume; ignore empty entries and unrelated qids."""
    output_path = Path(path)
    if not output_path.exists():
        return {}
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"WARNING: cannot parse existing rubric file {path}: {exc}; starting fresh.")
        return {}

    rubrics_map: RubricsMap = {}
    for qid, item in data.items():
        if qid not in valid_question_ids:
            continue
        rubrics: list[Rubric] = []
        for r in item.get("rubrics", []):
            if not isinstance(r, dict) or "description" not in r:
                continue
            # Keep existing IDs verbatim for resume compatibility. Older rubric
            # files may use semantic string IDs such as "format".
            rubrics.append(
                Rubric(id=r.get("id", len(rubrics) + 1), description=str(r["description"]))
            )
        if rubrics:
            rubrics_map[qid] = rubrics
    return rubrics_map


async def run_rubric_with_trajectory_log(
    agent: FSMAgent,
    context: FSMContext,
    question_dir: Path,
    state: FSMAgentState = FSMAgentState.RUBRIC,
) -> None:
    """Run the rubric state directly while mirroring normal rollout JSONL logs."""
    state_logger = FSMStateLogger(question_dir)
    await state_logger.transition(
        state,
        context.transition_cnt,
        context.trajectory.serialized_messages,
    )

    try:
        async for ev in agent.fsm_handler[state](context):
            if not isinstance(ev, Observation):
                continue

            if ev.state == FSMAgentState.ERROR:
                await state_logger.error(context.trajectory.serialized_messages)
                continue

            if ev.scope[-1] == "BaseAgent":
                if ev.state == AgentState.INITIAL:
                    continue
                await state_logger.write(context.trajectory.serialized_messages)

        await state_logger.write(context.trajectory.serialized_messages)
    finally:
        await state_logger.close(context.trajectory.serialized_messages)


class FSMRubricContextFactory(BaseContextFactory):
    def __init__(
        self,
        runtime: RuntimeConfig,
        backend_parser: LLMBackend | None,
    ):
        self.runtime = runtime
        self.backend_parser = backend_parser

    async def create(self, question: BCReturn, rollout_idx: int) -> Tuple[RunContext, AgentState]:
        context = FSMContext(
            runtime_config=self.runtime,
            trajectory=AgentTrajectory(
                messages=[],
                backend_parser=self.backend_parser,
            ),
            tools=[],
            max_turns=1,
        )
        context.question = question.question
        context.question_id = question.question_id
        context.rollout_dir = Path(self.runtime.work_dir) / f"rollout_{rollout_idx}"
        context.transition_cnt = 1
        context.prompt_set = "fsm_rubric_gen"
        return context, FSMAgentState.RUBRIC


class NullReviewLogger:
    async def remove_log_by_question_id(self, question_id) -> bool:
        return False

    async def pop_log_by_question_id(self, question_id):
        return None

    async def close(self):
        return None


class RubricCollector(NullReviewLogger):
    def __init__(self, output_path: str, rubrics_map: RubricsMap, resume: bool):
        self.rubrics_map = rubrics_map
        self.failures: dict[str, dict[str, str]] = {}
        self._lock = asyncio.Lock()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not resume:
            self.output_path.write_text("", encoding="utf-8")

    async def collect(self, question_id: str, rubrics: list[Rubric]) -> None:
        async with self._lock:
            self.rubrics_map[question_id] = rubrics
            record = {
                "question_id": question_id,
                "rubrics": [{"id": r.id, "description": r.description} for r in rubrics],
            }
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    async def record_failure(self, question_id: str, exc: BaseException) -> None:
        """Archive a single failed question instead of aborting the whole batch."""
        async with self._lock:
            self.failures[question_id] = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__, limit=20)
                ),
            }
            save_failures(self.failures, str(self.output_path))

    def _aggregated_path(self) -> Path:
        candidate = self.output_path.with_suffix(".json")
        # Never let the aggregate clobber the streamed file it is derived from.
        if candidate == self.output_path:
            candidate = self.output_path.with_suffix(".aggregated.json")
        return candidate

    async def flush(self) -> None:
        # The streamed JSONL is the primary artifact; also emit the aggregated
        # dict-shaped JSON that the RL dataset builders read.
        async with self._lock:
            save_rubrics(self.rubrics_map, str(self._aggregated_path()))
            if self.failures:
                print(
                    f"WARNING: rubric generation failed for {len(self.failures)} questions. "
                    f"See {save_failures_path(str(self.output_path))}."
                )


@dataclass
class PregenerateResult:
    is_correct: bool


class StaticRowsDataset:
    def __init__(self, rows: list[BCReturn]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    async def iter_rows(self):
        for row in self.rows:
            yield row
            await asyncio.sleep(0)


def load_streamed_rubrics(path: str) -> RubricsMap:
    output_path = Path(path)
    if not output_path.exists():
        return {}

    loaded: RubricsMap = {}
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                continue
            row = json.loads(line)
            loaded[row["question_id"]] = [
                Rubric(id=r["id"], description=r["description"], satisfied=False)
                for r in row["rubrics"]
            ]
    return loaded


async def rubric_worker(
    question: BCReturn,
    agent: FSMAgent,
    context: FSMContext,
    judge,
    review_logger: "RubricCollector",
    state: FSMAgentState = FSMAgentState.RUBRIC,
) -> PregenerateResult:
    question_id = str(question.question_id)
    try:
        # Drain only the rubric handler, mirroring normal rollout JSONL logs so a
        # failed generation can still be inspected per question.
        await run_rubric_with_trajectory_log(
            agent=agent,
            context=context,
            question_dir=question_dir(context.rollout_dir, question_id),
            state=state,
        )
    except Exception as exc:
        await review_logger.record_failure(question_id, exc)
        return PregenerateResult(is_correct=False)

    await review_logger.collect(question_id, context.rubrics)
    return PregenerateResult(is_correct=True)


async def _main() -> None:
    args = build_args()
    if args.rollout_n != 1:
        raise ValueError("pregenerate_rubrics requires --rollout_n 1 to avoid duplicated rubric outputs.")
    backend_parser = (
        LLMBackend[args.backend_parser.strip().upper()]
        if args.backend_parser
        else None
    )
    if not args.pregenerate_rubrics_output:
        raise ValueError("Please provide --pregenerate_rubrics_output for rubric pre-generation.")

    _configure_default_executor(args.concurrency)

    _prepare_tools(
        args.tool_config,
        search_tool_function=args.search_tool_function,
        link_summary_tool_function=args.link_summary_tool_function,
    )

    runtime = RuntimeConfig(
        main_llm=_get_llm_config(args, "agent"),
        completion_cap=int(os.getenv("RUBRIC_COMPLETION_CAP", "32768")),
        work_dir=args.work_dir,
        add_turn_budget=args.add_turn_budget,
        add_token_budget=args.add_token_budget,
        add_seal_budget=args.add_seal_budget,
        enable_budget_prompt=args.enable_budget_prompt,
        prompt_language=args.prompt_language,
    )
    runtime.main_llm.include_thoughts = env_flag("RUBRIC_INCLUDE_THOUGHTS", False)
    token_estimator = (
        tokenizer_estimator(args.tokenizer_path)
        if args.tokenizer_path
        else None
    )
    runtime.token_estimator = token_estimator

    base_handler = BaseAgentHandler()
    if args.retry_on_length_or_malformed:
        base_handler.register_turn_retry_hook([LengthOrMalformedRetry()])
    base_agent = BaseAgent(handler=base_handler)
    seal_handler = SealAgentHandler(base_agent=base_agent)
    seal_agent = SealMemoryAgent(seal_handler=seal_handler)
    fsm_handler = FSMAgentHandler(
        inner_agent=seal_agent,
        rubric_retry_hook=RubricParseRetry() if args.rubric_parse_retry else None,
    )
    agent = FSMAgent(fsm_handler=fsm_handler)

    dataset = get_bc_dataset(args.dataset_path, getattr(args, "num_questions"))
    existing_rubrics = load_streamed_rubrics(args.pregenerate_rubrics_output) if args.resume else {}
    completed_qids = set(existing_rubrics.keys())
    pending_rows: list[BCReturn] = []
    async for row in dataset.iter_rows():
        if row.question_id not in completed_qids:
            pending_rows.append(row)
    dataset = StaticRowsDataset(pending_rows)
    print(
        f"Pregenerate resume: loaded {len(existing_rubrics)} completed rubrics, "
        f"remaining {len(pending_rows)} questions."
    )

    context_factory = FSMRubricContextFactory(
        runtime=runtime,
        backend_parser=backend_parser,
    )
    collector = RubricCollector(
        output_path=args.pregenerate_rubrics_output,
        rubrics_map=existing_rubrics,
        resume=args.resume,
    )
    controller = GenerationController(
        args=args,
        agent=agent,
        judge=None,
        context_factory=context_factory,
        worker_func=rubric_worker,
        review_loggers=[collector for _ in range(args.rollout_n)],
        should_judge=False,
    )
    await controller.run(dataset)
    await collector.flush()
    print(f"Rubrics saved to {args.pregenerate_rubrics_output}")


if __name__ == "__main__":
    asyncio.run(_main())
