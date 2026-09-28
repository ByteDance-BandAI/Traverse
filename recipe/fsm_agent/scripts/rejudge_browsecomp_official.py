# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Re-judge finished BrowseComp rollouts with the official HLE-style grader.

This script is intentionally additive: it reads existing review.jsonl files and
writes a separate JSONL with `official_*` fields. It does not modify the
original rollout outputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from config import GeneralLLMConfig
from llm_clients import OpenAIClient
from rollout.review_paths import existing_review_jsonl_paths
from verifiers import get_verifier_cls
from verifiers.bc_llm_judge import BrowseCompOfficialJudge


DEFAULT_OFFICIAL_GRADER_MODEL = "gpt-4.1-2025-04-14"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Re-judge fsm_agent BrowseComp review.jsonl with the official grader"
    )
    inputs = parser.add_argument_group("Inputs")
    inputs.add_argument(
        "--review_jsonl",
        action="append",
        default=[],
        help=(
            "Path to a review.jsonl. Can be passed multiple times. If omitted, "
            "--work_dir is used."
        ),
    )
    inputs.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Work dir containing rollout_* subdirectories with review.jsonl files.",
    )
    inputs.add_argument(
        "--response_field",
        type=str,
        default=None,
        help=(
            "Optional field containing the full model response to grade. "
            "Defaults to answer/final response fields found in review.jsonl."
        ),
    )
    inputs.add_argument(
        "--no_wrap_answer",
        action="store_true",
        default=False,
        help=(
            "Do not wrap answer-only records into the official Explanation / "
            "Exact Answer / Confidence response format."
        ),
    )
    inputs.add_argument("--limit", type=int, default=None)

    outputs = parser.add_argument_group("Outputs")
    outputs.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help=(
            "Output JSONL. If multiple inputs are provided and this is omitted, "
            "one .official_judge.jsonl file is written next to each input."
        ),
    )
    outputs.add_argument("--resume", action="store_true", default=False)
    outputs.add_argument(
        "--include_candidate_response",
        action="store_true",
        default=False,
        help="Store the exact response sent to the official grader.",
    )

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--concurrency", type=int, default=8)

    judge = parser.add_argument_group("Official judge LLM")
    judge.add_argument("--judge_client_endpoint", type=str, default=None)
    judge.add_argument("--judge_client_api_key", type=str, default=None)
    judge.add_argument(
        "--judge_model_name", type=str, default=DEFAULT_OFFICIAL_GRADER_MODEL
    )
    judge.add_argument("--judge_temperature", type=float, default=0.5)
    judge.add_argument("--judge_max_tokens", type=int, default=2048)
    judge.add_argument("--judge_top_p", type=float, default=1.0)
    judge.add_argument("--judge_max_retry", type=int, default=3)
    judge.add_argument(
        "--judge_include_thoughts",
        action="store_true",
        default=False,
        help="Pass include_thoughts=True to local LLM client wrappers.",
    )

    args = parser.parse_args()
    if not args.review_jsonl and not args.work_dir:
        parser.error("Provide at least one --review_jsonl or --work_dir.")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive.")
    return args


def build_judge(args: argparse.Namespace) -> BrowseCompOfficialJudge:
    payload = {
        "api_key": args.judge_client_api_key,
        "endpoint": args.judge_client_endpoint,
        "model_name": args.judge_model_name,
    }
    config = GeneralLLMConfig(
        llm_client=OpenAIClient(**payload),
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_tokens=args.judge_max_tokens,
        include_thoughts=args.judge_include_thoughts,
    )
    judge_cls = get_verifier_cls("bc_official_qa_llm_judge")
    return judge_cls(
        prompt_set="bc_official_judge",
        client_config=config,
        max_retry_times=args.judge_max_retry,
    )


def discover_review_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.review_jsonl]
    if args.work_dir:
        work_dir = Path(args.work_dir)
        if work_dir.is_file():
            paths.append(work_dir)
        else:
            direct_reviews = existing_review_jsonl_paths(work_dir)
            if direct_reviews:
                paths.extend(direct_reviews)
            else:
                for rollout_dir in sorted(work_dir.glob("rollout_*")):
                    paths.extend(existing_review_jsonl_paths(rollout_dir))
    deduped: list[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Review JSONL not found: {path}")
        seen.add(resolved)
        deduped.append(path)
    return deduped


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.official_judge{input_path.suffix}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def output_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id", "")), str(row.get("answer", ""))


def load_completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {output_key(row) for row in load_jsonl(path)}


def normalize_ground_truth(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def candidate_response(
    row: dict[str, Any],
    response_field: str | None,
    wrap_answer: bool,
) -> tuple[str, str]:
    if response_field:
        value = row.get(response_field)
        return "" if value is None else str(value), response_field

    for field in ("response", "assistant_response", "final_response", "raw_response"):
        value = row.get(field)
        if value:
            return str(value), field

    answer = row.get("answer")
    if answer is None:
        return "", "answer"
    answer_text = str(answer)
    if not wrap_answer:
        return answer_text, "answer"
    return (
        "Explanation: Final answer extracted from a previous rollout.\n"
        f"Exact Answer: {answer_text}\n"
        "Confidence: 100%",
        "answer",
    )


def with_official_result(
    row: dict[str, Any],
    result,
    candidate_field: str,
    candidate_text: str,
    include_candidate_response: bool,
) -> dict[str, Any]:
    fields = BrowseCompOfficialJudge.parse_grader_fields(result.raw_response)
    output = dict(row)
    output.update(
        {
            "official_judge_candidate_field": candidate_field,
            "official_is_correct": result.is_correct,
            "official_score": result.score,
            "official_judge_reasoning": result.reasoning,
            "official_judge_raw_response": result.raw_response,
            "official_judge_complete_status": result.complete_status.name,
            "official_judge_extracted_final_answer": fields.get(
                "extracted_final_answer"
            ),
            "official_judge_confidence": fields.get("confidence"),
        }
    )
    if "is_correct" in row:
        output["official_disagrees_with_existing"] = (
            bool(row.get("is_correct")) != bool(result.is_correct)
        )
    if include_candidate_response:
        output["official_judge_candidate_response"] = candidate_text
    return output


def with_official_error(
    row: dict[str, Any],
    error: str,
    candidate_field: str | None = None,
    candidate_text: str | None = None,
    include_candidate_response: bool = False,
) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "official_judge_candidate_field": candidate_field,
            "official_is_correct": False,
            "official_score": 0.0,
            "official_judge_reasoning": "",
            "official_judge_raw_response": "",
            "official_judge_complete_status": "ERROR",
            "official_judge_error": error,
        }
    )
    if include_candidate_response and candidate_text is not None:
        output["official_judge_candidate_response"] = candidate_text
    return output


async def rejudge_rows(
    rows: list[dict[str, Any]],
    *,
    judge: BrowseCompOfficialJudge,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_keys(output_path) if args.resume else set()
    todo = [
        row
        for row in rows[: args.limit]
        if not args.resume or output_key(row) not in completed
    ]
    if not todo:
        print(f"{output_path}: nothing to do.")
        return

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.concurrency)
    written = 0

    async def handle(row: dict[str, Any]) -> None:
        nonlocal written
        candidate_text, candidate_field = candidate_response(
            row,
            response_field=args.response_field,
            wrap_answer=not args.no_wrap_answer,
        )
        ground_truth = normalize_ground_truth(row.get("ground_truth"))
        if not row.get("question"):
            output = with_official_error(
                row,
                "Missing question field.",
                candidate_field,
                candidate_text,
                args.include_candidate_response,
            )
        elif not ground_truth:
            output = with_official_error(
                row,
                "Missing ground_truth field.",
                candidate_field,
                candidate_text,
                args.include_candidate_response,
            )
        else:
            try:
                async with semaphore:
                    result = await judge.async_judge(
                        question=str(row["question"]),
                        ground_truth=ground_truth,
                        answer=candidate_text,
                    )
                output = with_official_result(
                    row,
                    result,
                    candidate_field,
                    candidate_text,
                    args.include_candidate_response,
                )
            except Exception as exc:
                output = with_official_error(
                    row,
                    repr(exc),
                    candidate_field,
                    candidate_text,
                    args.include_candidate_response,
                )

        async with lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0 or written == len(todo):
                print(f"{output_path}: wrote {written}/{len(todo)}")

    await asyncio.gather(*(handle(row) for row in todo))


async def async_main() -> None:
    args = build_args()
    review_paths = discover_review_paths(args)
    judge = build_judge(args)
    print(
        "Official grader (OpenAI-compatible): "
        f"{args.judge_model_name}, "
        f"temperature={args.judge_temperature}, max_tokens={args.judge_max_tokens}"
    )
    cleared_outputs: set[Path] = set()
    for review_path in review_paths:
        output_path = (
            Path(args.output_jsonl)
            if args.output_jsonl
            else default_output_path(review_path)
        )
        if output_path.resolve() == review_path.resolve():
            raise ValueError("Refusing to overwrite the input review_jsonl.")
        if not args.resume and output_path not in cleared_outputs:
            if output_path.exists():
                output_path.unlink()
            cleared_outputs.add(output_path)
        print(f"Re-judging {review_path} -> {output_path}")
        await rejudge_rows(
            load_jsonl(review_path),
            judge=judge,
            args=args,
            output_path=output_path,
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
