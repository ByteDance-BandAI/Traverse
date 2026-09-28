# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse
from arguments import add_general_args, add_llm_client_args


def build_args():
    parser = argparse.ArgumentParser("FSM Self-Verify Agent")

    parser = add_general_args(parser)
    # Optional external summarizer model, only required when
    # --inner_agent=external_seal (validated in main).
    add_llm_client_args(parser, "summary", required=False)

    group = parser.add_argument_group("FSM Agent Arguments")
    group.add_argument(
        "--inner_agent",
        type=str,
        choices=["seal", "compaction", "external_seal"],
        default="seal",
        help=(
            "Inner memory agent for ANSWER/VERIFY: 'seal' lets the model call "
            "seal_memory_tool, 'compaction' compacts automatically when the "
            "estimated context exceeds --compaction_token_threshold "
            "(requires --tokenizer_path), 'external_seal' triggers on the "
            "model's seal_memory_tool call but produces the seal with a "
            "separate model (requires --summary_config)."
        ),
    )
    group.add_argument(
        "--compaction_token_threshold",
        type=int,
        default=24000,
        help=(
            "When --inner_agent=compaction, compact memory before the next model "
            "call once the estimated context reaches this many tokens."
        ),
    )
    group.add_argument(
        "--skip_rubrics",
        action="store_true",
        default=False,
        help="Skip rubric generation and go directly to answering.",
    )
    group.add_argument(
        "--rubric_max_tokens",
        type=int,
        default=32_768,
        help=(
            "Maximum completion tokens for rubric generation/refinement calls. "
            "This does not change the per-call completion cap used by answer turns."
        ),
    )
    group.add_argument(
        "--pregenerate_rubrics_output",
        type=str,
        default=None,
        help="If set, generate rubrics for every question, save to this path, then exit.",
    )
    group.add_argument(
        "--pregenerated_rubrics",
        type=str,
        default=None,
        help="Path to a rubrics JSONL file produced by --pregenerate_rubrics_output. "
             "Rubrics are injected into each context and rubric generation is skipped.",
    )
    group.add_argument(
        "--skip_verify",
        action="store_true",
        default=False,
        help="Skip the verification step after generating an answer.",
    )
    group.add_argument(
        "--disable_rubric_reivision",
        action="store_true",
        default=False,
        help="Disable rubric revision path in verify decision; allow only PASS/REVISE_ANSWER.",
    )
    group.add_argument(
        "--no_rubric_verify",
        action="store_true",
        default=False,
        help="Rubric-free verification: the verifier judges answer correctness directly "
        "from the question (no rubrics injected into the verify prompt). Matches "
        "rubric-free verify RL training. Decision is limited to PASS/REVISE_ANSWER.",
    )
    group.add_argument(
        "--max_verify",
        type=int,
        default=3,
        help=(
            "Maximum number of Verify/revision cycles per question. An Answer "
            "rollout that still has no parseable answer after forced-final repair "
            "counts as an automatic failed Verify without calling the verifier. "
            "A Verify attempt that reaches its own turn/context limit retries Verify "
            "on the same Answer while attempts remain. "
            "If the last cycle requests revision, one final Answer rollout is "
            "generated without entering Verify again. Set to 0 to disable verification."
        ),
    )
    group.add_argument(
        "--max_verify_turns",
        type=int,
        default=128,
        help=(
            "Maximum model-call budget for each individual Verify attempt. Every "
            "Verify entry receives a fresh budget. This is independent of "
            "--max_turns, which is likewise applied separately to each Answer attempt."
        ),
    )
    group.add_argument(
        "--over_turn_retry",
        type=int,
        default=-1,
        help=(
            "If >= 0, abort and retry the whole trajectory when cumulative assistant "
            "turns across all seal-memory segments exceed this threshold during ANSWER "
            "state before a parseable answer is produced. Use -1 to disable."
        ),
    )
    group.add_argument(
        "--over_seal_retry",
        type=int,
        default=-1,
        help=(
            "If >= 0, abort and retry the whole trajectory when seal_count reaches "
            "this threshold during ANSWER state. Use -1 to disable. Mutually "
            "exclusive with --over_turn_retry."
        ),
    )
    group.add_argument(
        "--over_seal_retry_clean_start",
        action="store_true",
        default=False,
        help=(
            "Start each whole-trajectory retry without injecting retry-attempt "
            "metadata or sealed memories from earlier attempts. Previous attempts "
            "are still archived for debugging. Requires --over_seal_retry >= 0."
        ),
    )
    group.add_argument(
        "--rubric_parse_retry",
        action="store_true",
        default=False,
        help="Enable retry when rubric parsing fails (RubricGenerationError).",
    )
    group.add_argument(
        "--verify_parse_retry",
        action="store_true",
        default=False,
        help="Enable retry when verify parsing fails (VerifyGeneartionError).",
    )
    group.add_argument(
        "--answer_parse_retry",
        action="store_true",
        default=False,
        help="Enable turn-level retry when answer parsing fails.",
    )
    group.add_argument(
        "--source_difficulty_review_jsonl",
        type=str,
        default=None,
        help=(
            "Optional source-model review.jsonl used as a difficulty filter. "
            "Only questions whose source num_turns satisfy the configured "
            "threshold are run."
        ),
    )
    group.add_argument(
        "--source_min_num_turns",
        type=int,
        default=None,
        help=(
            "When --source_difficulty_review_jsonl is set, keep only questions "
            "whose source review num_turns >= this value."
        ),
    )
    group.add_argument(
        "--source_difficulty_correct_only",
        action="store_true",
        default=False,
        help=(
            "With --source_difficulty_review_jsonl, keep only rows where the "
            "source model was judged correct."
        ),
    )

    args = parser.parse_args()
    if args.max_verify < 0:
        raise ValueError("--max_verify must be non-negative.")
    if args.max_verify_turns < 0:
        raise ValueError("--max_verify_turns must be non-negative.")
    if args.over_turn_retry >= 0 and args.over_seal_retry >= 0:
        raise ValueError("Use either --over_turn_retry or --over_seal_retry, not both.")
    if args.over_seal_retry_clean_start and args.over_seal_retry < 0:
        raise ValueError(
            "--over_seal_retry_clean_start requires --over_seal_retry >= 0."
        )
    if args.dataset_shuffle_buffer_size < 0:
        raise ValueError("--dataset_shuffle_buffer_size must be non-negative.")
    if args.source_min_num_turns is not None and args.source_min_num_turns <= 0:
        raise ValueError("--source_min_num_turns must be positive when set.")
    if bool(args.source_difficulty_review_jsonl) != (
        args.source_min_num_turns is not None
    ):
        raise ValueError(
            "Use --source_difficulty_review_jsonl and --source_min_num_turns together."
        )
    return args
