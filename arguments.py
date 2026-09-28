# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import argparse


class _InjectBudgetAlias(argparse.Action):
    """Map the retired --inject_budget flag onto the per-tag budget switches.

    It used to turn on the token, turn and seal tags together, and a large
    number of launch scripts still pass it.
    """

    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, default=False, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        namespace.inject_budget = True
        namespace.add_token_budget = True
        namespace.add_turn_budget = True
        namespace.add_seal_budget = True


def add_llm_client_args(parser: argparse.ArgumentParser, prefix: str, required: bool = True):
    # Creates a neat section in your --help menu (e.g., "Agent LLM client")
    group = parser.add_argument_group(f"{prefix.title()} LLM client")

    group.add_argument(
        f"--{prefix}_config",
        type=str,
        default=None,
        help="Path to a YAML LLM config defining the client and sampling params.",
    )

    # Legacy per-flag form, still used by a large number of launch scripts.
    # `_get_llm_config` prefers --{prefix}_config and falls back to these, so
    # neither form can be argparse-level required.
    group.add_argument(f"--{prefix}_client_endpoint", type=str)
    group.add_argument(f"--{prefix}_client_api_key", type=str, default=None)
    group.add_argument(f"--{prefix}_temperature", type=float)
    group.add_argument(f"--{prefix}_max_tokens", type=int)
    group.add_argument(f"--{prefix}_top_p", type=float)
    group.add_argument(f"--{prefix}_model_name", type=str)


def add_general_args(parser: argparse.ArgumentParser):
    # LLM clients
    add_llm_client_args(parser, "agent")
    add_llm_client_args(parser, "judge")
    parser.add_argument("--judge_max_retry", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=1024)

    # Dataset
    group = parser.add_argument_group("Dataset Arguments")
    group.add_argument(
        "--benchmark",
        type=str,
        default="browsecomp",
        help=(
            "Benchmark protocol to use. Canonical values: browsecomp, "
            "browsecomp_zh, gaia_text_103, xbench_2510, deepsearchqa_900, "
            "widesearch_200, and widesearch_v2_1_hard_720. Short aliases such "
            "as bc185, bczh, gaia, xbench, dsqa, widesearch, and "
            "widesearch_v2_1 are also accepted."
        ),
    )
    group.add_argument("--dataset_path", type=str)
    group.add_argument("--num_questions", type=int)
    group.add_argument(
        "--dataset_shuffle_buffer_size",
        type=int,
        default=512,
        help=(
            "Streaming shuffle buffer size for parquet rows. Set to 0 to "
            "preserve dataset order, which is useful for block-wise rollout."
        ),
    )
    group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume an interrupted run. Questions already recorded in review logs "
            "are skipped; partial trajectory directories for unfinished questions "
            "are deleted so they are re-run cleanly."
        ),
    )
    group.add_argument("--backend_parser", type=str, default=None)

    # Rollout
    group = parser.add_argument_group("Rollout Arguments")
    group.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of questions processed concurrently.",
    )
    group.add_argument(
        "--rollout_n",
        type=int,
        default=1,
        help="Number of independent trajectories to run per question.",
    )
    group.add_argument(
        "--disable_judge",
        action="store_true",
        default=False,
        help="Skip the LLM judge after each rollout (agent rollout still runs).",
    )

    # Tools
    group = parser.add_argument_group("Tool Arguments")
    group.add_argument("--tool_config", type=str)
    group.add_argument(
        "--search_tool_function",
        type=str,
        default=None,
        metavar="MODULE:FUNCTION",
        help=(
            "Required when using search_api: user-provided search callable as "
            "module:function or /path/to/file.py:function. The callable receives "
            "query=... and num=...."
        ),
    )
    group.add_argument(
        "--link_summary_tool_function",
        type=str,
        default=None,
        metavar="MODULE:FUNCTION",
        help=(
            "Required when using link_summary_tool: user-provided callable as "
            "module:function or /path/to/file.py:function. The callable receives "
            "question=... and url=...."
        ),
    )
    group.add_argument(
        "--allowed_tools",
        nargs="+",
        type=str,
        default=["search_api", "link_summary_tool"],
        help="Optional allow-list of tool names, e.g. --allowed_tools search_api link_summary_tool",
    )
    group.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help=(
            "Optional HuggingFace tokenizer path"
        ),
    )
    group.add_argument(
        "--add_turn_budget",
        action="store_true",
        default=False,
        help="Whether to inject turn budget info into tool results.",
    )
    group.add_argument(
        "--add_token_budget",
        action="store_true",
        default=False,
        help="Whether to inject token budget info into tool results.",
    )
    group.add_argument(
        "--add_seal_budget",
        action="store_true",
        default=False,
        help="Whether to inject seal budget info into the seal carryover.",
    )
    group.add_argument(
        "--enable_budget_prompt",
        action="store_true",
        default=False,
        help="Whether to write the budget explanation into the system prompt.",
    )
    group.add_argument(
        "--inject_budget",
        action=_InjectBudgetAlias,
        help=(
            "Deprecated alias for --add_token_budget --add_turn_budget "
            "--add_seal_budget."
        ),
    )
    group.add_argument(
        "--prompt_language",
        type=str,
        choices=["en", "zh", "auto"],
        default="en",
        help=(
            "Language used to render FSM prompts. Use 'auto' to choose zh/en "
            "per question when supported by the runner."
        ),
    )
    group.add_argument(
        "--tool_loop_guard",
        action="store_true",
        default=False,
        help=(
            "Answer a tool call locally, without dispatching it, once it repeats "
            "a recent cycle of identical calls: the same call three turns in a "
            "row, or an ABAB / ABCABC alternation for two rounds. The model gets "
            "a warning that it is going in circles instead of results it already "
            "has. Off dispatches every call."
        ),
    )
    group.add_argument(
        "--tool_loop_guard_max_cycle",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help=(
            "Longest repeating cycle --tool_loop_guard looks for. 1 catches only "
            "the same call repeated back-to-back, 2 adds ABAB, 3 adds ABCABC."
        ),
    )

    # Output
    parser.add_argument("--work_dir", type=str)

    # Retry behaviour
    group = parser.add_argument_group("Retry Arguments")
    group.add_argument(
        "--retry_on_length_or_malformed",
        action="store_true",
        default=False,
        help=(
            "Retry a turn when it hits the output token limit (within the context "
            "limit) or emits a malformed tool call."
        ),
    )
    group.add_argument(
        "--retry_on_tool_answer",
        action="store_true",
        default=False,
        help=(
            "Retry if no tool and answer exists, can only be enabled when submit tool is disabled!"
        ),
    )
    group.add_argument(
        "--rollout_max_retry_limit",
        type=int,
        default=3,
        help="Trajectory level retry control, set to 0 if you want to disable.",
    )
    return parser
