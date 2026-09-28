#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BENCHMARK="${BENCHMARK:-gaia_text_103}"
# Each profile pins the judge model the benchmark was calibrated with. Two of
# them (deepsearchqa, widesearch_v2_1) use gemini-2.5-flash, which needs an
# endpoint that serves it -- set JUDGE_CLIENT_ENDPOINT, or override
# JUDGE_MODEL_NAME to a model your provider exposes.
DEFAULT_RUBRIC_MAX_TOKENS="32768"

case "${BENCHMARK}" in
  bc|bc185|browsecomp|browsecomp_185|browsecomp-185)
    CANONICAL_BENCHMARK="browsecomp"
    # BrowseComp is not auto-downloadable; point DATASET_PATH at your own copy.
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/browsecomp_185.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gpt-5.5"
    DEFAULT_JUDGE_TEMPERATURE=""
    DEFAULT_JUDGE_MAX_TOKENS=""
    # Rubrics are generated on the fly; set PREGENERATED_RUBRICS to reuse a
    # previously generated file and skip that pass.
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/bc185"
    DEFAULT_AUTO_PREPARE_DATASET="false"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="512"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="5"
    ;;
  bczh|bcch|bc_zh|browsecomp_zh|browsecomp-zh)
    CANONICAL_BENCHMARK="browsecomp_zh"
    # BrowseComp-ZH is not auto-downloadable; point DATASET_PATH at your own copy.
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/browsecomp_zh.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gpt-5.5"
    DEFAULT_JUDGE_TEMPERATURE=""
    DEFAULT_JUDGE_MAX_TOKENS=""
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/bczh"
    DEFAULT_AUTO_PREPARE_DATASET="false"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="512"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="5"
    ;;
  gaia|gaia_text|gaia_text_103|gaia-text-103)
    CANONICAL_BENCHMARK="gaia_text_103"
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/gaia_text_103.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gpt-5.5"
    DEFAULT_JUDGE_TEMPERATURE=""
    DEFAULT_JUDGE_MAX_TOKENS=""
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/gaia_text_103"
    DEFAULT_AUTO_PREPARE_DATASET="true"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="0"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="3"
    ;;
  xbench|xbench_2510|xbench-2510)
    CANONICAL_BENCHMARK="xbench_2510"
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/xbench_2510.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gpt-5.5"
    DEFAULT_JUDGE_TEMPERATURE=""
    DEFAULT_JUDGE_MAX_TOKENS=""
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/xbench_2510"
    DEFAULT_AUTO_PREPARE_DATASET="true"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="0"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="3"
    ;;
  dsqa|deepsearchqa|deepsearchqa_900|deepsearchqa-900)
    CANONICAL_BENCHMARK="deepsearchqa_900"
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/deepsearchqa_900.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gemini-2.5-flash"
    DEFAULT_JUDGE_TEMPERATURE="0.0"
    DEFAULT_JUDGE_MAX_TOKENS=""
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/deepsearchqa_900"
    DEFAULT_AUTO_PREPARE_DATASET="true"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="0"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="3"
    ;;
  widesearch|wide_search|wide-search|widesearch_200|widesearch-200)
    CANONICAL_BENCHMARK="widesearch_200"
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/widesearch_200.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gpt-4.1-2025-04-14"
    DEFAULT_JUDGE_TEMPERATURE="0.0"
    DEFAULT_JUDGE_MAX_TOKENS="10240"
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/widesearch_200"
    DEFAULT_AUTO_PREPARE_DATASET="true"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="0"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="3"
    ;;
  widesearch_v2_1|widesearch-v2-1|widesearch_v2_1_hard_720|widesearch-v2-1-hard-720|widesearch_hard_720|widesearch-hard-720)
    CANONICAL_BENCHMARK="widesearch_v2_1_hard_720"
    DEFAULT_DATASET_PATH="${REPO_ROOT}/data/benchmarks/widesearch_v2_1_hard_720.parquet"
    DEFAULT_JUDGE_MODEL_NAME="gemini-2.5-flash"
    DEFAULT_JUDGE_TEMPERATURE="0.0"
    DEFAULT_JUDGE_MAX_TOKENS=""
    DEFAULT_PREGENERATED_RUBRICS=""
    DEFAULT_WORK_DIR="${REPO_ROOT}/results/widesearch_v2_1_hard_720"
    # This is a locally derived dataset. Build it explicitly with
    # prepare_widesearch_v2_1_hard_720.py so source/selection provenance is clear.
    DEFAULT_AUTO_PREPARE_DATASET="false"
    DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE="0"
    DEFAULT_ROLLOUT_MAX_RETRY_LIMIT="3"
    ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}. Use bc185, bczh, gaia, xbench, dsqa, widesearch, or widesearch_v2_1." >&2
    exit 2
    ;;
esac

DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
AUTO_PREPARE_DATASET="${AUTO_PREPARE_DATASET:-${DEFAULT_AUTO_PREPARE_DATASET}}"
if [[ ! -f "${DATASET_PATH}" ]]; then
  if [[ "${AUTO_PREPARE_DATASET}" != "true" || "${DATASET_PATH}" != "${DEFAULT_DATASET_PATH}" ]]; then
    echo "Dataset not found: ${DATASET_PATH}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" recipe/fsm_agent/scripts/prepare_benchmark_data.py \
    --benchmark "${CANONICAL_BENCHMARK}"
fi

TOOL_CONFIG="${TOOL_CONFIG:-recipe/fsm_agent/assets/tools.json}"
ALLOWED_TOOLS="${ALLOWED_TOOLS:-search_api link_summary_tool seal_memory_tool read_memory_tool}"
read -r -a ALLOWED_TOOLS_ARR <<< "${ALLOWED_TOOLS}"
SEARCH_TOOL_FUNCTION="${SEARCH_TOOL_FUNCTION:-}"
LINK_SUMMARY_TOOL_FUNCTION="${LINK_SUMMARY_TOOL_FUNCTION:-}"

# Defaults target a locally hosted OpenAI-compatible SGLang/vLLM endpoint.
AGENT_CLIENT_ENDPOINT="${AGENT_CLIENT_ENDPOINT:-${OPENAI_BASE_URL:-http://localhost:8000/v1}}"
AGENT_CLIENT_API_KEY="${AGENT_CLIENT_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-gpt-5.5}"
AGENT_TEMPERATURE="${AGENT_TEMPERATURE:-1.0}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-262144}"
AGENT_TOP_P="${AGENT_TOP_P:-0.95}"

JUDGE_CLIENT_ENDPOINT="${JUDGE_CLIENT_ENDPOINT:-${AGENT_CLIENT_ENDPOINT}}"
JUDGE_CLIENT_API_KEY="${JUDGE_CLIENT_API_KEY:-${OPENAI_API_KEY:-EMPTY}}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-${DEFAULT_JUDGE_MODEL_NAME}}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-${DEFAULT_JUDGE_TEMPERATURE:-1.0}}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-${DEFAULT_JUDGE_MAX_TOKENS:-8192}}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.95}"
JUDGE_MAX_RETRY="${JUDGE_MAX_RETRY:-3}"

if [[ -z "${AGENT_CLIENT_ENDPOINT}" || -z "${AGENT_MODEL_NAME}" ]]; then
  echo "AGENT_CLIENT_ENDPOINT and AGENT_MODEL_NAME are required." >&2
  exit 1
fi
if [[ -z "${JUDGE_CLIENT_ENDPOINT}" || -z "${JUDGE_MODEL_NAME}" ]]; then
  echo "JUDGE_CLIENT_ENDPOINT and JUDGE_MODEL_NAME are required." >&2
  exit 1
fi

CONCURRENCY="${CONCURRENCY:-12}"
export LLM_THREAD_POOL_WORKERS="${LLM_THREAD_POOL_WORKERS:-${CONCURRENCY}}"
ROLLOUT_N="${ROLLOUT_N:-3}"
NUM_QUESTIONS="${NUM_QUESTIONS:-}"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
MAX_TURNS="${MAX_TURNS:-512}"
MAX_VERIFY="${MAX_VERIFY:-3}"
MAX_VERIFY_TURNS="${MAX_VERIFY_TURNS:-128}"
RUBRIC_MAX_TOKENS="${RUBRIC_MAX_TOKENS:-${DEFAULT_RUBRIC_MAX_TOKENS}}"
ROLLOUT_MAX_RETRY_LIMIT="${ROLLOUT_MAX_RETRY_LIMIT:-${DEFAULT_ROLLOUT_MAX_RETRY_LIMIT}}"
PROMPT_LANGUAGE="${PROMPT_LANGUAGE:-auto}"
# Short-circuit repeated tool calls (same call 3x in a row, or ABAB/ABCABC for
# two rounds) with a "you are going in circles" note instead of dispatching
# another backend request.
TOOL_LOOP_GUARD="${TOOL_LOOP_GUARD:-false}"
TOOL_LOOP_GUARD_MAX_CYCLE="${TOOL_LOOP_GUARD_MAX_CYCLE:-3}"
DATASET_SHUFFLE_BUFFER_SIZE="${DATASET_SHUFFLE_BUFFER_SIZE:-${DEFAULT_DATASET_SHUFFLE_BUFFER_SIZE}}"
# Only used for token accounting, so any tokenizer with comparable vocabulary
# works; accepts a Hugging Face repo id or a local directory.
TOKENIZER_PATH="${TOKENIZER_PATH:-Qwen/Qwen3-8B}"
PREGENERATED_RUBRICS="${PREGENERATED_RUBRICS-${DEFAULT_PREGENERATED_RUBRICS}}"

SKIP_RUBRICS="${SKIP_RUBRICS:-false}"
SKIP_VERIFY="${SKIP_VERIFY:-true}"
NO_RUBRIC_VERIFY="${NO_RUBRIC_VERIFY:-false}"
RESUME="${RESUME:-true}"
INJECT_BUDGET="${INJECT_BUDGET:-true}"
RETRY_ON_TOOL_ANSWER="${RETRY_ON_TOOL_ANSWER:-true}"
RUBRIC_PARSE_RETRY="${RUBRIC_PARSE_RETRY:-true}"
ANSWER_PARSE_RETRY="${ANSWER_PARSE_RETRY:-true}"
VERIFY_PARSE_RETRY="${VERIFY_PARSE_RETRY:-true}"
OVER_TURN_RETRY="${OVER_TURN_RETRY:--1}"
OVER_SEAL_RETRY="${OVER_SEAL_RETRY:--1}"
OVER_SEAL_RETRY_CLEAN_START="${OVER_SEAL_RETRY_CLEAN_START:-false}"
DRY_RUN="${DRY_RUN:-false}"

EXTRA_ARGS=()
if [[ -n "${SEARCH_TOOL_FUNCTION}" ]]; then
  EXTRA_ARGS+=(--search_tool_function "${SEARCH_TOOL_FUNCTION}")
fi
if [[ -n "${LINK_SUMMARY_TOOL_FUNCTION}" ]]; then
  EXTRA_ARGS+=(--link_summary_tool_function "${LINK_SUMMARY_TOOL_FUNCTION}")
fi
if [[ -n "${NUM_QUESTIONS}" ]]; then
  EXTRA_ARGS+=(--num_questions "${NUM_QUESTIONS}")
fi
if [[ -n "${TOKENIZER_PATH}" ]]; then
  EXTRA_ARGS+=(--tokenizer_path "${TOKENIZER_PATH}")
fi
if [[ -n "${PREGENERATED_RUBRICS}" ]]; then
  if [[ ! -f "${PREGENERATED_RUBRICS}" ]]; then
    echo "PREGENERATED_RUBRICS not found: ${PREGENERATED_RUBRICS}" >&2
    exit 1
  fi
  EXTRA_ARGS+=(--pregenerated_rubrics "${PREGENERATED_RUBRICS}")
fi
if [[ "${SKIP_RUBRICS}" == "true" ]]; then EXTRA_ARGS+=(--skip_rubrics); fi
if [[ "${SKIP_VERIFY}" == "true" ]]; then EXTRA_ARGS+=(--skip_verify); fi
if [[ "${NO_RUBRIC_VERIFY}" == "true" ]]; then EXTRA_ARGS+=(--no_rubric_verify); fi
if [[ "${RESUME}" == "true" ]]; then EXTRA_ARGS+=(--resume); fi
if [[ "${INJECT_BUDGET}" == "true" ]]; then EXTRA_ARGS+=(--inject_budget); fi
if [[ "${RETRY_ON_TOOL_ANSWER}" == "true" ]]; then EXTRA_ARGS+=(--retry_on_tool_answer); fi
if [[ "${RUBRIC_PARSE_RETRY}" == "true" ]]; then EXTRA_ARGS+=(--rubric_parse_retry); fi
if [[ "${ANSWER_PARSE_RETRY}" == "true" ]]; then EXTRA_ARGS+=(--answer_parse_retry); fi
if [[ "${VERIFY_PARSE_RETRY}" == "true" ]]; then EXTRA_ARGS+=(--verify_parse_retry); fi
if [[ "${OVER_SEAL_RETRY_CLEAN_START}" == "true" ]]; then
  EXTRA_ARGS+=(--over_seal_retry_clean_start)
fi
if [[ "${TOOL_LOOP_GUARD}" == "true" ]]; then
  EXTRA_ARGS+=(--tool_loop_guard --tool_loop_guard_max_cycle "${TOOL_LOOP_GUARD_MAX_CYCLE}")
fi

echo "Benchmark: ${CANONICAL_BENCHMARK}"
echo "Dataset: ${DATASET_PATH}"
echo "Work dir: ${WORK_DIR}"
echo "Rollouts: ${ROLLOUT_N}"
echo "Concurrency: ${CONCURRENCY}"
echo "Prompt language: ${PROMPT_LANGUAGE}"
echo "Tool loop guard: ${TOOL_LOOP_GUARD} (max cycle ${TOOL_LOOP_GUARD_MAX_CYCLE})"
echo "Rubric max tokens: ${RUBRIC_MAX_TOKENS}"
echo "Rubrics: $([[ -n "${PREGENERATED_RUBRICS}" ]] && echo "${PREGENERATED_RUBRICS}" || echo self-generated)"
echo "Agent: ${AGENT_MODEL_NAME} (OpenAI-compatible), top_p=${AGENT_TOP_P}"
echo "Judge: ${JUDGE_MODEL_NAME} (OpenAI-compatible)"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "Dry run: configuration validated; model execution skipped."
  exit 0
fi

"${PYTHON_BIN}" -m recipe.fsm_agent.main \
  --benchmark "${CANONICAL_BENCHMARK}" \
  --dataset_path "${DATASET_PATH}" \
  --dataset_shuffle_buffer_size "${DATASET_SHUFFLE_BUFFER_SIZE}" \
  --concurrency "${CONCURRENCY}" \
  --rollout_n "${ROLLOUT_N}" \
  --tool_config "${TOOL_CONFIG}" \
  --allowed_tools "${ALLOWED_TOOLS_ARR[@]}" \
  --work_dir "${WORK_DIR}" \
  --max_turns "${MAX_TURNS}" \
  --max_verify "${MAX_VERIFY}" \
  --max_verify_turns "${MAX_VERIFY_TURNS}" \
  --rubric_max_tokens "${RUBRIC_MAX_TOKENS}" \
  --over_turn_retry "${OVER_TURN_RETRY}" \
  --over_seal_retry "${OVER_SEAL_RETRY}" \
  --rollout_max_retry_limit "${ROLLOUT_MAX_RETRY_LIMIT}" \
  --prompt_language "${PROMPT_LANGUAGE}" \
  --agent_client_endpoint "${AGENT_CLIENT_ENDPOINT}" \
  --agent_client_api_key "${AGENT_CLIENT_API_KEY}" \
  --agent_model_name "${AGENT_MODEL_NAME}" \
  --agent_temperature "${AGENT_TEMPERATURE}" \
  --agent_max_tokens "${AGENT_MAX_TOKENS}" \
  --agent_top_p "${AGENT_TOP_P}" \
  --judge_client_endpoint "${JUDGE_CLIENT_ENDPOINT}" \
  --judge_client_api_key "${JUDGE_CLIENT_API_KEY}" \
  --judge_model_name "${JUDGE_MODEL_NAME}" \
  --judge_temperature "${JUDGE_TEMPERATURE}" \
  --judge_max_tokens "${JUDGE_MAX_TOKENS}" \
  --judge_top_p "${JUDGE_TOP_P}" \
  --judge_max_retry "${JUDGE_MAX_RETRY}" \
  "${EXTRA_ARGS[@]}"
