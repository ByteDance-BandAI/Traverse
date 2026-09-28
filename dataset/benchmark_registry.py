# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dataset.bc_dataset import BrowseCompDataset


@dataclass(frozen=True)
class BenchmarkSpec:
    """Runtime configuration for one supported evaluation benchmark."""

    name: str
    aliases: tuple[str, ...]
    verifier_name: str
    judge_prompt_set: str
    expected_num_rows: int | None = None
    default_dataset_path: str | None = None
    recommended_judge_model: str | None = None


_REPO_ROOT = Path(__file__).resolve().parents[1]

_BENCHMARK_SPECS = (
    BenchmarkSpec(
        name="browsecomp",
        aliases=(
            "bc",
            "bc185",
            "browsecomp_185",
            "browsecomp-185",
            "browsecomp_en",
            "browsecomp-en",
        ),
        verifier_name="bc_qa_llm_judge",
        judge_prompt_set="bc_judge",
    ),
    BenchmarkSpec(
        name="browsecomp_zh",
        aliases=("bc_zh", "bczh", "bcch", "browsecomp-zh"),
        verifier_name="bc_qa_llm_judge",
        judge_prompt_set="bc_judge",
    ),
    BenchmarkSpec(
        name="gaia_text_103",
        aliases=("gaia", "gaia_text", "gaia-text", "gaia-text-103"),
        verifier_name="gaia_qa_llm_judge",
        judge_prompt_set="gaia_judge",
        expected_num_rows=103,
        default_dataset_path="data/benchmarks/gaia_text_103.parquet",
    ),
    BenchmarkSpec(
        name="xbench_2510",
        aliases=("xbench", "xbench-2510", "xbench_deepsearch_2510"),
        verifier_name="xbench_qa_llm_judge",
        judge_prompt_set="xbench_judge",
        expected_num_rows=100,
        default_dataset_path="data/benchmarks/xbench_2510.parquet",
    ),
    BenchmarkSpec(
        name="deepsearchqa_900",
        aliases=("dsqa", "deepsearchqa", "deepsearch_qa", "deepsearchqa-900"),
        verifier_name="dsqa_qa_llm_judge",
        judge_prompt_set="dsqa_judge",
        expected_num_rows=900,
        default_dataset_path="data/benchmarks/deepsearchqa_900.parquet",
        recommended_judge_model="gemini-2.5-flash",
    ),
    BenchmarkSpec(
        name="widesearch_200",
        aliases=("widesearch", "wide_search", "wide-search"),
        verifier_name="widesearch_table_judge",
        judge_prompt_set="widesearch_judge",
        expected_num_rows=200,
        default_dataset_path="data/benchmarks/widesearch_200.parquet",
        # This is the evaluator model in the benchmark's official release.
        recommended_judge_model="gpt-4.1-2025-04-14",
    ),
    BenchmarkSpec(
        name="widesearch_v2_1_hard_720",
        aliases=(
            "widesearch_v2_1",
            "widesearch-v2-1",
            "widesearch_hard_720",
            "widesearch-hard-720",
        ),
        # These locally selected WideSearch v2.1 questions have set-valued
        # reference answers. Score them component-by-component with the DSQA
        # precision/recall/F1 protocol instead of BrowseComp exact correctness.
        verifier_name="dsqa_qa_llm_judge",
        judge_prompt_set="dsqa_judge",
        expected_num_rows=720,
        default_dataset_path="data/benchmarks/widesearch_v2_1_hard_720.parquet",
        recommended_judge_model="gemini-2.5-flash",
    ),
)

_SPECS_BY_NAME = {spec.name: spec for spec in _BENCHMARK_SPECS}
_ALIASES = {
    alias.lower().replace("-", "_"): spec.name
    for spec in _BENCHMARK_SPECS
    for alias in (spec.name, *spec.aliases)
}


def supported_benchmark_names() -> tuple[str, ...]:
    return tuple(_SPECS_BY_NAME)


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    normalized = (name or "").strip().lower().replace("-", "_")
    canonical_name = _ALIASES.get(normalized)
    if canonical_name is None:
        supported = ", ".join(supported_benchmark_names())
        raise ValueError(
            f"Unsupported benchmark {name!r}. Supported benchmarks: {supported}"
        )
    return _SPECS_BY_NAME[canonical_name]


def resolve_dataset_path(
    spec: BenchmarkSpec,
    dataset_path: str | None,
) -> str:
    if dataset_path:
        return dataset_path
    if spec.default_dataset_path:
        return str(_REPO_ROOT / spec.default_dataset_path)
    raise ValueError(
        f"--dataset_path is required for benchmark {spec.name!r}."
    )


def create_benchmark_dataset(
    benchmark: str,
    dataset_path: str | None,
    *,
    limit: int | None = None,
    shuffle_buffer_size: int = 512,
) -> tuple[BenchmarkSpec, BrowseCompDataset]:
    """Create and validate the dataset selected by ``--benchmark``."""

    spec = get_benchmark_spec(benchmark)
    resolved_path = resolve_dataset_path(spec, dataset_path)
    dataset = BrowseCompDataset(
        resolved_path,
        limit,
        shuffle_buffer_size=shuffle_buffer_size,
        benchmark=spec.name,
    )

    columns = set(dataset.parquet_file.schema_arrow.names)
    required_columns = {"extra_info", "reward_model"}
    missing_columns = sorted(required_columns - columns)
    if missing_columns:
        raise ValueError(
            f"Dataset for benchmark {spec.name!r} is not in the canonical QA "
            f"parquet format; missing columns: {', '.join(missing_columns)}. "
            "Run recipe/fsm_agent/scripts/prepare_benchmark_data.py first."
        )

    if (
        spec.expected_num_rows is not None
        and dataset._physical_total != spec.expected_num_rows
    ):
        raise ValueError(
            f"Benchmark {spec.name!r} requires exactly "
            f"{spec.expected_num_rows} rows, but {resolved_path!r} contains "
            f"{dataset._physical_total}. Re-run the benchmark data preparation "
            "script instead of evaluating an incomplete or different release."
        )

    return spec, dataset
