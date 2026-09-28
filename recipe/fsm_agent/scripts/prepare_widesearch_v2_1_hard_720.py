#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Build the canonical 720-row WideSearch v2.1 hard-question parquet.

The diversity selection JSONL intentionally contains no answers. This builder
joins its selected IDs back to the original RL candidate JSONL, validates the
balanced 12-domain/72-subdomain bilingual sample, and stores ground truth only
in the private dataset fields consumed by the judge.

Example:
  python recipe/fsm_agent/scripts/prepare_widesearch_v2_1_hard_720.py \
    --source-jsonl /path/to/widesearch_v2_1_rl_qa_candidates.jsonl \
    --selection-jsonl /path/to/widesearch_v2_1_hard_diverse_720.jsonl \
    --output data/benchmarks/widesearch_v2_1_hard_720.parquet
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


BENCHMARK = "widesearch_v2_1_hard_720"
EXPECTED_ROWS = 720
EXPECTED_DOMAINS = 12
EXPECTED_SUBDOMAINS = 72
EXPECTED_LANGUAGES = {"en", "zh"}

SCHEMA = pa.schema(
    [
        ("data_source", pa.string()),
        (
            "prompt",
            pa.list_(
                pa.struct([("content", pa.string()), ("role", pa.string())])
            ),
        ),
        (
            "reward_model",
            pa.struct(
                [("ground_truth", pa.list_(pa.string())), ("style", pa.string())]
            ),
        ),
        (
            "extra_info",
            pa.struct(
                [
                    ("answer", pa.list_(pa.string())),
                    ("answer_type", pa.string()),
                    ("benchmark", pa.string()),
                    ("difficulty", pa.string()),
                    ("domain", pa.string()),
                    ("id", pa.string()),
                    ("index", pa.int64()),
                    (
                        "judge_metadata",
                        pa.struct(
                            [
                                ("answer_type", pa.string()),
                                ("problem_category", pa.string()),
                            ]
                        ),
                    ),
                    ("language", pa.string()),
                    ("problem_category", pa.string()),
                    ("question", pa.string()),
                    ("source_query_id", pa.string()),
                    ("split", pa.string()),
                    ("sub_domain", pa.string()),
                ]
            ),
        ),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_language(question: str) -> str:
    return "zh" if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", question) else "en"


def read_selection(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            language = str(row.get("language", "")).strip()
            if not question_id or not question:
                raise ValueError(
                    f"Selection row {line_number} is missing id or question"
                )
            if question_id in seen_ids:
                raise ValueError(f"Duplicate selected id: {question_id}")
            normalized_question = re.sub(r"\s+", " ", question).casefold()
            if normalized_question in seen_questions:
                raise ValueError(
                    f"Duplicate selected question at row {line_number}: {question_id}"
                )
            if row.get("difficulty") != "hard" or row.get("split") != "rl_train":
                raise ValueError(
                    f"Selected row {question_id} is not hard/rl_train"
                )
            detected_language = detect_language(question)
            if language != detected_language:
                raise ValueError(
                    f"Selected row {question_id} has language={language!r}; "
                    f"detected {detected_language!r}"
                )
            rows.append(row)
            seen_ids.add(question_id)
            seen_questions.add(normalized_question)

    if len(rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} selected rows, found {len(rows)} in {path}"
        )
    return rows


def validate_balanced_720(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = collections.Counter(row["domain"] for row in rows)
    by_subdomain = collections.Counter(
        (row["domain"], row["sub_domain"]) for row in rows
    )
    by_stratum = collections.Counter(
        (row["domain"], row["sub_domain"], row["language"]) for row in rows
    )
    by_language = collections.Counter(row["language"] for row in rows)

    if len(by_domain) != EXPECTED_DOMAINS or set(by_domain.values()) != {60}:
        raise ValueError(
            "Expected 12 domains with 60 questions each; "
            f"found {dict(sorted(by_domain.items()))}"
        )
    if len(by_subdomain) != EXPECTED_SUBDOMAINS or set(by_subdomain.values()) != {10}:
        raise ValueError(
            "Expected 72 domain/subdomain groups with 10 questions each"
        )
    if len(by_stratum) != 144 or set(by_stratum.values()) != {5}:
        raise ValueError(
            "Expected 144 domain/subdomain/language strata with 5 questions each"
        )
    if set(by_language) != EXPECTED_LANGUAGES or set(by_language.values()) != {360}:
        raise ValueError(
            f"Expected 360 English and 360 Chinese questions; found {by_language}"
        )
    return {
        "domains": len(by_domain),
        "subdomains": len(by_subdomain),
        "strata": len(by_stratum),
        "languages": dict(sorted(by_language.items())),
    }


def load_selected_sources(
    source_path: Path,
    selection_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selected_ids = {str(row["id"]) for row in selection_rows}
    matched: dict[str, dict[str, Any]] = {}
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("id", "")).strip()
            if question_id not in selected_ids:
                continue
            if question_id in matched:
                raise ValueError(
                    f"Duplicate source id {question_id} at line {line_number}"
                )
            matched[question_id] = row

    missing = sorted(selected_ids - set(matched))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{len(missing)} selected IDs are absent from the source JSONL: {preview}"
        )
    return matched


def validate_and_normalize_ground_truth(
    selected: dict[str, Any],
    source: dict[str, Any],
) -> list[str]:
    question_id = str(selected["id"])
    for field in ("question", "domain", "sub_domain", "difficulty", "split"):
        if selected.get(field) != source.get(field):
            raise ValueError(
                f"Selection/source mismatch for {question_id}: field {field}"
            )
    ground_truth = source.get("ground_truth")
    if (
        not isinstance(ground_truth, list)
        or not ground_truth
        or any(not isinstance(item, str) or not item.strip() for item in ground_truth)
    ):
        raise ValueError(f"Invalid ground_truth for {question_id}")
    declared_size = source.get("ground_truth_size")
    if declared_size != len(ground_truth):
        raise ValueError(
            f"ground_truth_size mismatch for {question_id}: "
            f"declared {declared_size}, actual {len(ground_truth)}"
        )
    return [item.strip() for item in ground_truth]


def build_records(
    selection_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, selected in enumerate(selection_rows):
        source = source_rows[str(selected["id"])]
        ground_truth = validate_and_normalize_ground_truth(selected, source)
        question = str(selected["question"]).strip()
        records.append(
            {
                "data_source": BENCHMARK,
                "prompt": [{"content": question, "role": "user"}],
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": "rule",
                },
                "extra_info": {
                    "answer": ground_truth,
                    "answer_type": "Set Answer",
                    "benchmark": BENCHMARK,
                    "difficulty": "hard",
                    "domain": str(selected["domain"]),
                    "id": str(selected["id"]),
                    "index": index,
                    "judge_metadata": {
                        "answer_type": "Set Answer",
                        "problem_category": str(selected["domain"]),
                    },
                    "language": str(selected["language"]),
                    "problem_category": str(selected["domain"]),
                    "question": question,
                    "source_query_id": str(selected["id"]),
                    "split": "rl_train",
                    "sub_domain": str(selected["sub_domain"]),
                },
            }
        )
    return records


def write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".parquet",
        dir=output_path.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pq.write_table(table, temporary_path)
        validation = pq.ParquetFile(temporary_path)
        if validation.metadata.num_rows != len(records):
            raise ValueError(
                f"Parquet row count mismatch: expected {len(records)}, "
                f"found {validation.metadata.num_rows}"
            )
        os.replace(temporary_path, output_path)
        output_path.chmod(0o644)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_dataset(
    source_path: Path,
    selection_path: Path,
    output_path: Path,
    *,
    expected_rows: int = EXPECTED_ROWS,
    validate_distribution: bool = True,
) -> dict[str, Any]:
    selection_rows = read_selection(selection_path, expected_rows)
    distribution = (
        validate_balanced_720(selection_rows) if validate_distribution else {}
    )
    source_rows = load_selected_sources(source_path, selection_rows)
    records = build_records(selection_rows, source_rows)
    write_parquet(records, output_path)

    manifest = {
        "benchmark": BENCHMARK,
        "rows": len(records),
        "source_jsonl": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "selection_jsonl": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "output_sha256": sha256_file(output_path),
        "answer_type": "Set Answer",
        "metric_protocol": "DSQA component precision/recall/F1",
        "distribution": distribution,
    }
    manifest_path = output_path.with_suffix(".source.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical WideSearch v2.1 hard-720 parquet."
    )
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--selection-jsonl", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/widesearch_v2_1_hard_720.parquet"),
    )
    args = parser.parse_args()
    manifest = build_dataset(
        args.source_jsonl,
        args.selection_jsonl,
        args.output,
    )
    print(f"Wrote {manifest['rows']} rows -> {args.output}")
    print(f"Provenance manifest -> {args.output.with_suffix('.source.json')}")


if __name__ == "__main__":
    main()
