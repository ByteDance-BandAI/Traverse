#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Convert a BrowseComp-style raw .jsonl into the parquet schema fsm_agent expects.

The fsm_agent dataset loader (dataset/bc_dataset.py) reads:
  - extra_info["question"]
  - extra_info["id"]          (raw files usually lack this; we synthesize it)
  - reward_model["ground_truth"]

Output schema mirrors browsecomp.parquet exactly:
  data_source: string
  prompt:      list<struct<content: string, role: string>>
  reward_model: struct<ground_truth: list<string>, style: string>
  extra_info:  struct<answer: list<string>, id: string, index: int64,
                      question: string, split: string>

Usage:
  python -m recipe.fsm_agent.scripts.jsonl_to_parquet \
    --input  /path/to/raw.jsonl \
    --output /path/to/out.parquet \
    [--split train] [--id-prefix train]
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

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
                    ("id", pa.string()),
                    ("index", pa.int64()),
                    ("question", pa.string()),
                    ("split", pa.string()),
                ]
            ),
        ),
    ]
)


def _as_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def convert(input_path: Path, output_path: Path, split: str, id_prefix: str) -> int:
    data_source: list[str] = []
    prompt: list[list[dict]] = []
    reward_model: list[dict] = []
    extra_info: list[dict] = []

    with open(input_path, encoding="utf-8") as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ei = rec.get("extra_info", {}) or {}
            rm = rec.get("reward_model", {}) or {}

            question = ei.get("question", "")
            ground_truth = _as_str_list(rm.get("ground_truth"))
            answer = _as_str_list(ei.get("answer")) or ground_truth
            rec_split = ei.get("split") or split

            raw_prompt = rec.get("prompt")
            if not raw_prompt:
                raw_prompt = [{"content": question, "role": "user"}]
            norm_prompt = [
                {"content": str(m.get("content", "")), "role": str(m.get("role", "user"))}
                for m in raw_prompt
            ]

            data_source.append(str(rec.get("data_source", "")))
            prompt.append(norm_prompt)
            reward_model.append(
                {"ground_truth": ground_truth, "style": str(rm.get("style", "rule"))}
            )
            extra_info.append(
                {
                    "answer": answer,
                    "id": f"{id_prefix}_{idx}",
                    "index": idx,
                    "question": str(question),
                    "split": str(rec_split),
                }
            )
            idx += 1

    table = pa.table(
        {
            "data_source": pa.array(data_source, type=SCHEMA.field("data_source").type),
            "prompt": pa.array(prompt, type=SCHEMA.field("prompt").type),
            "reward_model": pa.array(reward_model, type=SCHEMA.field("reward_model").type),
            "extra_info": pa.array(extra_info, type=SCHEMA.field("extra_info").type),
        },
        schema=SCHEMA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--id-prefix",
        default=None,
        help="Prefix for synthesized extra_info.id (default: --split value).",
    )
    args = parser.parse_args()

    id_prefix = args.id_prefix or args.split
    n = convert(Path(args.input), Path(args.output), args.split, id_prefix)
    print(f"Wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
