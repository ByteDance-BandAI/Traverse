#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Prepare canonical search-benchmark parquets.

The downloaded source artifacts are pinned by immutable revisions and SHA-256.
Generated parquets contain plaintext benchmark answers and are intentionally
gitignored; do not publish them or expose them to the agent's web tools.
"""

import argparse
import base64
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "benchmarks"

GAIA_SOURCE_REVISION = "d4b5ad6f61f0f41621db1e5d84f80964a49d7634"
GAIA_SOURCE_URL = (
    "https://huggingface.co/datasets/OpenResearcher/web-bench/resolve/"
    f"{GAIA_SOURCE_REVISION}/data/gaia_text-00000-of-00001.parquet"
)
GAIA_SOURCE_SHA256 = (
    "58d459e111de90f151329c625bbe28d927cea378f32a92c391e96046a1b530fd"
)

XBENCH_SOURCE_REVISION = "17c562192cc7e62215bfb98b65e9f8806fb95504"
XBENCH_SOURCE_URL = (
    "https://raw.githubusercontent.com/xbench-ai/xbench-evals/"
    f"{XBENCH_SOURCE_REVISION}/data/DeepSearch-2510.csv"
)
XBENCH_SOURCE_SHA256 = (
    "a9378e56b05ec8f007b8ecc8f6ac74900abafd558267acd5839d0d05fbc6977a"
)

DSQA_SOURCE_REVISION = "b2623f8653065c2672de6d941fc5434cd652376c"
DSQA_SOURCE_URL = (
    "https://huggingface.co/datasets/google/deepsearchqa/resolve/"
    f"{DSQA_SOURCE_REVISION}/DSQA-full.csv"
)
DSQA_SOURCE_SHA256 = (
    "25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396"
)

WIDESEARCH_SOURCE_REVISION = "6531a7e5b497d44c8912407e0cb3dc95bd98cc09"
WIDESEARCH_SOURCE_REPO = "ByteDance-Seed/WideSearch"
# SHA-256 over each sorted relative path, NUL, its bytes, and a trailing NUL.
WIDESEARCH_SOURCE_SHA256 = (
    "72e49ba83511714221f82e9efe8e3cdd86ce03a6d588e64485b51b08f7fe21ee"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, expected_sha256, target):
    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [
                curl,
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--retry",
                "3",
                "--connect-timeout",
                "15",
                "--max-time",
                "180",
                "--output",
                str(target),
                url,
            ],
            check=True,
        )
    else:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Traverse benchmark preparer"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            with Path(target).open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

    actual_sha256 = sha256_file(target)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source checksum mismatch for {url}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )


def widesearch_source_sha256(source_dir):
    source_dir = Path(source_dir)
    paths = [source_dir / "widesearch.jsonl"]
    paths.extend(sorted((source_dir / "widesearch_gold").glob("*.csv")))
    if not paths[0].is_file():
        raise FileNotFoundError(
            f"WideSearch source metadata not found: {paths[0]}"
        )
    if len(paths) != 201:
        raise ValueError(
            f"WideSearch source must contain 200 gold CSVs; found {len(paths) - 1}"
        )

    digest = hashlib.sha256()
    for path in paths:
        relative_path = path.relative_to(source_dir).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_widesearch_source(source_dir):
    actual_sha256 = widesearch_source_sha256(source_dir)
    if actual_sha256 != WIDESEARCH_SOURCE_SHA256:
        raise ValueError(
            "WideSearch source checksum mismatch: expected "
            f"{WIDESEARCH_SOURCE_SHA256}, got {actual_sha256}"
        )


def download_widesearch_source():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Preparing WideSearch requires huggingface_hub. You can instead "
            "pass --source-dir pointing at the pinned official snapshot."
        ) from exc

    snapshot_path = snapshot_download(
        repo_id=WIDESEARCH_SOURCE_REPO,
        repo_type="dataset",
        revision=WIDESEARCH_SOURCE_REVISION,
        allow_patterns=("widesearch.jsonl", "widesearch_gold/*.csv"),
    )
    validate_widesearch_source(snapshot_path)
    return Path(snapshot_path)


def resolve_widesearch_source(source_dir=None):
    if source_dir is not None:
        source_path = Path(source_dir)
        validate_widesearch_source(source_path)
        return source_path

    env_source_dir = os.getenv("WIDESEARCH_SOURCE_DIR", "").strip()
    candidates = []
    if env_source_dir:
        candidates.append(Path(env_source_dir))
    # Reuse the already-downloaded official snapshot from the sibling project
    # when this workspace has it. The composite checksum below guarantees that
    # this is exactly the pinned source, not an arbitrary local copy.
    candidates.append(
        REPO_ROOT.parent / "SimpleInference" / "data" / "widesearch_hf"
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            validate_widesearch_source(candidate)
        except (FileNotFoundError, ValueError):
            if env_source_dir and candidate == Path(env_source_dir):
                raise
            continue
        print(f"Using verified local WideSearch source: {candidate}")
        return candidate

    print("Downloading pinned widesearch_200 source snapshot...")
    return download_widesearch_source()


def canonical_record(benchmark, question_id, index, question, answer, **metadata):
    extra_info = {
        "id": question_id,
        "index": index,
        "question": question,
        "answer": [answer],
        "benchmark": benchmark,
    }
    extra_info.update(metadata)
    return {
        "data_source": benchmark,
        "prompt": [{"content": question, "role": "user"}],
        "reward_model": {
            "ground_truth": [answer],
            "style": "rule",
        },
        "extra_info": extra_info,
    }


def write_canonical_parquet(records, output_path, expected_rows):
    if len(records) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} records, received {len(records)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".parquet",
        dir=str(output_path.parent),
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        pd.DataFrame(records).to_parquet(temporary_path, index=False)
        validate_canonical_parquet(temporary_path, expected_rows)
        os.replace(temporary_path, output_path)
        output_path.chmod(0o644)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_canonical_parquet(path, expected_rows):
    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata.num_rows != expected_rows:
        raise ValueError(
            f"{path} contains {parquet_file.metadata.num_rows} rows; "
            f"expected {expected_rows}"
        )
    required = {"data_source", "prompt", "reward_model", "extra_info"}
    missing = sorted(required - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(
            f"{path} is missing canonical columns: {', '.join(missing)}"
        )


def xor_decrypt(encoded_value, key):
    encrypted = base64.b64decode(encoded_value)
    key_bytes = key.encode("utf-8")
    return bytes(
        byte ^ key_bytes[index % len(key_bytes)]
        for index, byte in enumerate(encrypted)
    ).decode("utf-8")


def prepare_gaia(source_path, output_path):
    rows = pq.read_table(source_path).to_pylist()
    records = []
    for index, row in enumerate(rows):
        source_id = int(row["query_id"])
        records.append(
            canonical_record(
                benchmark="gaia_text_103",
                question_id=f"gaia_text_{source_id:03d}",
                index=index,
                question=str(row["question"]),
                answer=str(row["answer"]),
                source_query_id=source_id,
                split="validation",
            )
        )
    write_canonical_parquet(records, output_path, expected_rows=103)


def prepare_xbench(source_path, output_path):
    records = []
    with Path(source_path).open(encoding="utf-8-sig", newline="") as source:
        for index, row in enumerate(csv.DictReader(source)):
            key = row["canary"]
            source_id = str(row["id"])
            question = xor_decrypt(row["prompt"], key)
            answer = xor_decrypt(row["answer"], key)
            reference_steps = (
                xor_decrypt(row["reference_steps"], key)
                if row.get("reference_steps")
                else ""
            )
            records.append(
                canonical_record(
                    benchmark="xbench_2510",
                    question_id=f"deepsearch_{source_id}",
                    index=index,
                    question=question,
                    answer=answer,
                    source_query_id=source_id,
                    reference_steps=reference_steps,
                )
            )
    write_canonical_parquet(records, output_path, expected_rows=100)


def prepare_dsqa(source_path, output_path):
    records = []
    required_columns = {
        "problem",
        "problem_category",
        "answer",
        "answer_type",
    }
    with Path(source_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "DeepSearchQA source is missing columns: "
                + ", ".join(sorted(missing_columns))
            )

        for index, row in enumerate(reader):
            question = str(row["problem"]).strip()
            answer = str(row["answer"]).strip()
            answer_type = str(row["answer_type"]).strip()
            problem_category = str(row["problem_category"]).strip()
            if not question or not answer:
                raise ValueError(
                    f"DeepSearchQA source row {index} has an empty problem or answer"
                )
            if answer_type not in {"Single Answer", "Set Answer"}:
                raise ValueError(
                    f"DeepSearchQA source row {index} has unsupported "
                    f"answer_type={answer_type!r}"
                )

            records.append(
                canonical_record(
                    benchmark="deepsearchqa_900",
                    question_id=f"dsqa_{index:03d}",
                    index=index,
                    question=question,
                    answer=answer,
                    source_query_id=index,
                    problem_category=problem_category,
                    answer_type=answer_type,
                    judge_metadata={
                        "answer_type": answer_type,
                        "problem_category": problem_category,
                    },
                    split="eval",
                )
            )
    write_canonical_parquet(records, output_path, expected_rows=900)


def prepare_widesearch(source_dir, output_path):
    source_dir = Path(source_dir)
    validate_widesearch_source(source_dir)
    metadata_path = source_dir / "widesearch.jsonl"
    records = []
    seen_instance_ids = set()
    language_counts = {"en": 0, "zh": 0}

    with metadata_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id", "")).strip()
            question = str(row.get("query", "")).strip()
            language = str(row.get("language", "")).strip()
            evaluation = row.get("evaluation")
            if isinstance(evaluation, str):
                evaluation = json.loads(evaluation)
            if not instance_id or not question or not isinstance(evaluation, dict):
                raise ValueError(
                    f"Malformed WideSearch source row at line {index + 1}"
                )
            if instance_id in seen_instance_ids:
                raise ValueError(
                    f"Duplicate WideSearch instance_id: {instance_id}"
                )
            if language not in language_counts:
                raise ValueError(
                    f"Unsupported WideSearch language for {instance_id}: {language!r}"
                )
            seen_instance_ids.add(instance_id)
            language_counts[language] += 1

            required_columns = evaluation.get("required")
            unique_columns = evaluation.get("unique_columns")
            eval_pipeline = evaluation.get("eval_pipeline")
            if (
                not isinstance(required_columns, list)
                or not required_columns
                or not isinstance(unique_columns, list)
                or not unique_columns
                or not isinstance(eval_pipeline, dict)
            ):
                raise ValueError(
                    f"Invalid WideSearch evaluation config for {instance_id}"
                )
            if not set(unique_columns).issubset(required_columns):
                raise ValueError(
                    f"WideSearch unique columns are not required for {instance_id}"
                )
            if set(eval_pipeline) != set(required_columns):
                raise ValueError(
                    f"WideSearch eval pipeline does not match required columns "
                    f"for {instance_id}"
                )

            for column, pipeline in eval_pipeline.items():
                metrics = pipeline.get("metric", [])
                if len(metrics) != 1:
                    raise ValueError(
                        f"WideSearch {instance_id}/{column} must define exactly "
                        f"one metric, got {metrics!r}"
                    )

            gold_path = source_dir / "widesearch_gold" / f"{instance_id}.csv"
            gold_csv = gold_path.read_text(encoding="utf-8-sig")
            gold_df = pd.read_csv(gold_path, encoding="utf-8-sig")
            normalized_gold_columns = {
                str(column).strip().lower().replace(" ", "")
                for column in gold_df.columns
            }
            missing_columns = set(required_columns) - normalized_gold_columns
            if missing_columns:
                raise ValueError(
                    f"WideSearch gold table {instance_id} is missing columns: "
                    + ", ".join(sorted(missing_columns))
                )

            answer_summary = (
                f"Reference table with {len(gold_df)} rows and "
                f"{len(required_columns)} required columns"
            )
            records.append(
                canonical_record(
                    benchmark="widesearch_200",
                    question_id=instance_id,
                    index=index,
                    question=question,
                    answer=answer_summary,
                    source_query_id=instance_id,
                    language=language,
                    judge_metadata={
                        "instance_id": instance_id,
                        "language": language,
                        "evaluation_json": json.dumps(
                            evaluation,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        # Private judge-only reference. BrowseCompDataset does
                        # not add this metadata to the agent's question prompt.
                        "gold_csv": gold_csv,
                    },
                    split="full",
                )
            )

    if language_counts != {"en": 100, "zh": 100}:
        raise ValueError(
            "WideSearch must contain 100 English and 100 Chinese tasks; "
            f"found {language_counts}"
        )
    write_canonical_parquet(records, output_path, expected_rows=200)


def prepare_one(benchmark, output_dir, force, source_dir=None):
    configs = {
        "gaia_text_103": {
            "url": GAIA_SOURCE_URL,
            "sha256": GAIA_SOURCE_SHA256,
            "source_name": "gaia_text_103.source.parquet",
            "output_name": "gaia_text_103.parquet",
            "expected_rows": 103,
            "prepare": prepare_gaia,
        },
        "xbench_2510": {
            "url": XBENCH_SOURCE_URL,
            "sha256": XBENCH_SOURCE_SHA256,
            "source_name": "xbench_2510.encrypted.csv",
            "output_name": "xbench_2510.parquet",
            "expected_rows": 100,
            "prepare": prepare_xbench,
        },
        "deepsearchqa_900": {
            "url": DSQA_SOURCE_URL,
            "sha256": DSQA_SOURCE_SHA256,
            "source_name": "deepsearchqa_900.source.csv",
            "output_name": "deepsearchqa_900.parquet",
            "expected_rows": 900,
            "prepare": prepare_dsqa,
        },
        "widesearch_200": {
            "output_name": "widesearch_200.parquet",
            "expected_rows": 200,
            "prepare": prepare_widesearch,
        },
    }
    config = configs[benchmark]
    output_dir = Path(output_dir)
    output_path = output_dir / config["output_name"]

    if output_path.exists() and not force:
        validate_canonical_parquet(output_path, config["expected_rows"])
        print(f"Verified existing {benchmark}: {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    if benchmark == "widesearch_200":
        source_path = resolve_widesearch_source(source_dir)
        config["prepare"](source_path, output_path)
        source_url = (
            f"https://huggingface.co/datasets/{WIDESEARCH_SOURCE_REPO}/tree/"
            f"{WIDESEARCH_SOURCE_REVISION}"
        )
        source_sha256 = WIDESEARCH_SOURCE_SHA256
    else:
        with tempfile.TemporaryDirectory(prefix=f"prepare-{benchmark}-") as temp_dir:
            source_path = Path(temp_dir) / config["source_name"]
            print(f"Downloading pinned {benchmark} source...")
            download_verified(config["url"], config["sha256"], source_path)
            config["prepare"](source_path, output_path)
        source_url = config["url"]
        source_sha256 = config["sha256"]

    manifest_path = output_path.with_suffix(".source.json")
    manifest = {
        "benchmark": benchmark,
        "rows": config["expected_rows"],
        "source_url": source_url,
        "source_sha256": source_sha256,
        "output_sha256": sha256_file(output_path),
    }
    if benchmark == "widesearch_200":
        manifest["source_revision"] = WIDESEARCH_SOURCE_REVISION
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {benchmark}: {output_path}")
    print(f"Provenance manifest: {manifest_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Traverse evaluation benchmark parquets."
    )
    parser.add_argument(
        "--benchmark",
        choices=(
            "all",
            "gaia_text_103",
            "gaia",
            "xbench_2510",
            "xbench",
            "deepsearchqa_900",
            "deepsearchqa",
            "dsqa",
            "widesearch_200",
            "widesearch",
            "wide_search",
        ),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help=(
            "Optional local pinned source snapshot. Currently used only for "
            "WideSearch; otherwise the official source is downloaded."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and atomically replace existing generated files.",
    )
    args = parser.parse_args()

    aliases = {
        "gaia": "gaia_text_103",
        "xbench": "xbench_2510",
        "deepsearchqa": "deepsearchqa_900",
        "dsqa": "deepsearchqa_900",
        "widesearch": "widesearch_200",
        "wide_search": "widesearch_200",
    }
    selected = aliases.get(args.benchmark, args.benchmark)
    # Keep the pre-existing meaning of "all" stable. DSQA is prepared only
    # when explicitly requested so existing automation does not download a
    # new 900-row benchmark unexpectedly.
    benchmarks = (
        ("gaia_text_103", "xbench_2510")
        if selected == "all"
        else (selected,)
    )
    for benchmark in benchmarks:
        if args.source_dir is not None and benchmark != "widesearch_200":
            parser.error("--source-dir is currently supported only for WideSearch")
        prepare_one(
            benchmark,
            args.output_dir,
            args.force,
            source_dir=args.source_dir,
        )


if __name__ == "__main__":
    main()
