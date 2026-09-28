# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""WideSearch table evaluator adapted to Traverse's async judge API."""

from __future__ import annotations

import json
import re
from datetime import datetime
from io import StringIO
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from prompt.base import BasePromptBuilder, register_prompt_set
from verifiers import JudgeCompleteStatus, JudgeResult, register_verifier
from verifiers.base_qa_llm_judge import BaseQALLMJudge

try:
    import dateparser as _dateparser
except ImportError:  # The existing sandbox does not require this optional package.
    _dateparser = None

try:
    from dateutil import parser as _dateutil_parser
except ImportError:  # pragma: no cover - pandas normally provides this dependency.
    _dateutil_parser = None


PRIMARY_KEY_PREPROCESS_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""


EVAL_COLUMN_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""


@register_prompt_set("widesearch_judge")
class WideSearchJudgePrompt(BasePromptBuilder):
    """Registry placeholder; WideSearch builds several official judge prompts."""

    def build_system_prompt(self) -> str:
        return ""

    def build_user_prompt(self, **_: Any) -> str:
        return ""


def norm_column(column: str) -> str:
    """Match the normalization used by the official WideSearch evaluator."""

    return str(column).strip().lower().replace(" ", "")


def parse_markdown_json(content: str) -> dict[str, Any] | None:
    """Parse the last fenced JSON object, with a plain-JSON fallback."""

    matches = re.findall(r"```json\s*(\{.*?\})\s*```", content or "", re.DOTALL)
    candidates = list(reversed(matches))
    text = (content or "").strip()
    if text:
        candidates.append(text)
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            candidates.append(text[first_brace:last_brace + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_response_dataframe(response: str) -> tuple[pd.DataFrame | None, str]:
    """Extract a Markdown table using the official benchmark's protocol."""

    if not response or not response.strip():
        return None, "empty response"

    markdown_matches = re.findall(
        r"```markdown(.*?)```",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if not markdown_matches:
        pipe_positions = [match.start() for match in re.finditer(r"\|", response)]
        if len(pipe_positions) >= 4:
            first_pipe = pipe_positions[0]
            last_pipe = pipe_positions[-1]
            start = response.rfind("\n", 0, first_pipe)
            start = 0 if start == -1 else start
            end = response.find("\n", last_pipe)
            end = len(response) if end == -1 else end
            table_candidate = response[start:end]
            markdown_matches = re.findall(
                r"((?:\|.*\n?)+)",
                table_candidate,
            )
    if not markdown_matches:
        return None, "no Markdown table found"

    lines = markdown_matches[0].strip().splitlines()
    if not lines:
        return None, "empty Markdown table"
    lines[0] = lines[0].replace(" ", "").lower()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            continue
        if set(line).issubset(set("|- :")):
            continue
        cleaned_lines.append(
            "|".join(part.strip() for part in line.split("|"))
        )
    if len(cleaned_lines) < 2:
        return None, "Markdown table has no data rows"

    try:
        response_df = pd.read_csv(StringIO("\n".join(cleaned_lines)), sep="|")
    except Exception as exc:
        return None, f"could not parse Markdown table: {exc}"
    response_df = response_df.loc[
        :,
        ~response_df.columns.astype(str).str.startswith("Unnamed"),
    ]
    if response_df.empty:
        return None, "Markdown table has no data rows"
    return response_df, ""


def extract_number(content: Any) -> str:
    numbers = re.findall(
        r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?",
        str(content).replace(",", ""),
    )
    return numbers[0] if numbers else "NULL"


def norm_str(content: Any) -> str:
    return str(content).lower().strip().replace(" ", "").replace("*", "")


def _parse_date(content: Any) -> datetime | None:
    text = str(content).strip()
    if not text:
        return None
    if _dateparser is not None:
        try:
            return _dateparser.parse(
                text,
                settings={"PREFER_DAY_OF_MONTH": "first"},
            )
        except Exception:
            return None

    # A dependency-free-enough fallback for the current Python 3.10 sandbox.
    chinese_match = re.fullmatch(
        r"\s*(\d{4})年(\d{1,2})月(?:(\d{1,2})日?)?\s*",
        text,
    )
    if chinese_match:
        year, month, day = chinese_match.groups()
        try:
            return datetime(int(year), int(month), int(day or 1))
        except ValueError:
            return None
    if _dateutil_parser is not None:
        try:
            return _dateutil_parser.parse(
                text,
                fuzzy=True,
                default=datetime(2000, 1, 1),
            )
        except Exception:
            return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def norm_date(content: Any) -> str:
    parsed = _parse_date(content)
    return parsed.strftime("%Y-%m-%d") if parsed is not None else str(content)


def apply_preprocess(content: Any, preprocess_names: list[str]) -> Any:
    result = content
    for preprocess_name in preprocess_names:
        if preprocess_name == "norm_str":
            result = norm_str(result)
        elif preprocess_name == "extract_number":
            result = extract_number(result)
        elif preprocess_name == "norm_date":
            result = norm_date(result)
        else:
            raise ValueError(
                f"Unsupported WideSearch preprocess function: {preprocess_name}"
            )
    return result


def exact_match(response: str, target: str) -> float:
    return float(str(response).lower() == str(target).lower())


_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|"
    r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


def url_match(response: str, target: str) -> float:
    response_urls = [
        urlparse(url).netloc for url in _URL_PATTERN.findall(str(response))
    ]
    target_urls = [
        urlparse(url).netloc for url in _URL_PATTERN.findall(str(target))
    ]
    return float(set(response_urls) == set(target_urls))


def number_near(response: str, target: str, criterion: float) -> float:
    def to_number(value: str) -> float | None:
        try:
            if "%" in value:
                return float(value.replace("%", "")) / 100.0
            return float(value)
        except (TypeError, ValueError):
            return None

    response_num = to_number(str(response))
    target_num = to_number(str(target))
    if response_num is None or target_num is None:
        return float(
            response_num is None
            and target_num is None
            and str(response) == str(target)
        )
    return float(
        abs(response_num - target_num) <= abs(target_num) * float(criterion)
    )


def date_near(response: str, target: str) -> float:
    response_date = _parse_date(response)
    target_date = _parse_date(target)
    if response_date is None or target_date is None:
        # Preserve the official evaluator's treatment of two unparseable dates.
        return float(response_date is None and target_date is None)
    return float(abs((response_date - target_date).days) <= 31)


@register_verifier("widesearch_table_judge")
class WideSearchTableJudge(BaseQALLMJudge):
    """Official-style Success Rate, Row-F1, and Item-F1 evaluator."""

    async def _call_evaluator(self, prompt: str) -> str | None:
        for _ in range(self.max_retry_times):
            try:
                result = await self.client_config.llm_client.call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.client_config.max_tokens,
                    temperature=self.client_config.temperature,
                    include_thoughts=self.client_config.include_thoughts,
                    top_p=self.client_config.top_p,
                )
            except Exception:
                continue
            content = (result.content or "").strip()
            if content:
                return content
        return None

    async def _semantic_map(
        self,
        response_values: list[str],
        reference_values: list[str],
    ) -> tuple[dict[str, str], bool]:
        content = await self._call_evaluator(
            PRIMARY_KEY_PREPROCESS_PROMPT.format(
                response=response_values,
                reference=reference_values,
            )
        )
        if content is None:
            return {}, True
        mapping = parse_markdown_json(content)
        if mapping is None:
            return {}, True

        response_set = set(response_values)
        reference_set = set(reference_values)
        filtered = {
            str(origin): str(transform)
            for origin, transform in mapping.items()
            if str(origin) in response_set and str(transform) in reference_set
        }
        return filtered, False

    async def _llm_judge_column(
        self,
        responses: list[str],
        targets: list[str],
        criterion: Any,
    ) -> tuple[list[float], bool]:
        response_dict = {
            f"idx_{index}": {"response": response, "target": target}
            for index, (response, target) in enumerate(zip(responses, targets))
        }
        content = await self._call_evaluator(
            EVAL_COLUMN_PROMPT.format(
                criterion=criterion,
                response=response_dict,
            )
        )
        if content is None:
            return [0.0] * len(responses), True
        scores = parse_markdown_json(content)
        if scores is None:
            return [0.0] * len(responses), True

        parsed_scores = []
        for index in range(len(responses)):
            value = scores.get(f"idx_{index}", 0)
            if isinstance(value, bool):
                value = int(value)
            if value not in (0, 1):
                value = 0
            parsed_scores.append(float(value))
        return parsed_scores, False

    def _failure_result(self, message: str) -> JudgeResult:
        judge_model = str(
            getattr(self.client_config.llm_client, "model_name", "")
        )
        return JudgeResult(
            is_correct=False,
            score=0.0,
            reasoning=message,
            complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
            metrics={
                "success_rate": 0.0,
                "row_precision": 0.0,
                "row_recall": 0.0,
                "row_f1": 0.0,
                "item_precision": 0.0,
                "item_recall": 0.0,
                "item_f1": 0.0,
                "judge_model": judge_model,
                "evaluation_error": message,
            },
        )
    async def async_judge(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeResult:
        del question, ground_truth
        if not isinstance(metadata, dict):
            return self._failure_result(
                "WideSearch judging requires private row metadata"
            )
        evaluation_json = metadata.get("evaluation_json")
        gold_csv = metadata.get("gold_csv")
        if not isinstance(evaluation_json, str) or not isinstance(gold_csv, str):
            return self._failure_result(
                "WideSearch row metadata is missing evaluation_json or gold_csv"
            )
        try:
            evaluation = json.loads(evaluation_json)
            required_columns = [norm_column(c) for c in evaluation["required"]]
            unique_columns = [
                norm_column(c) for c in evaluation["unique_columns"]
            ]
            raw_pipeline = evaluation["eval_pipeline"]
            pipeline = {
                norm_column(column): config
                for column, config in raw_pipeline.items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return self._failure_result(
                f"Invalid WideSearch evaluation metadata: {exc}"
            )

        response_df, parse_error = extract_response_dataframe(answer)
        if response_df is None:
            return self._failure_result(parse_error)
        try:
            answer_df = pd.read_csv(StringIO(gold_csv))
        except Exception as exc:
            return self._failure_result(
                f"Could not parse WideSearch gold table: {exc}"
            )

        answer_df.columns = [norm_column(column) for column in answer_df.columns]
        response_df.columns = [
            norm_column(column) for column in response_df.columns
        ]
        llm_failures = 0

        if set(required_columns) != set(response_df.columns):
            column_map, failed = await self._semantic_map(
                response_df.columns.tolist(),
                required_columns,
            )
            llm_failures += int(failed)
            response_df.rename(columns=column_map, inplace=True)
        if (
            set(required_columns) != set(response_df.columns)
            or response_df.columns.duplicated().any()
        ):
            return self._failure_result(
                "Predicted table columns do not match the required columns"
            )
        if not set(required_columns).issubset(answer_df.columns):
            return self._failure_result(
                "Gold table columns do not match the required columns"
            )

        answer_df = answer_df[required_columns].copy()
        response_df = response_df[required_columns].copy()
        for column in required_columns:
            answer_df[column] = answer_df[column].astype(str)
            response_df[column] = response_df[column].astype(str)
        answer_df.drop_duplicates(subset=unique_columns, inplace=True)
        response_df.drop_duplicates(subset=unique_columns, inplace=True)

        for column in unique_columns:
            item = pipeline.get(column)
            if item is None:
                continue
            metric_names = item.get("metric", [])
            if "llm_judge" in metric_names or "exact_match" in metric_names:
                primary_key_map, failed = await self._semantic_map(
                    response_df[column].tolist(),
                    answer_df[column].tolist(),
                )
                llm_failures += int(failed)
                response_df[column] = response_df[column].apply(
                    lambda value: primary_key_map.get(value, value)
                )

        try:
            for column, item in pipeline.items():
                preprocess_names = item.get("preprocess", [])
                for preprocess_name in preprocess_names:
                    response_df[column] = response_df[column].apply(
                        lambda value, name=preprocess_name: apply_preprocess(
                            value,
                            [name],
                        )
                    )
                    answer_df[column] = answer_df[column].apply(
                        lambda value, name=preprocess_name: apply_preprocess(
                            value,
                            [name],
                        )
                    )
        except (KeyError, TypeError, ValueError) as exc:
            return self._failure_result(
                f"WideSearch preprocessing failed: {exc}"
            )

        inner_df = pd.merge(
            answer_df,
            response_df,
            on=unique_columns,
            how="inner",
            suffixes=("_query", "_response"),
        )
        score_columns: dict[str, list[float]] = {}
        for column in required_columns:
            if column in unique_columns:
                score_columns[f"{column}_exact_match"] = [1.0] * len(inner_df)
                continue

            item = pipeline[column]
            metric_names = item.get("metric", [])
            criterion = item.get("criterion")
            for metric_name in metric_names:
                responses = inner_df[f"{column}_response"].tolist()
                targets = inner_df[f"{column}_query"].tolist()
                if metric_name == "llm_judge":
                    scores, failed = await self._llm_judge_column(
                        responses,
                        targets,
                        criterion,
                    )
                    llm_failures += int(failed)
                elif metric_name == "exact_match":
                    scores = [
                        exact_match(response, target)
                        for response, target in zip(responses, targets)
                    ]
                elif metric_name == "number_near":
                    scores = [
                        number_near(response, target, criterion)
                        for response, target in zip(responses, targets)
                    ]
                elif metric_name == "date_near":
                    scores = [
                        date_near(response, target)
                        for response, target in zip(responses, targets)
                    ]
                elif metric_name == "url_match":
                    scores = [
                        url_match(response, target)
                        for response, target in zip(responses, targets)
                    ]
                else:
                    return self._failure_result(
                        f"Unsupported WideSearch metric: {metric_name}"
                    )
                score_columns[f"{column}_{metric_name}"] = scores

        inner_scores = pd.DataFrame(score_columns, index=inner_df.index)
        if inner_scores.empty:
            true_positive_rows = 0.0
            true_positive_items = 0.0
        else:
            true_positive_rows = float(inner_scores.min(axis=1).sum())
            true_positive_items = float(inner_scores.sum().sum())

        predicted_rows = len(response_df)
        gold_rows = len(answer_df)
        predicted_items = predicted_rows * len(required_columns)
        gold_items = gold_rows * len(required_columns)
        row_precision = (
            true_positive_rows / predicted_rows if predicted_rows else 0.0
        )
        row_recall = true_positive_rows / gold_rows if gold_rows else 0.0
        item_precision = (
            true_positive_items / predicted_items if predicted_items else 0.0
        )
        item_recall = true_positive_items / gold_items if gold_items else 0.0

        def f1(precision: float, recall: float) -> float:
            return (
                2 * precision * recall / (precision + recall)
                if precision + recall > 1e-9
                else 0.0
            )

        row_f1 = f1(row_precision, row_recall)
        item_f1 = f1(item_precision, item_recall)
        success = (
            row_precision
            == row_recall
            == row_f1
            == item_precision
            == item_recall
            == item_f1
            == 1.0
        )
        judge_model = str(
            getattr(self.client_config.llm_client, "model_name", "")
        )
        instance_id = str(metadata.get("instance_id", ""))
        return JudgeResult(
            is_correct=success,
            score=1.0 if success else 0.0,
            reasoning=(
                f"{instance_id or 'WideSearch'}: "
                f"Row-F1={row_f1:.6f}, Item-F1={item_f1:.6f}"
            ),
            complete_status=JudgeCompleteStatus.SUCCESS,
            metrics={
                "success_rate": 1.0 if success else 0.0,
                "row_precision": row_precision,
                "row_recall": row_recall,
                "row_f1": row_f1,
                "item_precision": item_precision,
                "item_recall": item_recall,
                "item_f1": item_f1,
                "row_true_positives": true_positive_rows,
                "item_true_positives": true_positive_items,
                "predicted_rows": predicted_rows,
                "gold_rows": gold_rows,
                "required_columns": len(required_columns),
                "llm_judge_failures": llm_failures,
                "judge_model": judge_model,
                "configured_judge_model": "gpt-4.1-2025-04-14",
                "matches_configured_judge_model": (
                    "gpt-4.1-2025-04-14" in judge_model.lower()
                ),
            },
        )
