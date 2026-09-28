# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from llm_clients.base import LLMCallResult
from prompt.base import BasePromptBuilder, register_prompt_set
from verifiers import JudgeCompleteStatus, JudgeResult, register_verifier
from verifiers.base_qa_llm_judge import BaseQALLMJudge


# This prompt follows the evaluation protocol published with
# google/deepsearchqa. In particular, Set Answer items are judged component by
# component and any unsupported extra answer is counted as a false positive.
DSQA_JUDGE_PROMPT = """\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.

**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.

**Example (Partial):**

```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true
    }},
    "Excessive Answers": ["Italy"]
  }}
}}
```

**Now, proceed with the evaluation for the following inputs:**

User Prompt:
<prompt>
{prompt}
</prompt>

Prompt Type: {prompt_type}

Correct Answer:
<answer>
{correct_answer}
</answer>

AI Response:
<response>
{response}
</response>

Rating:
"""


@register_prompt_set("dsqa_judge")
class DSQAJudgePrompt(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        # The official starter sends the rubric and inputs as one user message.
        # This method remains implemented to satisfy the common prompt interface.
        return ""

    def build_user_prompt(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any],
    ) -> str:
        prompt_type = str(metadata.get("answer_type", "")).strip()
        if prompt_type not in {"Single Answer", "Set Answer"}:
            raise ValueError(
                "DeepSearchQA row is missing a valid answer_type; expected "
                f"'Single Answer' or 'Set Answer', got {prompt_type!r}"
            )

        correct_answer = (
            str(ground_truth[0])
            if len(ground_truth) == 1
            else json.dumps(ground_truth, ensure_ascii=False)
        )
        return DSQA_JUDGE_PROMPT.format(
            prompt=question,
            prompt_type=prompt_type,
            correct_answer=correct_answer,
            response=answer,
        )


@register_verifier("dsqa_qa_llm_judge")
class DSQAJudge(BaseQALLMJudge):
    """DeepSearchQA judge implementing the benchmark's published protocol."""

    def _build_messages(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if metadata is None:
            raise ValueError("DeepSearchQA judging requires row metadata")
        user_prompt = self.prompt_factory.get_prompt(
            role="user",
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            metadata=metadata,
        )
        return [{"role": "user", "content": user_prompt}]

    @staticmethod
    def _extract_payload(
        content: str,
        *,
        allow_empty_details: bool = False,
    ) -> dict[str, Any] | None:
        text = (content or "").strip()
        if not text or text.upper() == "NULL":
            return None

        if text.startswith("```"):
            first_newline = text.find("\n")
            last_fence = text.rfind("```")
            if first_newline >= 0 and last_fence > first_newline:
                text = text[first_newline + 1:last_fence].strip()

        decoder = json.JSONDecoder()
        candidates = [text]
        first_brace = text.find("{")
        if first_brace > 0:
            candidates.append(text[first_brace:])

        payload = None
        for candidate in candidates:
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
        if payload is None:
            return None

        correctness = payload.get("Answer Correctness")
        if not isinstance(correctness, dict):
            return None
        explanation = correctness.get("Explanation")
        details = correctness.get("Correctness Details")
        excessive = correctness.get("Excessive Answers")
        if not isinstance(explanation, str):
            return None
        if (
            not isinstance(details, dict)
            or (not details and not allow_empty_details)
            or not all(
            isinstance(key, str) and isinstance(value, bool)
            for key, value in details.items()
            )
        ):
            return None
        if not isinstance(excessive, list) or not all(
            isinstance(item, str) for item in excessive
        ):
            return None
        return correctness

    def should_retry_llm_call(
        self,
        result: LLMCallResult | None = None,
        exc: Exception | None = None,
    ) -> bool:
        if exc is not None:
            return True
        if result is None:
            return True
        content = result.content or ""
        if self._extract_payload(content) is not None:
            return False
        relaxed = self._extract_payload(content, allow_empty_details=True)
        return not relaxed or not relaxed["Excessive Answers"]

    async def async_judge(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeResult:
        result = await super().async_judge(
            question=question,
            ground_truth=ground_truth,
            answer=answer,
            metadata=metadata,
        )
        if result.complete_status is not JudgeCompleteStatus.EXTRACTION_FAILED:
            return result

        payload = self._extract_payload(
            result.raw_response,
            allow_empty_details=True,
        )
        if not payload or payload["Correctness Details"] or not payload["Excessive Answers"]:
            return result

        false_negatives = len(ground_truth)
        false_positives = len(payload["Excessive Answers"])
        judge_model = str(
            getattr(self.client_config.llm_client, "model_name", "")
        )
        return JudgeResult(
            is_correct=False,
            score=0.0,
            reasoning=payload["Explanation"],
            complete_status=JudgeCompleteStatus.SUCCESS,
            raw_response=result.raw_response,
            metrics={
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "true_positives": 0,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "expected_answers": len(ground_truth),
                "excessive_answers": false_positives,
                "judge_model": judge_model,
                "configured_judge_model": "gemini-2.5-flash",
                "matches_configured_judge_model": (
                    "gemini-2.5-flash" in judge_model.lower()
                ),
            },
        )

    def _parse_response(self, content: str) -> JudgeResult:
        payload = self._extract_payload(content)
        if payload is None:
            return JudgeResult(
                is_correct=False,
                complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
                raw_response=content,
            )

        details = payload["Correctness Details"]
        excessive = payload["Excessive Answers"]
        true_positives = sum(value is True for value in details.values())
        false_negatives = sum(value is False for value in details.values())
        false_positives = len(excessive)

        precision_denominator = true_positives + false_positives
        recall_denominator = true_positives + false_negatives
        precision = (
            true_positives / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positives / recall_denominator
            if recall_denominator
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        is_correct = all(details.values()) and not excessive

        judge_model = str(
            getattr(self.client_config.llm_client, "model_name", "")
        )
        return JudgeResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=payload["Explanation"],
            complete_status=JudgeCompleteStatus.SUCCESS,
            raw_response=content,
            metrics={
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "expected_answers": len(details),
                "excessive_answers": false_positives,
                "judge_model": judge_model,
                "configured_judge_model": "gemini-2.5-flash",
                "matches_configured_judge_model": (
                    "gemini-2.5-flash" in judge_model.lower()
                ),
            },
        )
