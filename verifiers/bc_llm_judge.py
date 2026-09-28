# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from verifiers.base_qa_llm_judge import BaseQALLMJudge, register_verifier
from typing import Any
from verifiers import JudgeResult, JudgeCompleteStatus
from prompt.base import BasePromptBuilder, register_prompt_set
import re
from llm_clients.base import LLMCallResult
from config import GeneralLLMConfig


BCJUDGE_PROMPT = """Please determine whether the provided answer is correct.

Question: {question}

Ground truth: {gt_str}

User answer: {answer}

Evaluate whether the user answer is equivalent to the ground truth. Consider:
1. Numeric equality (ignore formatting differences such as 1000 vs 1,000)
2. Semantic equivalence of text (ignore casing, punctuation, whitespace)
3. For lists, ensure elements match regardless of order
4. Equivalence of date formats (e.g., 2024-01-01 vs January 1, 2024)

**But ignore format inconsistency.**

Return the result in JSON:
{{
    "reasoning": "Brief explanation",
    "decision": true/false
}}

Return JSON only, with no extra commentary."""


BC_OFFICIAL_GRADER_PROMPT = r"""
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def _format_correct_answer(ground_truth: Any) -> str:
    if isinstance(ground_truth, (list, tuple)):
        if len(ground_truth) == 1:
            return str(ground_truth[0])
        return json.dumps(list(ground_truth), ensure_ascii=False)
    return str(ground_truth)


@register_prompt_set("bc_judge")
class BCJudgePrompt(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        return (
            "You are a professional answer evaluator. Determine whether "
            "the provided answer is equivalent to the ground truth. Respond in JSON."
        )

    def build_user_prompt(self, question, ground_truth, answer) -> str:
        return BCJUDGE_PROMPT.format(
            question=question,
            gt_str=ground_truth,
            answer=answer
        )


@register_prompt_set("bc_official_judge")
class BCOfficialJudgePrompt(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        return "You are a helpful assistant."

    def build_user_prompt(self, question, ground_truth, answer) -> str:
        return BC_OFFICIAL_GRADER_PROMPT.format(
            question=question,
            correct_answer=_format_correct_answer(ground_truth),
            response=answer,
        )


@register_verifier("bc_qa_llm_judge")
class BrowseCompJudge(BaseQALLMJudge):
    def __init__(
        self,
        prompt_set: str,
        client_config: GeneralLLMConfig,
        max_retry_times: int = 3,
        retry_infinitely_on_429: bool = False,
    ) -> None:
        super().__init__(
            prompt_set,
            client_config,
            max_retry_times,
            retry_infinitely_on_429=retry_infinitely_on_429,
        )

    def should_retry_llm_call(self, result: LLMCallResult | None = None, exc: Exception | None = None) -> bool:
        # Retry if we can't parse the json, this should cover the repeation length error
        if result is not None:
            json_match = re.search(r'\{[^{}]*\}', result.content or "", re.DOTALL)
            if not json_match:
                return True

        # Retry on any exceptions
        if exc is not None:
            return True

        return False

    def _parse_response(self, content: str) -> JudgeResult:
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)

        if not json_match:
            return JudgeResult(
                is_correct=False, 
                complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
                raw_response=content,
            )

        try:
            json_str = json_match.group(0)
            judgement = json.loads(json_str)
            decision = judgement["decision"]
            reasoning = judgement["reasoning"]
        except Exception:
            # Some otherwise usable judge responses contain unescaped quotes in
            # the free-form reasoning string (for example a quoted nickname).
            # Recover only when the required boolean decision field is still
            # unambiguous; malformed or missing decisions remain hard failures.
            decision_match = re.search(
                r'"decision"\s*:\s*(true|false)\b', json_str, re.IGNORECASE
            )
            if not decision_match:
                return JudgeResult(
                    is_correct=False,
                    complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
                    raw_response=content,
                )
            decision = decision_match.group(1).lower() == "true"
            reasoning_match = re.search(
                r'"reasoning"\s*:\s*"(.*)"\s*,\s*"decision"\s*:',
                json_str,
                re.DOTALL,
            )
            reasoning = reasoning_match.group(1) if reasoning_match else ""

        return JudgeResult(
            is_correct=decision,
            reasoning=reasoning,
            complete_status=JudgeCompleteStatus.SUCCESS,
            raw_response=content,
        )


@register_verifier("bc_official_qa_llm_judge")
class BrowseCompOfficialJudge(BaseQALLMJudge):
    def __init__(
        self,
        prompt_set: str,
        client_config: GeneralLLMConfig,
        max_retry_times: int = 3,
    ) -> None:
        super().__init__(
            prompt_set,
            client_config,
            max_retry_times,
        )

    @staticmethod
    def parse_grader_fields(content: str) -> dict[str, str]:
        fields = {}
        pattern = re.compile(
            r"(?im)^\s*(extracted_final_answer|reasoning|correct|confidence)\s*:\s*"
        )
        matches = list(pattern.finditer(content or ""))
        for idx, match in enumerate(matches):
            key = match.group(1).lower()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            fields[key] = content[start:end].strip()
        return fields

    @staticmethod
    def _find_correct(content: str) -> re.Match | None:
        return re.search(r"(?im)^\s*correct\s*:\s*(yes|no)\b", content or "")

    def should_retry_llm_call(self, result: LLMCallResult | None = None, exc: Exception | None = None) -> bool:
        if result is not None and not self._find_correct(result.content or ""):
            return True

        if exc is not None:
            return True

        return False

    def _parse_response(self, content: str) -> JudgeResult:
        correct_match = self._find_correct(content)
        if not correct_match:
            return JudgeResult(
                is_correct=False,
                complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
                raw_response=content,
            )

        fields = self.parse_grader_fields(content)
        is_correct = correct_match.group(1).lower() == "yes"
        return JudgeResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=fields.get("reasoning", ""),
            complete_status=JudgeCompleteStatus.SUCCESS,
            raw_response=content,
        )
