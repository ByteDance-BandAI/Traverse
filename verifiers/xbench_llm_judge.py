# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import re

from llm_clients.base import LLMCallResult
from prompt.base import BasePromptBuilder, register_prompt_set
from verifiers import JudgeCompleteStatus, JudgeResult, register_verifier
from verifiers.bc_llm_judge import BrowseCompJudge


XBENCH_JUDGE_PROMPT = """你是一个通用人工智能助手。根据下面给出的[正确答案]，判断以下对[原问题]的[回答]是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]: {answer}

你的判断必须按照以下格式和标准进行：

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案，则填写“无”。

解释: 解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异，不要尝试重新解题。

结论: 如果[最终答案]与[正确答案]一致，或在数值题目中处于可接受的微小误差范围内，则填写“正确”；否则填写“错误”。
"""


@register_prompt_set("xbench_judge")
class XBenchJudgePrompt(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        return "你是一个严谨的答案评测助手。"

    def build_user_prompt(self, question, ground_truth, answer) -> str:
        return XBENCH_JUDGE_PROMPT.format(
            question=question,
            correct_answer=", ".join(str(item) for item in ground_truth),
            answer=answer,
        )


@register_verifier("xbench_qa_llm_judge")
class XBenchJudge(BrowseCompJudge):
    """Judge compatible with the official xbench-DeepSearch grading format."""

    _CORRECT_RE = re.compile(r"结论\s*[:：]\s*(正确|错误)")
    _REASONING_RE = re.compile(
        r"解释\s*[:：]\s*(.*?)(?=\n\s*结论\s*[:：]|\Z)",
        re.DOTALL,
    )

    async def async_judge(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
    ) -> JudgeResult:
        normalized_answer = str(answer).strip()
        if any(normalized_answer == str(item).strip() for item in ground_truth):
            return JudgeResult(
                is_correct=True,
                score=1.0,
                reasoning="答案与标准答案完全一致，无需调用 LLM judge。",
                complete_status=JudgeCompleteStatus.SUCCESS,
            )
        return await super().async_judge(question, ground_truth, answer)

    def should_retry_llm_call(
        self,
        result: LLMCallResult | None = None,
        exc: Exception | None = None,
    ) -> bool:
        if exc is not None:
            return True
        return result is not None and not self._CORRECT_RE.search(result.content or "")

    def _parse_response(self, content: str) -> JudgeResult:
        correct_match = self._CORRECT_RE.search(content or "")
        if not correct_match:
            return JudgeResult(
                is_correct=False,
                complete_status=JudgeCompleteStatus.EXTRACTION_FAILED,
                raw_response=content,
            )

        is_correct = correct_match.group(1) == "正确"
        reasoning_match = self._REASONING_RE.search(content or "")
        return JudgeResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=reasoning_match.group(1).strip() if reasoning_match else "",
            complete_status=JudgeCompleteStatus.SUCCESS,
            raw_response=content,
        )
