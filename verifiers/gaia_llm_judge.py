# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from prompt.base import BasePromptBuilder, register_prompt_set
from verifiers import register_verifier
from verifiers.bc_llm_judge import BrowseCompJudge


GAIA_JUDGE_PROMPT = """Please determine whether the provided answer is correct.

Question: {question}

Ground truth: {gt_str}

User answer: {answer}

Evaluate whether the user answer matches the ground truth. Consider:
1. Numeric equivalence (ignore formatting differences such as 1000 vs 1,000)
2. Semantic equivalence of text (ignore case, punctuation, whitespace)
3. For lists, ensure elements match even if order differs
4. Equivalence of date formats (e.g., 2024-01-01 vs January 1, 2024)

Return the judgment as JSON:
{{
    "reasoning": "Brief explanation",
    "decision": true/false
}}

Return JSON only, with no additional text."""


@register_prompt_set("gaia_judge")
class GAIAJudgePrompt(BasePromptBuilder):
    def build_system_prompt(self) -> str:
        return (
            "You are a professional answer evaluator. Determine whether the "
            "provided answer matches the ground truth. Respond in JSON."
        )

    def build_user_prompt(self, question, ground_truth, answer) -> str:
        return GAIA_JUDGE_PROMPT.format(
            question=question,
            gt_str=", ".join(str(item) for item in ground_truth),
            answer=answer,
        )


@register_verifier("gaia_qa_llm_judge")
class GAIAJudge(BrowseCompJudge):
    """GAIA-Text-103 judge matching the legacy SimpleInference protocol."""
