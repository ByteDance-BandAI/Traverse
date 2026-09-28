# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from prompt.base import BasePromptBuilder, register_prompt_set
from tools import tool_registry


def _read_prompt_template(filename: str) -> str:
    prompt_file = Path(__file__).resolve().parent / "assets" / filename
    return prompt_file.read_text(encoding="utf-8")


SEAL_SYS_PROMPT_EN = _read_prompt_template("prompt.txt")
SEAL_SYS_PROMPT_ZH = _read_prompt_template("prompt_zh.txt")


@register_prompt_set("seal")
class SealAgentPrompt(BasePromptBuilder):
    def build_system_prompt(self, tool_set: list[str], prompt_language: str = "en") -> str:
        tool_prompt = tool_registry.get_prompt(tool_set)
        system_prompt = SEAL_SYS_PROMPT_ZH if prompt_language == "zh" else SEAL_SYS_PROMPT_EN
        return system_prompt.format(available_tools=tool_prompt)

    def build_user_prompt(
        self,
        question: str,
        carryover_text: str = None,
        prompt_language: str = "en",
    ) -> str:
        if prompt_language == "zh":
            base = f"问题：\n{question}"
            if carryover_text:
                return f"{base}\n\n{carryover_text}"
            return base
        if carryover_text:
            return f"{question}\n\n{carryover_text}"
        return question
