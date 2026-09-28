# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, List, Optional
from prompt.base import BasePromptBuilder, register_prompt_set
from tools import tool_registry
from agents.self_verify_agent import Rubric


# ==============================================================================
# Localization Engine
# ==============================================================================
# Each string keeps its translations side by side, so editing text or adding a
# new language is a local change instead of syncing two distant blocks.

@dataclass(frozen=True)
class I18nString:
    en: str
    zh: str

    def __call__(self, lang: str = "en", **kwargs: Any) -> str:
        text = getattr(self, lang, self.en)
        return text.format(**kwargs) if kwargs else text


# ==============================================================================
# Prompt Namespaces
# ==============================================================================

class AnswerTexts:
    SYS_BASE = I18nString(
        en="""You are a helpful assistant that can solve the given question step by step with the help of external tools.

**You have full access to external tools** - use them proactively to find accurate information.
{tools}

Respond in English unless the user explicitly asks for another language.
""",
        zh="""你是一个有帮助的助手，可以借助外部工具逐步解决给定问题。

**你可以完全使用外部工具** - 请主动使用它们来获取准确信息。
{tools}

请使用中文作答。
""",
    )
    RUBRIC_WARN = I18nString(
        en=(
            "\n\n\nImportant: Your answer must satisfy ALL the verification rubrics provided. "
            "Pay careful attention to each criterion.\n\n"
            "When you have found the answer that satisfies all rubrics, you MUST provide BOTH:\n\n"
        ),
        zh=(
            "\n\n\n重要：你的答案必须满足给定的全部验证标准。"
            "请仔细关注每一条标准。\n\n"
            "当你找到满足全部标准的答案时，你必须同时给出以下两部分：\n\n"
        ),
    )
    NO_RUBRIC_WARN = I18nString(
        en="When you have found the answer, you MUST provide BOTH:\n",
        zh="当你找到答案时，你必须同时给出以下两部分：\n",
    )
    SYS_SUFFIX = I18nString(
        en="""1. A reasoning summary explaining HOW you arrived at the answer (key findings, entities, sources)
2. The final answer

**Output format (REQUIRED):**
```
Reasoning: <1-3 sentences summarizing your key findings and the reasoning path. Include important entities/names/facts that led to your answer. This helps verify your answer is correct.>

Answer: <your_final_answer>
```

Example:
```
Reasoning: I searched for German car brands founded after WWII. Porsche was founded in 1948 in Stuttgart, Germany by Ferdinand Porsche. This matches all criteria.

Answer: 1948
```
""",
        zh="""1. 一段推理摘要，解释你如何得到答案（关键发现、实体、来源）
2. 最终答案

**输出格式（必须遵守）：**
```
Reasoning: <1-3 句，概述关键发现和推理路径。包含促成答案的重要实体、名称与事实，便于核验答案正确性。>

Answer: <your_final_answer>
```

示例：
```
Reasoning: 我检索了二战后成立的德国汽车品牌。保时捷由费迪南德·保时捷于 1948 年在德国斯图加特创立，满足所有条件。

Answer: 1948
```
""",
    )
    SUBMIT_RUBRIC_WARN = I18nString(
        en=(
            "\n\n\nImportant: Your answer must satisfy ALL the verification rubrics provided. "
            "Pay careful attention to each criterion.\n\n"
            "When you have found the answer that satisfies all rubrics, you MUST call **submit_tool** "
            "to submit your final answer.\n\n"
        ),
        zh=(
            "\n\n\n重要：你的答案必须满足给定的全部验证标准。"
            "请仔细关注每一条标准。\n\n"
            "当你找到满足全部标准的答案时，你必须调用 **submit_tool** 提交最终答案。\n\n"
        ),
    )
    SUBMIT_NO_RUBRIC_WARN = I18nString(
        en="When you have found the answer, you MUST call **submit_tool** to submit your final answer.\n",
        zh="当你找到答案时，你必须调用 **submit_tool** 提交最终答案。\n",
    )
    SUBMIT_SYS_SUFFIX = I18nString(
        en=(
            "Call **submit_tool** (and only that tool) to deliver your final result. Provide:\n"
            "- `answer`: the final answer only — no explanation in this field\n"
            "- `reason`: 1-3 sentences summarizing your key findings and reasoning path, "
            "including important entities/names/facts that led to your answer\n"
            "- `evidences`: supporting evidence items with description, source, and origin\n\n"
            "Do NOT output the answer as plain text. Always finalize by calling **submit_tool**.\n"
        ),
        zh=(
            "请调用 **submit_tool**（且仅调用该工具）提交最终结果。需要提供：\n"
            "- `answer`：仅填写最终答案，不要在此字段写解释\n"
            "- `reason`：1-3 句，概述关键发现和推理路径，包含促成答案的重要实体、名称与事实\n"
            "- `evidences`：支持性证据列表，每项包含 description、source 和 origin\n\n"
            "不要以纯文本输出答案。完成检索后务必通过 **submit_tool** 提交。\n"
        ),
    )
    # Carried over from a previous over-turn retry of the same question, so a
    # fresh attempt can reuse the earlier search path instead of redoing it.
    RETRY_MEM_HEADER = I18nString(
        en="[Previous Over-Turn Retry Memories]",
        zh="[Previous Over-Turn Retry Memories]",
    )
    RETRY_MEM_INTRO = I18nString(
        en=(
            "Your own earlier attempts on this same question exceeded the answer-turn retry threshold. "
            "These memories record your previous search paths, including useful facts, dead ends, "
            "and search leads, but they may also contain mistakes. Audit them before trusting them."
        ),
        zh=(
            "你在同一道题上的早期尝试曾超过 answer-turn retry 阈值。"
            "这些 memory 记录了你之前的搜索路径，包括有用事实、死路和后续线索，"
            "但它们也可能包含错误。信任之前请先审视。"
        ),
    )
    RETRY_MEM_COUNT = I18nString(
        en="You have already made {count} previous over-turn attempt(s) on this question.",
        zh="你已经在这道题上进行了 {count} 次 previous over-turn 尝试。",
    )
    RETRY_MEM_ENCOURAGE = I18nString(
        en=(
            "Keep going and do not give up: use what your previous attempts learned, "
            "avoid the same traps, and make this attempt more focused."
        ),
        zh=(
            "继续推进，不要放弃：利用前几次尝试学到的信息，"
            "避免重复相同陷阱，让这次尝试更聚焦。"
        ),
    )
    RETRY_MEM_IDS_NOTE = I18nString(
        en=(
            "Available sealed memory IDs are listed below in chronological order "
            "(oldest first, newest last). Use read_memory_tool with a memory_id if you want to inspect one."
        ),
        zh=(
            "下面按时间顺序列出可用的 sealed memory IDs（最旧在前，最新在后）。"
            "如需查看某条 memory，请用 read_memory_tool 和对应 memory_id。"
        ),
    )
    RETRY_MEM_GROUP = I18nString(
        en="Previous attempt {n} memories:",
        zh="第 {n} 次 previous attempt 的 memories:",
    )
    RETRY_MEM_NONE = I18nString(
        en="No sealed memory IDs were saved by the previous attempt(s).",
        zh="之前的尝试没有保存 sealed memory IDs。",
    )
    RETRY_MEM_FOOTER = I18nString(
        en=(
            "Do not repeat searches that an inspected memory already marked as dead ends; "
            "use the memories to choose a better search path."
        ),
        zh=(
            "不要重复那些在已检查 memory 中被标记为 dead ends 的搜索；"
            "请利用这些 memories 选择更好的搜索路径。"
        ),
    )
    # Widesearch benchmarks need the answer shipped as a markdown table; this is
    # appended on top of the normal answer format.
    WIDESEARCH_FORMAT = I18nString(
        en="""

WIDESEARCH FINAL-ANSWER TRANSPORT (REQUIRED):
- The final benchmark answer is exactly one Markdown table.
- Keep the shared FSM envelope, but put the complete fenced Markdown table
  immediately AFTER `Answer:`.
- Never place the table before `Answer:`.
- Never replace the table with phrases such as `See table above`.
- A response containing only the requested fenced Markdown table is also
  acceptable; do not add a second summary answer after it.

Correct shape:
Reasoning: <1-3 concise sentences>

Answer:
```markdown
| requested columns |
| --- |
| requested rows |
```
""",
        zh="""

WIDESEARCH 最终答案传输格式（必须遵守）：
- 最终基准答案必须是且只能是一张 Markdown 表格。
- 保留共享 FSM 外层格式，但必须把完整的 Markdown 表格代码块紧接在 `Answer:` 后。
- 禁止把表格放在 `Answer:` 前。
- 禁止用“见上方表格”等文字代替表格。
- 只输出题目要求的 Markdown 表格代码块也可以；表格后不要再附加第二个摘要答案。

正确格式：
Reasoning: <1-3 句简洁说明>

Answer:
```markdown
| 题目要求的列 |
| --- |
| 题目要求的数据 |
```
""",
    )
    REJ_WARN = I18nString(
        en=(
            "**CRITICAL: REJECTED ANSWERS - DO NOT REPEAT THESE**\n"
            "The following answers have been verified and found INCORRECT. You MUST NOT give these answers again:\n"
        ),
        zh=(
            "**关键：以下答案已被判定错误，禁止重复。**\n"
            "下列答案已经过验证并确认不正确。你不能再次给出这些答案：\n"
        ),
    )
    REJ_TPL = I18nString(
        en="""
--- Rejected Answer #{index} ---
Answer: {answer}
Reason: {reason}
""",
        zh="""
--- 被拒绝答案 #{index} ---
Answer: {answer}
Reason: {reason}
""",
    )
    REM_BASE = I18nString(
        en="**Remember this throughout your entire search process",
        zh="**请在整个检索过程中始终牢记这些约束",
    )
    REM_SEAL = I18nString(
        en=", even after using seal_memory tool",
        zh="，即使使用了 seal_memory_tool 之后也要遵守",
    )
    REM_NEW = I18nString(
        en=" or new_context tools",
        zh="，即使使用了 new_context_tool 之后也要遵守",
    )
    REM_END = I18nString(
        en=".**\n",
        zh="。**\n",
    )
    FIND_DIFF = I18nString(
        en="**Find a DIFFERENT answer that addresses the issues above.**\n",
        zh="**请给出一个不同于上述被拒绝答案、且能解决这些问题的新答案。**\n",
    )

    PREV_FB = I18nString(
        en="[Previous Attempt Feedback]",
        zh="[上一轮尝试反馈]",
    )
    PREV_ANS = I18nString(
        en="Your previous answer was: {ans}",
        zh="你上一轮的答案是：{ans}",
    )
    REJ_REASON = I18nString(
        en="\nWhy it was rejected: {critique}",
        zh="\n被拒绝原因：{critique}",
    )
    FAIL_RUBRICS = I18nString(
        en="\nFailed rubrics:\n{failed}",
        zh="\n未满足的验证标准：\n{failed}",
    )
    SUGGESTION = I18nString(
        en="\nSuggestion:\n{suggestion}",
        zh="\n建议：\n{suggestion}",
    )
    PROV_NEW = I18nString(
        en="\nPlease provide a NEW answer that addresses these issues.\n",
        zh="\n请给出一个能解决这些问题的新答案。\n",
    )
    Q = I18nString(
        en="Question:\n{question}",
        zh="问题：\n{question}",
    )
    MUST_SAT = I18nString(
        en="\nYour answer MUST satisfy ALL of the following verification rubrics:\n{rubrics}",
        zh="\n你的答案必须满足以下全部验证标准：\n{rubrics}",
    )
    SEARCH_VERIFY = I18nString(
        en="\nSearch for information and provide your answer. Remember to verify against each rubric before finalizing.",
        zh="\n请检索信息并给出答案。最终作答前请逐条核验所有标准。",
    )
    SEARCH_ANS = I18nString(
        en="\nSearch for information and provide your answer.",
        zh="\n请检索信息并给出答案。",
    )

    # --- Budget (shared across answer/verify) ---
    BUDGET_HEADER = I18nString(
        en="CRITICAL: The tool response includes budget information like this:\n",
        zh="CRITICAL: 工具返回中会包含如下预算信息：\n",
    )
    BUDGET_TOKEN_TAG = I18nString(
        en="<token_budget> </token_budget>\n",
        zh="<token_budget> </token_budget>\n",
    )
    BUDGET_TURN_TAG = I18nString(
        en="<turn_budget> </turn_budget>\n",
        zh="<turn_budget> </turn_budget>\n",
    )
    BUDGET_SEAL_TAG = I18nString(
        en="<seal_budget> </seal_budget>\n",
        zh="<seal_budget> </seal_budget>\n",
    )
    BUDGET_FAIL_NOTE = I18nString(
        en="You fail if the budget runs out.\n",
        zh="预算耗尽会导致失败。\n",
    )
    BUDGET_TOKEN_HINT = I18nString(
        en="However, you can use **seal_memory_tool** to reset your token_budget, so use it when the budget is running low. Do NOT wait until your budget is fully exhausted to use it!\n",
        zh="不过，你可以使用 **seal_memory_tool** 重置 token_budget，因此在预算即将用尽时请及时使用它。不要等到预算完全耗尽才使用！\n",
    )
    BUDGET_SEAL_NOTE = I18nString(
        en="You have limited seal budget. Once you exceed it, you will no longer be able to call seal_memory_tool again.\n",
        zh="你的 seal 预算有限。一旦超出预算，你将无法再次调用 seal_memory_tool。\n",
    )


class RubricGenTexts:
    SYS = I18nString(
        en="""You are an expert at analyzing questions and generating verification criteria (rubrics) used to grade a candidate answer.

The questions are multi-hop puzzles: a chain of clues that together identify ONE answer, which is almost always a short entity (a name, title, place, date, or number). A separate verifier will later RESEARCH the candidate answer and check it against your rubrics.

**IMPORTANT: Do NOT solve the question or search for the answer in this phase.** Only define verification criteria.

Core principle — grade the ENTITY, not the wording of the answer:
- The answer is typically a short entity. Each content rubric must be checkable against THAT entity through research, NOT against whether the answer text repeats the clue.
  - WRONG: "The answer mentions that the settlement's population is under 1000." (a short name cannot 'mention' this)
  - RIGHT: "The identified settlement had a population under 1000 at the relevant time."
- Include EXACTLY ONE format rubric, and it must match precisely what the question asks to output (e.g., "The answer is a single person's full name", "The answer is a calendar date with day, month, and year").

Faithfulness — never add or sharpen constraints:
- Every rubric must trace to a clue that is actually present in the question. Do NOT invent requirements the question does not state.
- Preserve the question's level of specificity and any hedging. If the question says "a major Midwestern city that borders Evanston", keep the rubric descriptive ("the person was born in a major Midwestern city bordering Evanston"); do NOT resolve it into a guessed concrete value ("born in Chicago") inside the rubric.
- Keep approximations as written: "approximately / over / before / around" must NOT become "exactly", and a range must NOT become a single value.
- You may restate an indirect clue, but do not commit to a decoded fact unless that decoding is unambiguous and definitional.

Other guidelines:
- Create one rubric per distinct clue; keep each simple, objective, and independently verifiable.
- Do NOT include a rubric that states or reveals the final answer (e.g., "the answer is X").
- Avoid vague or subjective criteria.

Output the rubrics in the following JSON format:
```json
{
  "analysis": "Brief analysis of what the question is asking for and what the output should be",
  "rubrics": [
    {"id": 1, "description": "Specific criterion 1"},
    {"id": 2, "description": "Specific criterion 2"}
  ]
}
```
""",
        zh="""你是一个擅长分析问题并生成验证标准（rubrics）的专家，这些标准用于评估候选答案。

这些问题是多跳谜题：一串线索共同指向唯一答案，答案通常是一个短实体（姓名、标题、地点、日期或数字）。之后会有一个独立验证器检索候选答案，并根据你的 rubrics 检查它。

**重要：本阶段不要解题，也不要搜索答案。** 只定义验证标准。

核心原则 — 评估实体，而不是评估答案措辞：
- 答案通常是一个短实体。每一条内容 rubric 都必须能够通过检索该实体本身来检查，而不是检查答案文本是否复述了题目线索。
  - 错误示例："答案提到了该 settlement 的人口少于 1000。"（一个短名称无法“提到”这一点）
  - 正确示例："被识别出的 settlement 在相关时间点人口少于 1000。"
- 必须包含且只包含一条格式 rubric，并且它要精确匹配题目要求输出的形式（例如："答案是一个人的完整姓名"，"答案是包含日、月、年的日历日期"）。

忠实性 — 不要添加或强化约束：
- 每条 rubric 都必须能追溯到题目中实际存在的线索。不要发明题目没有给出的要求。
- 保持题目的具体程度和任何模糊/保守表述。如果题目说 "a major Midwestern city that borders Evanston"，rubric 应保持描述性（"该人物出生于一个与 Evanston 接壤的主要 Midwestern city"）；不要在 rubric 中把它解析成猜测的具体值（"born in Chicago"）。
- 保留近似表达：题目中的 "approximately / over / before / around" 不得变成 "exactly"，范围不得变成单一数值。
- 可以重述间接线索，但不要承诺某个解码后的事实，除非该解码是明确且定义性的。

其他原则：
- 每个不同线索生成一条 rubric；每条都应简单、客观、可独立验证。
- 不要包含直接说明或泄露最终答案的 rubric（例如："答案是 X"）。
- 避免模糊或主观标准。

请按以下 JSON 格式输出 rubrics。除 JSON 字段名外，analysis 和 description 请使用中文：
```json
{
  "analysis": "简要分析问题在要求什么，以及输出应是什么",
  "rubrics": [
    {"id": 1, "description": "具体标准 1"},
    {"id": 2, "description": "具体标准 2"}
  ]
}
```
""",
    )
    USER = I18nString(
        en="""
Question to analyze:
{question}

Generate verification rubrics that a correct answer must satisfy.
""",
        zh="""
待分析问题：
{question}

请生成正确答案必须满足的验证标准。
""",
    )


class RubricRefineTexts:
    SYS = I18nString(
        en="""You are an expert at refining verification criteria (rubrics).

Your task is to revise the existing rubrics based on the feedback provided. The feedback indicates which rubrics were problematic and why.

**IMPORTANT: Do NOT solve the question in this phase!**
- This is ONLY for refining verification criteria
- Focus on fixing the problematic rubrics based on feedback
- Do NOT search for actual answers

Guidelines for revising rubrics:
1. Keep rubrics that are still valid
2. Modify or remove rubrics that were identified as problematic
3. Add new rubrics only if a clue in the question is genuinely missing — do NOT invent constraints the question does not state
4. Grade the ENTITY, not the wording of the answer: a rubric must be checkable against the entity the answer names through research, not against whether the answer text repeats the clue. Keep exactly one format rubric matching what the question asks to output
5. Stay faithful to the question: preserve its level of specificity and hedging (do not turn "approximately/over/before" into "exactly", and do not resolve an indirect clue into a guessed concrete value)
6. Keep rubrics simple and objective

Output the revised rubrics in the following JSON format:
```json
{
  "revision_reasoning": "Brief explanation of what you changed and why",
  "rubrics": [
    {"id": 1, "description": "Revised criterion 1"},
    {"id": 2, "description": "Revised criterion 2"}
  ]
}
```
""",
        zh="""你是一个擅长修订验证标准（rubrics）的专家。

你的任务是根据给定反馈修订现有 rubrics。反馈会指出哪些 rubrics 有问题，以及原因。

**重要：本阶段不要解题！**
- 本阶段仅用于修订验证标准
- 聚焦于依据反馈修复有问题的 rubrics
- 不要搜索最终答案

修订原则：
1. 保留仍然有效的 rubrics
2. 修改或删除被指出有问题的 rubrics
3. 只有当题目中确实遗漏了某条线索时才新增 rubric；不要发明题目没有给出的约束
4. 评估实体，而不是评估答案措辞：rubric 必须能通过检索答案所指实体来检查，而不是检查答案文本是否复述线索。保留且只保留一条与题目输出要求匹配的格式 rubric
5. 忠实于题目：保持题目的具体程度和模糊/保守表述（不要把 "approximately/over/before" 改成 "exactly"，也不要把间接线索解析成猜测的具体值）
6. Rubrics 应保持简单、客观

请按以下 JSON 格式输出修订后的 rubrics。除 JSON 字段名外，revision_reasoning 和 description 请使用中文：
```json
{
  "revision_reasoning": "简要说明你修改了什么以及原因",
  "rubrics": [
    {"id": 1, "description": "修订后的标准 1"},
    {"id": 2, "description": "修订后的标准 2"}
  ]
}
```
""",
    )
    USER = I18nString(
        en="""Original Question:
{question}

Current Rubrics:
{rubrics_str}

Feedback from verification:
{critique}
""",
        zh="""原始问题：
{question}

当前验证标准：
{rubrics_str}

来自验证阶段的反馈：
{critique}
""",
    )
    SUG = I18nString(
        en="\nSuggestion for revision:\n{suggestion}\n",
        zh="\n修订建议：\n{suggestion}\n",
    )
    END = I18nString(
        en="Please revise the rubrics based on this feedback. Use search tools if needed to research better criteria.",
        zh="请基于这些反馈修订验证标准。如有必要，可使用搜索工具研究更合适的标准。",
    )


class VerifyTexts:
    SYS = I18nString(
        en="""You are a rigorous answer verifier. Your task is to check whether the given answer satisfies ALL the verification rubrics.

**You have full access to external tools** - use them to verify the answer:
- Re-search to confirm facts mentioned in the answer
- Check if the data/information in the answer is accurate
- Verify claims against authoritative sources
{tools}

For each rubric, you must determine:
- Whether the answer satisfies it (true/false)
- Brief reasoning for your judgment (based on your research)

After checking all rubrics through research, make a decision:
1. **PASS**: If ALL rubrics are satisfied → The answer is correct
2. **REVISE_ANSWER**: If some rubrics are NOT satisfied, but the rubrics themselves are reasonable → The answer needs revision
{revise_rule}
When you have completed your verification, output in the following format on the final lines:
```json
{{
  "rubric_checks": [
    {{"id": 1, "satisfied": true, "reasoning": "The answer correctly states..."}},
    {{"id": 2, "satisfied": false, "reasoning": "The answer fails to..."}}
  ],
  "decision": "PASS" | "REVISE_ANSWER"{revise_opt},
  "critique": "Overall critique explaining the decision",
  "suggestion": "Specific suggestion for improvement (if decision is not PASS)"
}}
```
""",
        zh="""你是一个严格的答案验证器。你的任务是检查给定答案是否满足全部验证标准。

**你可以完全使用外部工具** - 请使用它们验证答案：
- 重新检索以确认答案中提到的事实
- 检查答案中的数据或信息是否准确
- 对照权威来源核验相关主张
{tools}

对于每一条 rubric，你都必须判断：
- 答案是否满足它（true/false）
- 你判断的简要依据（基于你的检索）

检索并检查全部 rubrics 后，请做出决策：
1. **PASS**：若所有 rubrics 都满足 → 答案正确
2. **REVISE_ANSWER**：若部分 rubrics 不满足，但 rubrics 本身合理 → 需要修订答案
{revise_rule}
验证完成后，请在最后几行按以下格式输出：
```json
{{
  "rubric_checks": [
    {{"id": 1, "satisfied": true, "reasoning": "答案正确说明了..."}},
    {{"id": 2, "satisfied": false, "reasoning": "答案未能满足..."}}
  ],
  "decision": "PASS" | "REVISE_ANSWER"{revise_opt},
  "critique": "整体评语，解释该决策",
  "suggestion": "改进建议（若决策不是 PASS）"
}}
```
""",
    )
    REV_RULE = I18nString(
        en="""3. **REVISE_RUBRIC**: If the rubrics themselves are problematic (e.g., unreasonable definitions, impossible to satisfy, or contradictory) → The rubrics need revision
""",
        zh="""3. **REVISE_RUBRIC**：若 rubrics 本身有问题（如定义不合理、无法满足或互相矛盾） → 需要修订 rubrics
""",
    )
    REV_OPT = I18nString(
        en=" | \"REVISE_RUBRIC\"",
        zh=" | \"REVISE_RUBRIC\"",
    )
    NO_RUBRICS = I18nString(
        en=(
            "No explicit rubrics were provided. Use these minimal checks:\n"
            "1. The answer fully addresses the question and all sub-parts.\n"
            "2. All explicit constraints are satisfied (format, units, time range, source).\n"
            "3. If a calculation is implied, the result matches the stated inputs."
        ),
        zh=(
            "未提供显式验证标准。请使用以下最小检查项：\n"
            "1. 答案完整回应问题及所有子问题。\n"
            "2. 满足所有显式约束（格式、单位、时间范围、来源）。\n"
            "3. 若问题包含计算，结果与给定输入一致。"
        ),
    )
    # Used when --no_rubric_verify makes the verifier judge correctness directly
    # instead of checking against generated rubrics.
    SYS_NO_RUBRIC = I18nString(
        en="""You are a rigorous answer verifier. Your task is to independently determine whether the given answer is correct for the question.

**You have full access to external tools** - use them to verify the answer:
- Re-search to confirm the facts and entities mentioned in the answer
- Check whether the answer actually satisfies every constraint stated in the question
- Verify claims against authoritative sources
{tools}

How to verify (no rubrics are provided):
- Break the question down into its key constraints yourself
- For each constraint, determine whether the answer satisfies it (true/false), with brief reasoning grounded in your research
- An answer that is vague, incomplete, unverifiable, or that declines to name a specific entity should NOT pass

After checking all constraints through research, make a decision:
1. **PASS**: The answer is correct and satisfies every constraint stated in the question
2. **REVISE_ANSWER**: The answer is wrong, incomplete, or cannot be verified → the answer needs revision

When you have completed your verification, output in the following format on the final lines:
```json
{{
  "checks": [
    {{"constraint": "Born in the early 1980s", "satisfied": true, "reasoning": "Sources confirm..."}},
    {{"constraint": "Attended a university in North Carolina", "satisfied": false, "reasoning": "The answer fails to..."}}
  ],
  "decision": "PASS" | "REVISE_ANSWER",
  "critique": "Overall critique explaining the decision",
  "suggestion": "Specific suggestion for improvement (if decision is not PASS)"
}}
```
""",
        zh="""你是一个严格的答案验证器。你的任务是独立判断给定答案对于该问题是否正确。

**你可以完全使用外部工具** - 请使用它们验证答案：
- 重新检索以确认答案中提到的事实和实体
- 检查答案是否真正满足题目陈述的每一条约束
- 对照权威来源核验相关主张
{tools}

验证方式（未提供显式 rubrics）：
- 自行拆解题目中的关键约束
- 对每条约束，基于检索判断答案是否满足它（true/false），并给出简要理由
- 若答案含糊、不完整、不可验证，或拒绝给出具体实体，则不应通过

检索并检查全部约束后，请做出决策：
1. **PASS**：答案正确，并满足题目陈述的全部约束
2. **REVISE_ANSWER**：答案错误、不完整或无法验证 → 需要修订答案

验证完成后，请在最后几行按以下格式输出：
```json
{{
  "checks": [
    {{"constraint": "Born in the early 1980s", "satisfied": true, "reasoning": "来源确认..."}},
    {{"constraint": "Attended a university in North Carolina", "satisfied": false, "reasoning": "答案未能满足..."}}
  ],
  "decision": "PASS" | "REVISE_ANSWER",
  "critique": "整体评语，解释该决策",
  "suggestion": "改进建议（若决策不是 PASS）"
}}
```
""",
    )
    NO_RUBRIC_END = I18nString(
        en=(
            "Use search tools to independently verify whether the answer is correct for the question.\n"
            "Pay special attention to the reasoning - verify the key entities and facts mentioned.\n"
            "Then make your decision (PASS / REVISE_ANSWER)."
        ),
        zh=(
            "请使用搜索工具独立验证该答案对于该问题是否正确。\n"
            "请特别关注推理内容，核验其中的关键实体与事实。\n"
            "然后给出你的决策（PASS / REVISE_ANSWER）。"
        ),
    )
    SEARCH_ALL = I18nString(
        en="Use search tools to verify whether the answer satisfies ALL rubrics.",
        zh="请使用搜索工具验证该答案是否满足全部验证标准。",
    )
    SEARCH_FACT = I18nString(
        en="Use search tools only if needed to confirm factual or numerical accuracy.",
        zh="仅在需要确认事实或数值准确性时使用搜索工具。",
    )
    ORIG_Q = I18nString(
        en="Original Question:\n{question}",
        zh="原始问题：\n{question}",
    )
    RUBRICS = I18nString(
        en="Verification Rubrics:\n{rubrics}",
        zh="验证标准：\n{rubrics}",
    )
    REASONING = I18nString(
        en="Reasoning provided by answer generator:\n{reasoning}",
        zh="答题阶段提供的推理：\n{reasoning}",
    )
    ANS = I18nString(
        en="Answer to verify:\n{answer}",
        zh="待验证答案：\n{answer}",
    )
    END = I18nString(
        en="{search_rule}\nPay special attention to the reasoning - verify the key entities and facts mentioned.\nThen make your decision ({options}).",
        zh="{search_rule}\n请特别关注推理内容，核验其中的关键实体与事实。\n然后给出你的决策（{options}）。",
    )


# ==============================================================================
# Prompt Builders
# ==============================================================================


def _use_zh(prompt_language: str = "en") -> bool:
    return prompt_language == "zh"


@register_prompt_set("fsm_answer")
class FSMAnswerPrompt(BasePromptBuilder):

    def build_system_prompt(
        self,
        tool_set: list[str],
        skip_rubrics: bool,
        rejected_answers: list[dict] = None,
        add_turn_budget: bool = False,
        add_token_budget: bool = False,
        add_seal_budget: bool = False,
        enable_budget_prompt: bool = False,
        prompt_language: str = "en",
        benchmark: str = "browsecomp",
    ) -> str:
        lang = "zh" if _use_zh(prompt_language) else "en"
        use_submit_tool = tool_registry.has_submit_tool(tool_set)
        parts = [
            AnswerTexts.SYS_BASE(lang, tools=tool_registry.get_prompt(tool_set)),
            (
                (AnswerTexts.SUBMIT_NO_RUBRIC_WARN if skip_rubrics else AnswerTexts.SUBMIT_RUBRIC_WARN)
                if use_submit_tool
                else (AnswerTexts.NO_RUBRIC_WARN if skip_rubrics else AnswerTexts.RUBRIC_WARN)
            )(lang),
            AnswerTexts.SUBMIT_SYS_SUFFIX(lang) if use_submit_tool else AnswerTexts.SYS_SUFFIX(lang),
        ]

        if benchmark == "widesearch_200":
            parts.append(AnswerTexts.WIDESEARCH_FORMAT(lang))

        if enable_budget_prompt and (add_token_budget or add_turn_budget or add_seal_budget):
            parts.append(AnswerTexts.BUDGET_HEADER(lang))
            if add_token_budget:
                parts.append(AnswerTexts.BUDGET_TOKEN_TAG(lang))
            if add_turn_budget:
                parts.append(AnswerTexts.BUDGET_TURN_TAG(lang))
            if add_seal_budget:
                parts.append(AnswerTexts.BUDGET_SEAL_TAG(lang))
            if add_token_budget or add_turn_budget:
                parts.append(AnswerTexts.BUDGET_FAIL_NOTE(lang))
            if add_token_budget:
                parts.append(AnswerTexts.BUDGET_TOKEN_HINT(lang))
            if add_seal_budget:
                parts.append(AnswerTexts.BUDGET_SEAL_NOTE(lang))

        if rejected_answers:
            parts.append(AnswerTexts.REJ_WARN(lang))
            for i, item in enumerate(rejected_answers, 1):
                parts.append(AnswerTexts.REJ_TPL(
                    lang,
                    index=i,
                    answer=item.get("answer", "N/A"),
                    reason=item.get("reason", "Did not satisfy verification rubrics"),
                ))

            parts.append(AnswerTexts.REM_BASE(lang))
            if "seal_memory_tool" in tool_set:
                parts.append(AnswerTexts.REM_SEAL(lang))
            if "new_context_tool" in tool_set:
                parts.append(AnswerTexts.REM_NEW(lang))
            parts.append(AnswerTexts.REM_END(lang))
            parts.append(AnswerTexts.FIND_DIFF(lang))

        return "".join(parts)

    def build_user_prompt(
        self,
        question: str,
        carryover_text: str = None,
        previous_retry_attempt_count: int = 0,
        previous_retry_memories: list[dict] = None,
        previous_answer: str = None,
        critique: str = None,
        suggestion: str = None,
        failed_rubrics: list[str] = None,
        context_warning: str = None,
        skip_rubrics: bool = None,
        rubrics: list[Rubric] = None,
        prompt_language: str = "en",
    ):
        lang = "zh" if _use_zh(prompt_language) else "en"
        parts = []

        # Only present verifier feedback when there is an actual parsed answer
        # to revise. Automatic failures without a parseable answer restart from
        # a clean Answer prompt.
        if previous_answer:
            parts.extend([
                AnswerTexts.PREV_FB(lang),
                AnswerTexts.PREV_ANS(lang, ans=previous_answer),
            ])
            if critique:
                parts.append(AnswerTexts.REJ_REASON(lang, critique=critique))
            if failed_rubrics:
                parts.append(AnswerTexts.FAIL_RUBRICS(lang, failed=failed_rubrics))
            if suggestion:
                parts.append(AnswerTexts.SUGGESTION(lang, suggestion=suggestion))

            parts.extend([
                AnswerTexts.PROV_NEW(lang),
                "=" * 50,
                "",
            ])

        if previous_retry_attempt_count or previous_retry_memories:
            previous_retry_memories = previous_retry_memories or []
            if not previous_retry_attempt_count and previous_retry_memories:
                previous_retry_attempt_count = max(
                    int(item.get("retry_attempt", 0)) for item in previous_retry_memories
                ) + 1

            parts.extend([
                AnswerTexts.RETRY_MEM_HEADER(lang),
                AnswerTexts.RETRY_MEM_INTRO(lang),
                AnswerTexts.RETRY_MEM_COUNT(lang, count=previous_retry_attempt_count),
                AnswerTexts.RETRY_MEM_ENCOURAGE(lang),
                AnswerTexts.RETRY_MEM_IDS_NOTE(lang),
            ])

            memory_groups: dict[int, list[dict]] = {}
            for item in previous_retry_memories:
                memory_groups.setdefault(int(item.get("retry_attempt", 0)), []).append(item)
            if memory_groups:
                for retry_attempt in sorted(memory_groups):
                    parts.append(AnswerTexts.RETRY_MEM_GROUP(lang, n=retry_attempt + 1))
                    for item in sorted(
                        memory_groups[retry_attempt],
                        key=lambda x: int(x.get("memory_order", 0)),
                    ):
                        parts.append(
                            f"- memory {item.get('memory_order')}: {item.get('memory_id')}"
                        )
            else:
                parts.append(AnswerTexts.RETRY_MEM_NONE(lang))

            parts.extend([
                AnswerTexts.RETRY_MEM_FOOTER(lang),
                "=" * 50,
                "",
            ])

        parts.append(AnswerTexts.Q(lang, question=question))

        if not skip_rubrics and rubrics:
            rubrics_str = "\n".join([f"{r.id}. {r.description}" for r in rubrics])
            parts.append(AnswerTexts.MUST_SAT(lang, rubrics=rubrics_str))
            parts.append(AnswerTexts.SEARCH_VERIFY(lang))
        else:
            parts.append(AnswerTexts.SEARCH_ANS(lang))

        prefix = "\n".join(parts)
        return f"{prefix}\n\n{carryover_text}" if carryover_text else prefix


@register_prompt_set("fsm_rubric_gen")
class FSMRubricGenPrompt(BasePromptBuilder):

    def build_system_prompt(self, prompt_language: str = "en") -> str:
        return RubricGenTexts.SYS(prompt_language)

    def build_user_prompt(self, question: str, prompt_language: str = "en"):
        return RubricGenTexts.USER(prompt_language, question=question)


@register_prompt_set("fsm_rubric_refine")
class FSMRubricRefinePrompt(BasePromptBuilder):

    def build_system_prompt(self, prompt_language: str = "en") -> str:
        return RubricRefineTexts.SYS(prompt_language)

    def build_user_prompt(
        self,
        question: str,
        current_rubrics: list[Rubric],
        suggestion: str = None,
        critique: str = None,
        prompt_language: str = "en",
    ):
        lang = prompt_language
        rubrics_str = "\n".join([f"{r.id}. {r.description}" for r in current_rubrics])
        prompt = RubricRefineTexts.USER(lang, question=question, rubrics_str=rubrics_str, critique=critique)

        if suggestion:
            prompt += RubricRefineTexts.SUG(lang, suggestion=suggestion)

        prompt += RubricRefineTexts.END(lang)
        return prompt


@register_prompt_set("fsm_verify")
class FSMVerifyPrompt(BasePromptBuilder):

    def build_system_prompt(
        self,
        tool_set: list[str],
        disable_rubric_reivision: bool = False,
        no_rubric_verify: bool = False,
        prompt_language: str = "en",
    ) -> str:
        lang = "zh" if _use_zh(prompt_language) else "en"
        tools = tool_registry.get_prompt(tool_set)
        if no_rubric_verify:
            return VerifyTexts.SYS_NO_RUBRIC(lang, tools=tools)
        rule = "" if disable_rubric_reivision else VerifyTexts.REV_RULE(lang)
        opt = "" if disable_rubric_reivision else VerifyTexts.REV_OPT(lang)
        return VerifyTexts.SYS(lang, tools=tools, revise_rule=rule, revise_opt=opt)

    def build_user_prompt(
        self,
        question: str,
        answer: str,
        carryover_text: str = None,
        rubrics: List["Rubric"] = None,
        reasoning: Optional[str] = None,
        disable_rubric_reivision: bool = False,
        no_rubric_verify: bool = False,
        prompt_language: str = "en",
    ):
        lang = "zh" if _use_zh(prompt_language) else "en"
        if no_rubric_verify:
            sections = [
                VerifyTexts.ORIG_Q(lang, question=question),
                VerifyTexts.REASONING(lang, reasoning=reasoning) if reasoning else None,
                VerifyTexts.ANS(lang, answer=answer),
                VerifyTexts.NO_RUBRIC_END(lang),
            ]
        else:
            if rubrics:
                rubrics_text = "\n".join(f"{r.id}. {r.description}" for r in rubrics)
                search_rule = VerifyTexts.SEARCH_ALL(lang)
                options = "PASS / REVISE_ANSWER" if disable_rubric_reivision else "PASS / REVISE_ANSWER / REVISE_RUBRIC"
            else:
                rubrics_text = VerifyTexts.NO_RUBRICS(lang)
                search_rule = VerifyTexts.SEARCH_FACT(lang)
                options = "PASS / REVISE_ANSWER"

            sections = [
                VerifyTexts.ORIG_Q(lang, question=question),
                VerifyTexts.RUBRICS(lang, rubrics=rubrics_text),
                VerifyTexts.REASONING(lang, reasoning=reasoning) if reasoning else None,
                VerifyTexts.ANS(lang, answer=answer),
                VerifyTexts.END(lang, search_rule=search_rule, options=options),
            ]

        prefix = "\n\n".join(filter(None, sections))
        return f"{prefix}\n\n{carryover_text}" if carryover_text else prefix
