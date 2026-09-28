# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from constants import Message
from constants import MessageRole
import json
import re


def _strip_answer_artifacts(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", answer)
    answer = re.sub(r"\s*```\s*$", "", answer)
    return answer.strip()


def _extract_markdown_table(content: str) -> str | None:
    """Return a Markdown table from a final response, preserving its fence.

    WideSearch prompts require a fenced Markdown table, while the shared FSM
    transport requires an ``Answer:`` envelope. Models sometimes satisfy both
    instructions in the wrong order and append ``Answer: See table above``
    after an otherwise valid table. In that benchmark-specific mode, prefer the
    actual table over the low-information pointer.
    """

    fenced_tables = re.findall(
        r"```markdown\s*(.*?)\s*```",
        content or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for candidate in reversed(fenced_tables):
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        pipe_lines = [
            line
            for line in lines
            if line.startswith("|") and line.endswith("|") and line.count("|") >= 3
        ]
        separator_found = any(
            set(line).issubset(set("|- :")) and "-" in line
            for line in pipe_lines
        )
        if len(pipe_lines) >= 3 and separator_found:
            return f"```markdown\n{candidate.strip()}\n```"

    # Accept an unfenced pipe table as a fallback. Keep only the contiguous
    # table block so reasoning text cannot leak into the benchmark answer.
    lines = (content or "").splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 3
        ):
            current.append(stripped)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    for block in reversed(blocks):
        separator_found = any(
            set(line).issubset(set("|- :")) and "-" in line
            for line in block
        )
        if len(block) >= 3 and separator_found:
            return "\n".join(block)
    return None


def submit_result_parser(messages: list[Message]) -> dict | None:
    """Return the full SubmitTool result JSON, or ``None``.

    When the agent finishes by calling a SubmitTool, the trajectory ends with the
    assistant's submit tool call (``messages[-2]``) followed by the tool result
    (``messages[-1]``) whose content is the JSON returned by ``SubmitTool.run()``
    (e.g. ``answer``, ``reason``, ``evidences``).

    Returns the decoded payload dict, or ``None`` if the run did not end with a
    successful SubmitTool call.
    """
    # Local import keeps verifiers importable without triggering the tools
    # package to load at module import time.
    from tools import tool_registry

    if not messages or len(messages) < 2:
        return None

    tool_msg, assistant_msg = messages[-1], messages[-2]
    if tool_msg.role is not MessageRole.TOOL or not tool_msg.tool_call_succeed:
        return None
    if assistant_msg.role is not MessageRole.ASSISTANT or not assistant_msg.tool_calls:
        return None
    if not tool_registry.is_submit_tool_call(assistant_msg.tool_calls[0]):
        return None

    content = tool_msg.content or ""
    for candidate in (content, content.split("\n", 1)[0]):
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def submit_answer_parser(messages: list[Message]) -> str | None:
    """Read the final answer from the SubmitTool result.

    Returns the ``answer`` string, or ``None`` if the run did not end with a
    successful SubmitTool call.
    """
    payload = submit_result_parser(messages)
    if payload is None:
        return None
    answer = str(payload.get("answer", "")).strip()
    return answer or None


def _context_tool_names(context) -> list[str]:
    """Tool names available to the agent for this run.

    Prefers the explicit allowed set; falls back to the tool schemas actually
    sent to the LLM (``context.tools``) for recipes that don't populate
    ``allowed_tool_names``.
    """
    if getattr(context, "allowed_tool_names", None):
        return list(context.allowed_tool_names)
    names: list[str] = []
    for schema in getattr(context, "tools", None) or []:
        if not isinstance(schema, dict):
            continue
        fn = schema.get("function", schema)
        name = fn.get("name") if isinstance(fn, dict) else None
        if name:
            names.append(name)
    return names


def submit_tool_enabled(context) -> bool:
    """True when a SubmitTool is part of the agent's allowed tool set."""
    from tools import tool_registry

    return tool_registry.has_submit_tool(_context_tool_names(context))


def parse_answer(context) -> str | None:
    """Extract the final answer for a completed run.

    Dispatches on whether a SubmitTool is enabled: when it is, read the
    submitted answer from the tool result; otherwise fall back to the legacy
    text parser over the last assistant message.
    """
    messages = context.trajectory.messages
    if not messages:
        return None
    if submit_tool_enabled(context):
        return submit_answer_parser(messages)
    return answer_parser(messages[-1])


def answer_parser(
    msg: Message,
    strict: bool = False,
    *,
    prefer_markdown_table: bool = False,
):
    if not isinstance(msg, Message):
        raise ValueError("Answer parser requires Message class!")
    if not msg.role == MessageRole.ASSISTANT:
        return None
    # 1. Extract content and cleanly strip thinking tags
    content = (msg.content or "").strip()
    if "</think>" in content:
        after_think = content.split('</think>')[-1].strip()
        # Strict mode is used by retry gates, so hidden reasoning must not count
        # as a parseable final answer.
        if strict and not after_think:
            return None
        # If nothing follows </think>, fall back to the text inside the think block
        # only for legacy non-strict extraction.
        content = after_think if after_think else content.split('</think>')[0].strip()
    if "<answer>" in content:
        content = content.split("<answer>")[-1].strip().split("</answer>")[0].strip()
    reasoning = msg.reasoning_content or ""
    reasoning = reasoning.strip()
    
    # Fallback to reasoning_content if main content is empty. Strict mode is used
    # for retry decisions and should only accept visible final-answer content.
    if not content:
        if strict or not reasoning:
            return None
        content = reasoning

    if prefer_markdown_table:
        markdown_table = _extract_markdown_table(content)
        if markdown_table:
            return markdown_table

    # 2. Primary Extraction: Everything after "Answer:" or "**Answer:**"
    match = re.search(r'\*?\*?Answer:\*?\*?\s*(.*)', content, flags=re.DOTALL)
    if match:
        ans = _strip_answer_artifacts(match.group(1))
        if ans:
            if (
                prefer_markdown_table
                and re.fullmatch(
                    r"(?is)(?:please\s+)?(?:see|refer\s+to)\s+"
                    r"(?:the\s+)?(?:markdown\s+)?table\s+(?:shown\s+)?above[.!]?",
                    ans,
                )
            ):
                return None
            return ans

    # 3. Secondary Extraction: Single-line fallback patterns
    fallback_pattern = r'(?i)(?:答案[是为:]|answer:|the answer is)\s*(.+)'
    match = re.search(fallback_pattern, content)
    if match:
        ans = match.group(1).splitlines()[0].strip()
        ans = ans.split('。')[0].strip()             # Truncate at Chinese period
        ans = re.split(r'\.\s+(?=\S)', ans)[0]        # Truncate at English period ending a sentence
        ans = _strip_answer_artifacts(ans)
        if ans:
            return ans

    # Final fallback, return the original content
    if not strict:
        return _strip_answer_artifacts(content)
    else:
        return None
