# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from constants import ToolCall

# Only a back-to-back cycle counts, so the window never needs to be long. The
# cap keeps the history from growing with every turn of a long trajectory.
SIGNATURE_HISTORY_LIMIT = 32
MAX_CYCLE_LENGTH_CAP = 3


def _required_repeats(cycle_length: int) -> int:
    """How many back-to-back rounds of a cycle make it a loop."""
    # Reissuing a single call once more is a common reaction to a truncated or
    # flaky result, so only the third identical call counts. Alternating cycles
    # have no such benign reading and are conclusive after two rounds.
    return 3 if cycle_length == 1 else 2


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        # Casing and whitespace do not change what a search or fetch returns,
        # so they must not let an otherwise identical call through.
        return " ".join(value.split()).casefold()
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def tool_call_signature(tool_call: ToolCall) -> str:
    """Identify a call by tool name plus its normalized arguments."""
    arguments = tool_call.arguments
    if isinstance(arguments, str):
        # Arguments the model emitted as undecodable JSON; compare them as text,
        # normalized the same way _normalize treats any other string.
        rendered = " ".join(arguments.split()).casefold()
    else:
        rendered = json.dumps(
            _normalize(arguments), ensure_ascii=False, sort_keys=True
        )
    return f"{tool_call.name}\x00{rendered}"


def detect_repeat_cycle(
    signatures: list[str], max_cycle_length: int = MAX_CYCLE_LENGTH_CAP
) -> int | None:
    """Return the length of the cycle `signatures` ends on, or None.

    A cycle of length `n` is reported once the last `n * _required_repeats(n)`
    signatures form that many identical consecutive blocks. The shortest
    matching cycle wins, so a warning names the tightest loop it can.
    """
    limit = min(max(max_cycle_length, 0), MAX_CYCLE_LENGTH_CAP)
    for cycle_length in range(1, limit + 1):
        window = cycle_length * _required_repeats(cycle_length)
        if len(signatures) < window:
            continue
        tail = signatures[-window:]
        block = tail[:cycle_length]
        if all(
            tail[start:start + cycle_length] == block
            for start in range(cycle_length, window, cycle_length)
        ):
            return cycle_length
    return None


def record_signature(signatures: list[str], tool_call: ToolCall) -> None:
    """Append `tool_call`'s signature in place, keeping only the recent tail."""
    signatures.append(tool_call_signature(tool_call))
    del signatures[:-SIGNATURE_HISTORY_LIMIT]


def repeated_call_notice(cycle_length: int, prompt_language: str = "en") -> str:
    """The tool output a looping model receives in place of real results."""
    repeats = _required_repeats(cycle_length)
    if prompt_language == "zh":
        pattern = (
            f"你已经连续 {repeats} 轮发起了完全相同的调用"
            if cycle_length == 1
            else f"你在同样的 {cycle_length} 个调用之间来回循环，已经重复了 {repeats} 轮"
        )
        return (
            f"本次调用已被拦截，并未真正执行：{pattern}，再执行一次也只会返回你已经拿到的结果。\n"
            "你正在原地打转。请不要再重复这个调用，改为：\n"
            "1. 回看上文已有的返回结果，答案可能已经在里面；\n"
            "2. 如果确实还需要新信息，请换一条路：换关键词、换语言、换检索角度，"
            "或者换用别的来源和工具；\n"
            "3. 如果线索确实已经穷尽，就立即停止检索，基于现有证据给出最有把握的答案。"
        )
    pattern = (
        f"you have issued this identical call {repeats} turns in a row"
        if cycle_length == 1
        else f"you are cycling through the same {cycle_length} calls, now for {repeats} rounds"
    )
    return (
        f"This call was blocked and never executed: {pattern}, so running it again "
        "would only return results you already have.\n"
        "You are going in circles. Do not repeat this call. Instead:\n"
        "1. Re-read the results already in this conversation; the answer may be there.\n"
        "2. If you still need new information, change course: different keywords, a "
        "different language, a different angle, or a different source or tool.\n"
        "3. If the leads are genuinely exhausted, stop searching and give your "
        "best-supported answer now."
    )
