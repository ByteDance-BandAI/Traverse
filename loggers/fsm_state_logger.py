# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List

from agents.self_verify_agent import FSMAgentState
from loggers.segment_logger import SegmentFileLogger


class FSMStateLogger:
    """Swaps an inner SegmentFileLogger on every FSM state transition.

    File naming under question_dir/ (prefix drives the stem):
        rubric_t{tc}_attempt_{i}_{seg}.jsonl
        answer_t{tc}_attempt_{i}_{seg}.jsonl
        verify_t{tc}_attempt_{i}_{seg}.jsonl

    `i` is the retry attempt index inside the same FSM state transition:
    - i=0 for the first try
    - i=1 after one exception, etc.
    """

    def __init__(self, question_dir: Path) -> None:
        question_dir.mkdir(parents=True, exist_ok=True)
        self._question_dir = question_dir
        self._inner: SegmentFileLogger | None = None
        self._prefix: str | None = None
        self._attempt_idx: int = 0
        # A state can transition via a rule-only fast path without running its
        # BaseAgent (for example invalid Answer -> automatic Verify failure).
        # In that case, the context still contains the previous state's
        # messages. Do not mislabel those stale messages as this state's log.
        self._has_state_activity: bool = False
        self._entry_message_signature: tuple[int, int | None, int | None] = (
            0,
            None,
            None,
        )

    @staticmethod
    def _message_signature(
        messages: List[Dict[str, Any]],
    ) -> tuple[int, int | None, int | None]:
        if not messages:
            return (0, None, None)
        return (len(messages), id(messages[0]), id(messages[-1]))

    def _attempt_prefix(self) -> str:
        if self._prefix is None:
            raise RuntimeError("FSMStateLogger has no active state prefix.")
        return f"{self._prefix}_attempt_{self._attempt_idx}"

    async def transition(self, state: FSMAgentState, tc: int, messages: List[Dict[str, Any]]) -> None:
        """Close the current inner logger and open a fresh one for *state*."""
        if self._inner is not None:
            await self._inner.close(messages if self._has_state_activity else [])
        self._prefix = f"{state.name.lower()}_t{tc}"
        self._attempt_idx = 0
        self._inner = SegmentFileLogger(self._question_dir, prefix=self._attempt_prefix())
        self._has_state_activity = False
        self._entry_message_signature = self._message_signature(messages)

    async def seal(self, messages: List[Dict[str, Any]]) -> None:
        """Rotate to the next segment file (called on SealMemoryAgentState.SEAL)."""
        if self._inner is not None:
            await self._inner.rotate(messages)
            self._has_state_activity = True

    async def error(self, messages: List[Dict[str, Any]]) -> None:
        """Flush and close current attempt log, then reopen for next attempt."""
        if self._inner is None:
            return
        await self._inner.close(messages if self._has_state_activity else [])
        self._attempt_idx += 1
        self._inner = SegmentFileLogger(self._question_dir, prefix=self._attempt_prefix())
        self._has_state_activity = False
        self._entry_message_signature = self._message_signature(messages)

    async def write(self, messages: List[Dict[str, Any]]) -> None:
        if self._inner is not None:
            if (
                not self._has_state_activity
                and self._message_signature(messages) == self._entry_message_signature
            ):
                return
            await self._inner.write(messages)
            self._has_state_activity = True

    async def close(self, messages: List[Dict[str, Any]]) -> None:
        if self._inner is not None:
            await self._inner.close(messages if self._has_state_activity else [])
            self._inner = None
            self._has_state_activity = False
