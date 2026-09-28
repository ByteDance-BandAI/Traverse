# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from typing import IO, Any, Dict, List, Optional


class SegmentFileLogger:
    """Async segment file logger with SEAL rotation support."""

    def __init__(self, question_dir: Path, prefix: str = "") -> None:
        self._question_dir = question_dir
        self._prefix = prefix
        self._seg_idx: int = 0
        self._cursor: int = 0
        
        self._file: Optional[IO[str]] = None
        self._lock = asyncio.Lock()  # Prevents concurrent file operations on this instance

    def _seg_name(self, idx: int) -> str:
        stem = f"{self._prefix}_" if self._prefix else "segment_"
        return f"{stem}{idx}.jsonl"

    async def _ensure_open(self) -> IO[str]:
        """Lazily initialize the directory and first file to avoid blocking __init__."""
        if self._file is None or self._file.closed:
            def _open_initial():
                self._question_dir.mkdir(parents=True, exist_ok=True)
                return open(self._question_dir / self._seg_name(self._seg_idx), "w", encoding="utf-8")
            self._file = await asyncio.to_thread(_open_initial)
        return self._file

    async def write(self, messages: List[Dict[str, Any]]) -> None:
        """Flush messages[cursor:] to the current segment."""
        async with self._lock:
            file = await self._ensure_open()
            new_msgs = messages[self._cursor :]
            new_cursor = len(messages) 
            
            if new_msgs:
                await asyncio.to_thread(_write_lines, file, new_msgs, flush=True)
                # Only advance cursor AFTER successful write to prevent data loss
                self._cursor = new_cursor

    async def rotate(self, messages: List[Dict[str, Any]]) -> None:
        """Flush remaining messages, close the current segment, open segment_{n+1}.jsonl."""
        async with self._lock:
            file = await self._ensure_open()
            tail_msgs = messages[self._cursor :]

            self._seg_idx += 1
            new_path = self._question_dir / self._seg_name(self._seg_idx)
            
            # _rotate_sync must handle writing tail_msgs, closing file, and returning new file
            self._file = await asyncio.to_thread(_rotate_sync, file, tail_msgs, new_path)
            
            self._cursor = 0

    async def close(self, messages: List[Dict[str, Any]]) -> None:
        """Flush remaining messages and close the file (idempotent)."""
        async with self._lock:
            if self._file is not None and self._file.closed:
                return
                
            tail_msgs = messages[self._cursor :]
            new_cursor = len(messages)

            if self._file is None:
                if not tail_msgs:
                    return
                self._file = await self._ensure_open()

            await asyncio.to_thread(_close_sync, self._file, tail_msgs)
            self._cursor = new_cursor

    # ------------------------------------------------------------------
    # Sync helpers — executed inside asyncio.to_thread worker threads
    # ------------------------------------------------------------------


def _write_lines(file: IO[str], msgs: List[Dict[str, Any]], flush: bool = False) -> None:
    for msg in msgs:
        file.write(json.dumps(msg, ensure_ascii=False) + "\n")
    if flush:
        file.flush()


def _rotate_sync(
    old_file: IO[str], tail_msgs: List[Dict[str, Any]], new_path: Path
) -> IO[str]:
    _write_lines(old_file, tail_msgs)
    old_file.close()
    return open(new_path, "w", encoding="utf-8")


def _close_sync(file: IO[str], tail_msgs: List[Dict[str, Any]]) -> None:
    _write_lines(file, tail_msgs)
    file.close()
