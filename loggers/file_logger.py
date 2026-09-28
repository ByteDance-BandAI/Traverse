# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from typing import Any, Dict, Optional


class JsonlFileLogger:
    def __init__(self, file_path: str, lock: Optional[asyncio.Lock] = None):
        self.file_path = file_path
        self._lock = lock or asyncio.Lock()
        self._closed = False

        parent = os.path.dirname(os.path.abspath(file_path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Open once at startup so the file is created immediately.
        self._file = open(file_path, "a", encoding="utf-8", buffering=1)

    def _prepare_line(self, payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _write_and_flush(self, line: str) -> None:
        self._file.write(f"{line}\n")
        self._file.flush()

    def _close_file(self) -> None:
        self._file.close()

    def _remove_last_line_and_reopen(self) -> bool:
        self._file.flush()
        self._file.close()

        try:
            with open(self.file_path, "r", encoding="utf-8") as rf:
                lines = rf.readlines()
        except FileNotFoundError:
            lines = []

        removed = bool(lines)
        remaining = lines[:-1] if removed else []

        with open(self.file_path, "w", encoding="utf-8") as wf:
            wf.writelines(remaining)
            wf.flush()

        self._file = open(self.file_path, "a", encoding="utf-8", buffering=1)
        return removed

    def _pop_latest_log_by_question_id_and_reopen(
        self,
        question_id: str,
    ) -> Optional[Dict[str, Any]]:
        self._file.flush()
        self._file.close()

        try:
            with open(self.file_path, "r", encoding="utf-8") as rf:
                lines = rf.readlines()
        except FileNotFoundError:
            lines = []

        remove_idx = -1
        removed_payload: Optional[Dict[str, Any]] = None
        for idx in range(len(lines) - 1, -1, -1):
            raw = lines[idx].strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if payload.get("question_id") == question_id:
                remove_idx = idx
                removed_payload = payload
                break

        removed = remove_idx >= 0
        if removed:
            del lines[remove_idx]

        with open(self.file_path, "w", encoding="utf-8") as wf:
            wf.writelines(lines)
            wf.flush()

        self._file = open(self.file_path, "a", encoding="utf-8", buffering=1)
        return removed_payload

    def _remove_latest_log_by_question_id_and_reopen(self, question_id: str) -> bool:
        return self._pop_latest_log_by_question_id_and_reopen(question_id) is not None

    async def write_log(self, payload: Dict[str, Any]) -> None:
        line = self._prepare_line(payload)
        async with self._lock:
            if self._closed:
                raise RuntimeError("logger is closed")
            await asyncio.to_thread(self._write_and_flush, line)

    async def remove_last_log(self) -> bool:
        async with self._lock:
            if self._closed:
                raise RuntimeError("logger is closed")
            return await asyncio.to_thread(self._remove_last_line_and_reopen)

    async def remove_log_by_question_id(self, question_id: str) -> bool:
        async with self._lock:
            if self._closed:
                raise RuntimeError("logger is closed")
            return await asyncio.to_thread(
                self._remove_latest_log_by_question_id_and_reopen,
                question_id,
            )

    async def pop_log_by_question_id(self, question_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if self._closed:
                raise RuntimeError("logger is closed")
            return await asyncio.to_thread(
                self._pop_latest_log_by_question_id_and_reopen,
                question_id,
            )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await asyncio.to_thread(self._close_file)
            self._closed = True

    async def __aenter__(self) -> "JsonlFileLogger":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
