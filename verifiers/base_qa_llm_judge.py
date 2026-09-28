# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from llm_clients.base import BaseLLMClient, LLMCallResult
from prompt.base import PromptFactory
from verifiers import BaseVerifier, JudgeResult, register_verifier
from config import GeneralLLMConfig


_RATE_LIMIT_KEYWORDS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "qpm limit",
)


class BaseQALLMJudge(BaseVerifier):
    """LLM-based judge that evaluates whether *answer* correctly addresses
    *question* given one or more *ground_truth* reference answers.

    Parameters
    ----------
    prompt_set:
        Name of a prompt set registered via ``@register_prompt_set``.
        The builder must implement at least ``build_system_prompt`` and
        ``build_user_prompt``; both receive ``question``, ``ground_truth``,
        and ``answer`` as keyword arguments.
    client_config:
        OpenAI-compatible client and sampling settings used by the judge.
    max_retry_times:
        Maximum number of judge retries after a malformed response.
    """

    name = "question_answer_llm_judge"

    def __init__(
        self,
        prompt_set: str,
        client_config: GeneralLLMConfig,
        max_retry_times: int = 3,
        retry_infinitely_on_429: bool = False,
    ) -> None:
        self.prompt_factory = PromptFactory(prompt_set)
        self.client_config = client_config

        self.max_retry_times = max_retry_times
        self.retry_infinitely_on_429 = retry_infinitely_on_429

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        if any(k in msg for k in _RATE_LIMIT_KEYWORDS):
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        if status == 429:
            return True
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 429:
            return True
        return False

    def _build_messages(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        system_prompt = self.prompt_factory.get_prompt(
            role="system"
        )
        user_prompt_kwargs = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
        }
        if metadata is not None:
            user_prompt_kwargs["metadata"] = metadata
        user_prompt = self.prompt_factory.get_prompt(
            role="user",
            **user_prompt_kwargs,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_response(self, raw: str) -> JudgeResult:
        raise NotImplementedError("Subclass should implement the parsing logic!")

    def should_retry_llm_call(
        self,
        result: LLMCallResult | None = None,
        exc: Exception | None = None,
    ) -> bool:
        """Return ``True`` if the LLM call should be retried.

        Called in two situations:
        - After a successful call: ``result`` is populated, ``exc`` is ``None``.
        - After an exception: ``exc`` is populated, ``result`` is ``None``.

        Subclasses override this to inspect the outcome and decide whether a
        retry is warranted (e.g. empty content, unexpected finish reason,
        transient network error, etc.).  The base implementation always returns
        ``False`` (no retry).
        """
        return False

    async def async_judge(
        self,
        question: str,
        ground_truth: list[str],
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeResult:
        """Async variant of :meth:`judge` for use inside a running event loop."""
        messages = self._build_messages(
            question,
            ground_truth,
            answer,
            metadata=metadata,
        )
        last_content: str | None = None
        # Counts only non-429 retries (or all retries when infinite-429 is off).
        attempts = 0
        while True:
            try:
                llm_result: LLMCallResult = await self.client_config.llm_client.call_llm(
                    messages=messages,
                    max_tokens=self.client_config.max_tokens,
                    temperature=self.client_config.temperature,
                    include_thoughts=self.client_config.include_thoughts,
                    top_p=self.client_config.top_p,
                    top_k=self.client_config.top_k,
                    min_p=self.client_config.min_p,
                    presence_penalty=self.client_config.presence_penalty,
                    repetition_penalty=self.client_config.repetition_penalty,
                )
                last_content = llm_result.content or ""
                if self.should_retry_llm_call(result=llm_result):
                    attempts += 1
                    if attempts >= self.max_retry_times:
                        break
                    continue
                return self._parse_response(last_content)
            except Exception as exc:
                if self.retry_infinitely_on_429 and self._is_rate_limit_error(exc):
                    await asyncio.sleep(3)
                    continue
                if self.should_retry_llm_call(exc=exc):
                    attempts += 1
                    if attempts >= self.max_retry_times:
                        # An earlier attempt may have returned parseable content;
                        # fall back to it rather than losing the whole judgement.
                        if last_content is not None:
                            break
                        raise
                    continue
                raise

        if last_content is not None:
            return self._parse_response(last_content)

        raise RuntimeError("LLM judge exhausted retries without a parseable response.")
