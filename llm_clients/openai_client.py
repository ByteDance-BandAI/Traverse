# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from llm_clients.base import BaseLLMClient, get_first_env
from llm_clients.http import build_async_httpx_client


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        extra_body_overrides: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = endpoint
        self.endpoint = endpoint
        super().__init__(
            model_name=model_name,
            extra_body_overrides=extra_body_overrides,
            timeout=timeout,
        )

    def _init_client(self) -> None:
        self.api_key = (
            self.api_key
            or get_first_env("OPENAI_API_KEY", "LLM_API_KEY", default="")
        )
        self.base_url = (
            self.base_url
            or get_first_env(
                "OPENAI_BASE_URL",
                "LLM_BASE_URL",
                default="http://localhost:8000/v1",
            )
        )
        self.endpoint = self.base_url

        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key or "EMPTY",
            "base_url": self.base_url,
        }
        openai_timeout = self._client_timeout or int(
            os.getenv("OPENAI_CLIENT_TIMEOUT", "600")
        )
        http_client = build_async_httpx_client(timeout=openai_timeout)
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self.client = AsyncOpenAI(**client_kwargs)

        openai_limit = get_first_env("OPENAI_MAX_TOKENS", "LLM_MAX_TOKENS")
        if openai_limit and openai_limit.isdigit():
            self.max_tokens_limit = int(openai_limit)

    def _prepare_extra_body(
        self,
        *,
        extra_body: Dict[str, Any],
        include_thoughts: bool,
        chat_id: Optional[str],
    ) -> Dict[str, Any]:
        if chat_id and "session_id" not in extra_body:
            extra_body["session_id"] = abs(hash(chat_id))

        extra_body["chat_template_kwargs"] = {"enable_thinking": include_thoughts}

        return extra_body
