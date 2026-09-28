# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any

try:
    from openai import DefaultAsyncHttpxClient  # type: ignore
except Exception:  # pragma: no cover
    DefaultAsyncHttpxClient = None  # type: ignore


def build_async_httpx_client(*, timeout: int) -> Any:
    """Build a shared async httpx client for OpenAI-compatible SDK clients."""
    if DefaultAsyncHttpxClient is None:
        return None
    return DefaultAsyncHttpxClient(trust_env=False, timeout=timeout)
