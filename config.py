# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""General configs of search agent data composition"""

from dataclasses import dataclass
from llm_clients.base import BaseLLMClient
from typing import Callable, Optional


@dataclass
class GeneralLLMConfig:
    llm_client: BaseLLMClient
    temperature: float
    top_p: float
    max_tokens: int
    include_thoughts: bool
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None

@dataclass
class RuntimeConfig:
    main_llm: GeneralLLMConfig = None
    judge_llm: GeneralLLMConfig = None
    summary_llm: GeneralLLMConfig = None  # external model for the seal sub-run
    completion_cap: int = 0
    work_dir: str = ""
    add_turn_budget: bool = False
    add_token_budget: bool = False
    add_seal_budget: bool = False
    enable_budget_prompt: bool = False
    prompt_language: str = "en"
    token_estimator: Callable = None
    rubric_completion_cap: int | None = None
    # Answer repeated tool calls locally instead of dispatching them, so a model
    # looping over the same searches is told to break out (see
    # agents/tool_loop_guard.py). Off keeps every call going to the tool layer.
    tool_loop_guard: bool = False
    tool_loop_guard_max_cycle: int = 3
