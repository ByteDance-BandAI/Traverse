# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto
from pydantic import BaseModel, ConfigDict, Field, model_validator
from llm_clients.base import BaseLLMClient
from typing import Any, Optional, Callable
import json
from dataclasses import dataclass, field
from config import RuntimeConfig
from pathlib import Path
import os


# Per-rollout on-disk layout.
# Question dirs sit directly under the rollout dir rather than in a nested
# subdir: every run produced so far uses that flat shape, and the offline
# analysis scripts read those runs. Route new path construction through these
# helpers so the layout stays changeable from one place.
REVIEW_FILENAME = "review.jsonl"


def trajectories_dir(rollout_dir: "str | Path") -> Path:
    """Return the directory that holds all per-question trajectory dirs."""
    return Path(rollout_dir)


def question_dir(rollout_dir: "str | Path", question_id: str) -> Path:
    """Return the per-question trajectory dir for a given rollout."""
    return trajectories_dir(rollout_dir) / f"question_{question_id}"


def review_log_path(rollout_dir: "str | Path") -> Path:
    """Return the review log path for a given rollout dir."""
    return Path(rollout_dir) / REVIEW_FILENAME


class AgentState(Enum):
    INITIAL = auto()
    TOOL = auto()
    API_CALL = auto()
    END = auto()

class EndReason(Enum):
    TURN_LIMIT_REACHED = auto()
    AGENT_EXIT = auto()
    ERROR = auto()
    CONTEXT_EXCEED = auto()
    MEMORY_TOOL_CALLED = auto()
    OVER_TURN_RETRY = auto()
    OVER_SEAL_RETRY = auto()

class MessageRole(Enum):
    USER = auto()
    ASSISTANT = auto()
    SYSTEM = auto()
    TOOL = auto()

class ToolCall(BaseModel):
    id: str
    name: str
    # Normally a parsed JSON object. May fall through as the raw API string
    # when the model emits arguments that fail JSON decoding; the tool layer
    # detects this case and returns a structured error.
    arguments: dict[str, Any] | str
    metadata: dict[str, Any]

    def to_api_dict(self) -> dict[str, Any]:
        """Shape expected by OpenAI-compatible chat APIs for assistant tool_calls."""
        args = self.arguments
        if isinstance(args, str):
            arguments_str = args
        else:
            arguments_str = json.dumps(args, ensure_ascii=False)
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": arguments_str},
        }


class LLMBackend(Enum):
    SGLANG = auto()
    VLLM = auto()


class Message(BaseModel):
    role: MessageRole
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    tool_call_succeed: bool = None

    @model_validator(mode="after")
    def validate_role_specific_fields(self) -> "Message":
        # Keep assistant-only metadata out of non-assistant messages.
        if self.role is not MessageRole.ASSISTANT:
            if self.reasoning_content is not None:
                raise ValueError("reasoning is only allowed for assistant messages")
            if self.tool_calls is not None:
                raise ValueError("tool_calls is only allowed for assistant messages")
        
        return self


class AgentTrajectory(BaseModel):
    messages: list[Message]
    serialized_messages: list[dict[str, Any]] = Field(default_factory=list)
    force_reasoning: bool = False
    backend_parser: LLMBackend | None = None
    pending_tool_calls: list[ToolCall] = Field(default_factory=list)
    # Recent tool call signatures, used to spot a model looping over the same
    # calls. Lives here so the window resets together with the conversation the
    # model can actually see (see clear()).
    recent_tool_signatures: list[str] = Field(default_factory=list)
    turn: int = 0
    num_tokens_api: list[int] = Field(default_factory=list)
    end_reason: EndReason | None = None
    chat_id: int = Field(default_factory=lambda: int.from_bytes(os.urandom(4), "big"))

    def _serialize_message(self, msg: Message) -> dict[str, Any]:
        row: dict[str, Any] = {
            "role": msg.role.name.lower(),
            "content": msg.content,
        }
        if msg.role is MessageRole.ASSISTANT and (self.force_reasoning or msg.reasoning_content is not None):
            if self.backend_parser == LLMBackend.VLLM:
                row["reasoning"] = msg.reasoning_content or ""
            else:
                row["reasoning_content"] = msg.reasoning_content or ""
        if msg.tool_calls:
            row["tool_calls"] = [tc.to_api_dict() for tc in msg.tool_calls]
        if msg.role is MessageRole.TOOL and msg.tool_call_id:
            row["tool_call_id"] = msg.tool_call_id
        return row

    @model_validator(mode="after")
    def init_serialized_messages(self) -> "AgentTrajectory":
        # Keep a cached serialization in sync for faster repeated API calls.
        self.serialized_messages = [self._serialize_message(msg) for msg in self.messages]
        return self

    def add_messages(self, *messages: Message) -> None:
        for message in messages:
            self.messages.append(message)
            self.serialized_messages.append(self._serialize_message(message))

    def clear(self) -> None:
        self.messages.clear()
        self.serialized_messages.clear()
        self.pending_tool_calls.clear()
        self.recent_tool_signatures.clear()
        self.end_reason = None
        # When trajectory changed, assign a new chat_id. Backends that key a
        # server-side session cache on it will then start a fresh session.
        self.chat_id = int.from_bytes(os.urandom(4), "big")
        self.num_tokens_api.append(0)

    def record_tokens_api(self, num_tokens: int):
        if not self.num_tokens_api:
            self.num_tokens_api.append(0)
        self.num_tokens_api[-1] = num_tokens

    def get_total_tokens(self) -> int:
        return sum(self.num_tokens_api)

    @property
    def current_num_tokens_api(self) -> int:
        # Empty before the first API call; readers run earlier than that.
        return self.num_tokens_api[-1] if self.num_tokens_api else 0

@dataclass(slots=True)
class ToolCallResult:
    ok: bool
    output: str = None
    error: Exception | None = None
    extra_info: str | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptPayload:
    system: dict[str, Any] = field(default_factory=dict)
    user: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentFrame:
    agent: str
    state: str


@dataclass
class RunContext:
    """Encapsulates all state and configuration for a single agent execution."""
    runtime_config: RuntimeConfig
    trajectory: AgentTrajectory = field(default_factory=lambda: AgentTrajectory(messages=[]))
    
    # Core LLM parameters
    question_id: Optional[str] = None
    tools: list[Any] = field(default_factory=list)
    allowed_tool_names: list[str] = field(default_factory=list)
    max_retry_limit: int = 7
    max_turns: int = 1024
    
    # Extensible state for custom handlers or prompts
    prompt_set="seal"
    prompt_payload: PromptPayload = field(default_factory=PromptPayload)
    state_stack: list[AgentFrame] = field(default_factory=list)
    # Metadata of the most recent LLM API call (updated every API turn).
    last_api_finish_reason: str | None = None
    last_api_prompt_tokens: int | None = None
    last_api_completion_tokens: int | None = None
    last_api_reasoning_tokens: int | None = None
    # Tool calls short-circuited by the repeated-call guard. Cumulative over the
    # whole rollout, so it survives the per-segment trajectory resets.
    tool_loop_blocked_count: int = 0

    rollout_dir: "Path | None" = None # set per-rollout; used by memory tools for path resolution
    rollout_idx: int = 0 # set per-rollout; tracking unit for the metric logger

    def push_state(self, agent: str, state: Any) -> None:
        self.state_stack.append(AgentFrame(agent=agent, state=state.name if hasattr(state, "name") else str(state)))

    def pop_state(self, agent: str, state: Any) -> None:
        expected = AgentFrame(agent=agent, state=state.name if hasattr(state, "name") else str(state))
        if not self.state_stack:
            raise RuntimeError(f"Attempted to pop empty state stack for {expected}")
        frame = self.state_stack[-1]
        if frame == expected:
            self.state_stack.pop()
            return

        # Async generators can be closed out of order when an outer state exits
        # early while nested agent streams are suspended at a yield. Leave newer
        # frames for their own finalizers and remove the expected frame in place.
        for idx in range(len(self.state_stack) - 2, -1, -1):
            if self.state_stack[idx] == expected:
                del self.state_stack[idx]
                return

        raise RuntimeError(f"State stack mismatch: expected {expected}, got {frame}")
