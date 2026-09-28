# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import uuid
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from agents.seal_memory_agent import SealMemoryContext
from constants import question_dir
from tools.base import MemoryTool, ReturnableToolError, ToolCall, ToolSchema, BaseTool
from tools import tool_registry


VALID_TOKEN_ESTIMATORS = {"api_response", "naive"}
VALID_STATUSES = {"verified", "conflicting", "partial"}

def _normalize_kg_entry(item: Any) -> Dict[str, str]:
    """Normalize a single knowledge graph entry into a standard dictionary."""
    if not isinstance(item, dict):
        return {
            "fact": str(item),
            "source_url": "",
            "status": "partial"
        }

    # Extract with fallbacks
    fact = item.get("fact") or item.get("summary") or item.get("value") or ""
    source = item.get("source_url") or item.get("source") or item.get("url") or ""
    status = item.get("status")

    return {
        "fact": str(fact),
        "source_url": str(source),
        "status": status if status in VALID_STATUSES else "partial"
    }

def _normalize_knowledge_graph(entries: Optional[List[Any]]) -> List[Dict[str, str]]:
    """Normalize a list of knowledge graph entries."""
    return [_normalize_kg_entry(item) for item in (entries or [])]

def _normalize_navigation_state(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize navigation state to ensure consistent keys and types."""
    state = state or {}

    # Handle visited summary
    visited = state.get("visited_summary") or state.get("visited") or ""

    # Handle frontier queue
    frontier_queue = []
    for item in state.get("frontier_queue", []):
        if isinstance(item, dict):
            target = str(item.get("target") or item.get("url") or item.get("path") or "")
            reason = str(item.get("reason") or item.get("why") or "")
        else:
            target, reason = str(item), ""
        frontier_queue.append({"target": target, "reason": reason})

    return {
        "visited_summary": str(visited),
        "frontier_queue": frontier_queue,
        "dead_ends": [str(d) for d in state.get("dead_ends", [])]
    }

def build_seal_memory_tool_response(
    knowledge_graph: Optional[List[Dict[str, Any]]] = None,
    navigation_state: Optional[Dict[str, Any]] = None,
    meta_learnings: Optional[List[str]] = None,
    next_step_plan: Optional[str] = None,
    task_progress: Optional[str] = None,
    stage: Optional[str] = None,
    tags: Optional[List[str]] = None,
    comment: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    structured_summary: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a "cognitive checkpoint": compresses the current conversation into a structured state,
    facilitating quick recovery in a new context.

    It is recommended to seal the memory when a sufficient amount of information is gathered
    or when the context becomes long; do not wait until the final answer to call this.
    It can be called multiple times.

    Newly added fields include knowledge graph, navigation state, meta-learnings, next step plan,
    and task progress; these fields can be partial or empty.
    `conversation_history` can be automatically populated by the caller, without the need
    for the model to repeatedly transmit it.
    """
    memory_id = str(uuid.uuid4())
    now = datetime.datetime.now(datetime.timezone.utc)

    # Normalize inputs
    kg = _normalize_knowledge_graph(knowledge_graph)
    nav_state = _normalize_navigation_state(navigation_state)
    meta = [str(m) for m in (meta_learnings or [])]
    plan = next_step_plan or ""
    progress = task_progress or ""

    # Flat shape is shared with `read_memory_tool` (see its field list): saving
    # this dict lets the reader round-trip every field it knows about, and
    # `_build_seal_summary_text` consumes a strict subset of the same keys.
    return {
        "memory_id": memory_id,
        "timestamp": now.isoformat() + "Z",
        "stage": stage,
        "tags": tags or [],
        "comment": comment,
        "knowledge_graph": kg,
        "navigation_state": nav_state,
        "meta_learnings": meta,
        "next_step_plan": plan,
        "task_progress": progress,
        "conversation_history": conversation_history or [],
        "structured_summary": structured_summary,
    }

@tool_registry.register_tool(name="seal_memory_tool")
class SealMemoryTool(MemoryTool):
    def __init__(
        self,
        max_seal_times: int = 5,
        min_seal_tokens: int = 20000,
    ) -> None:
        custom_tool_prompt = (
            "Mid-task cognitive checkpoint that compresses the current context into a structured memory. When calling, populate:\n"
            "   - knowledge_graph: concise facts with status (verified/conflicting/partial) and source_url\n"
            "   - navigation_state: visited_summary, frontier_queue (unvisited leads + reasons), dead_ends (avoid)\n"
            "   - meta_learnings: procedural/site heuristics; task_progress: short progress note; next_step_plan: first action in the new window"
        )
        super().__init__(name="seal_memory_tool", custom_tool_prompt=custom_tool_prompt)

        self.min_seal_tokens = min_seal_tokens
        self.max_seal_times = max_seal_times

        # 1. Define the inner schema for items in the knowledge_graph
        kg_item_schema = (
            ToolSchema()
            .add_string("fact", "The core information (e.g., 'iPhone 15 Price: $799').", required=True)
            .add_string("source_url", "The URL where this fact was found (for verification).", required=False)
            .add_string("status", "Reliability of this fact.", required=True, enum=["verified", "conflicting", "partial"])
        )

        # 2. Define the inner schema for items in the frontier_queue
        frontier_item_schema = (
            ToolSchema()
            .add_string("target", "URL or Element Name.", required=False) # Following your JSON, neither is explicitly required
            .add_string("reason", "Why we should visit this (e.g., 'Check specs', 'Compare price').", required=False)
        )

        # 3. Define the nested navigation_state object
        navigation_state_schema = (
            ToolSchema()
            .add_string("visited_summary", "A concise narrative of the path taken so far (e.g., 'Home -> Search Results -> Product A').", required=True)
            .add_array_of_objects("frontier_queue", "Promising URLs or Buttons discovered but NOT yet visited (BFS/DFS queue).", frontier_item_schema, required=True)
            .add_array("dead_ends", "List of URLs/Paths that are irrelevant, 404, or login-walled. DO NOT VISIT AGAIN.", required=True)
        )

        # 4. Assemble the final Master Tool
        seal_memory_tool = (
            ToolSchema(
                name="seal_memory_tool",
                description="Mid-task cognitive checkpoint to compress current context into a reusable state; call as soon as you have enough info or the context is getting long (do not wait for the final answer). Can be called multiple times."
            )
            .add_array_of_objects("knowledge_graph", "Optional, partial OK. Structured facts extracted so far.", kg_item_schema, required=False)
            .add_object("navigation_state", "Optional. The topological map of the browsing session to prevent loops and lost-in-navigation.", navigation_state_schema, required=False)
            .add_array("meta_learnings", "Optional. Procedural rules or heuristics learned so far.", required=False)
            .add_string("next_step_plan", "The immediate action plan for the very first step in the NEW context window.", required=True)
            .add_string("task_progress", "Current progress towards the final user goal (e.g., 'Found 2/3 items, need to compare prices').", required=True)
            .add_string("stage", "Optional marker for current stage/intent (e.g., 'initial_search', 'mid_task_checkpoint').", required=False)
            .add_array("tags", "Custom tags for easier retrieval later.", required=False)
            .add_string("comment", "Optional additional explanation.", required=False)
            .add_string("structured_summary", "Optional structured/Markdown summary for quick browsing.", required=False)
        )

        # Output exact target structure
        self.schema = seal_memory_tool.build(wrap_in_function=True)

    async def run(self, payload: ToolCall, context: SealMemoryContext) -> str:
        # Schema-level checks (types, required keys, enums) run in BaseTool.pre_tool_call.
        args = payload.arguments

        if self.min_seal_tokens > 0 and (context.runtime_config.token_estimator is None):
            raise ValueError(f"Set minimal seal token {self.min_seal_tokens} but didn't provide tokenizer!")

        if context.runtime_config.token_estimator:
            current_traj_num_tokens = await asyncio.to_thread(
                context.runtime_config.token_estimator,
                messages=context.trajectory.serialized_messages,
                context=context,
            )
            if self.min_seal_tokens > 0 and current_traj_num_tokens < self.min_seal_tokens:
                raise ReturnableToolError(f"Do not call seal_memory_tool too early (< {self.min_seal_tokens} tokens)")
        if self.max_seal_times > 0 and context.seal_count >= self.max_seal_times:
            raise ReturnableToolError(f"Max seal times reached ({context.seal_count}/{self.max_seal_times})")

        response = build_seal_memory_tool_response(**args)

        rollout_dir = getattr(context, "rollout_dir", None)
        if rollout_dir is not None:
            save_path = (
                question_dir(rollout_dir, context.question_id)
                / "memory"
                / f"{response['memory_id']}.json"
            )
        else:
            save_path = (
                Path(context.runtime_config.work_dir)
                / context.question_id
                / "memory"
                / f"{response['memory_id']}.json"
            )
        def _save():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as fh:
                json.dump(response, fh, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_save)
        return json.dumps(response, ensure_ascii=False)

def _find_memory_file(snapshot_dir: Path, memory_id: str) -> Optional[Path]:
    """Helper to locate the memory file by checking filenames, then contents."""
    compact_id = memory_id.replace("-", "")

    # 1. Fast path: check exact or compact filename patterns
    for pattern in (f"*{memory_id}.json", f"*{compact_id}.json"):
        if target_file := next(snapshot_dir.glob(pattern), None):
            return target_file

    # 2. Slow path: check all JSON files for stem matches or internal payload
    for file_path in snapshot_dir.glob("*.json"):
        if file_path.stem.endswith(compact_id) or memory_id in file_path.stem:
            return file_path

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                if json.load(fh).get("memory_id") == memory_id:
                    return file_path
        except (json.JSONDecodeError, OSError):
            continue

    return None

def _get_truncated_conversation(
    conversation: List[Any], max_messages: int, max_chars: int
) -> List[Any]:
    """Helper to safely truncate conversation history by message count and character limit."""
    if not conversation or max_messages <= 0:
        return []

    trimmed_convo = conversation[-max_messages:]
    safe_trimmed = []
    total_chars = 0

    for msg in trimmed_convo:
        try:
            msg_text = json.dumps(msg, ensure_ascii=False)
        except (TypeError, ValueError):
            msg_text = str(msg)

        # Stop adding messages if we exceed the character limit
        if total_chars + len(msg_text) > max_chars:
            break

        safe_trimmed.append(msg)
        total_chars += len(msg_text)

    return safe_trimmed


@tool_registry.register_tool(name="read_memory_tool")
class ReadMemoryTool(BaseTool):
    """Read a previously sealed memory file written by `SealMemoryTool`.

    The snapshot directory layout mirrors `SealMemoryTool.run`:
        <runtime_config.work_dir>/<question_id>/memory/<memory_id>.json
    """

    def __init__(self) -> None:
        custom_tool_prompt = "Read previously sealed memory."
        super().__init__(name="read_memory_tool", custom_tool_prompt=custom_tool_prompt)

        read_memory_schema = (
            ToolSchema(
                name="read_memory_tool",
                description=(
                    "Read a sealed memory by its memory_id for 'link summary'-style backtracking in a new context. "
                    "Returns the knowledge graph, navigation state, next-step plan, task progress, and optional "
                    "conversation/structured_summary slices."
                ),
            )
            .add_string("memory_id", "UUID returned by seal_memory_tool.", required=True)
            .add_boolean(
                "include_conversation",
                "Whether to include a truncated tail of the conversation history.",
                required=False,
                default=False,
            )
            .add_number(
                "max_messages",
                "Max number of trailing messages to include when include_conversation is true.",
                required=False,
                is_integer=True,
                default=10,
            )
            .add_number(
                "max_chars",
                "Max total characters of conversation history to include.",
                required=False,
                is_integer=True,
                default=8000,
            )
            .add_boolean(
                "include_structured",
                "Whether to include the structured_summary field (truncated).",
                required=False,
                default=False,
            )
            .add_number(
                "structured_max_chars",
                "Max characters of structured_summary to include when include_structured is true.",
                required=False,
                is_integer=True,
                default=6000,
            )
        )

        self.schema = read_memory_schema.build(wrap_in_function=True)

    def _get_snapshot_dirs(self, context: SealMemoryContext) -> List[Path]:
        rollout_dir = getattr(context, "rollout_dir", None)
        if rollout_dir is not None:
            rollout_dir = Path(rollout_dir)
            dirs = [question_dir(rollout_dir, context.question_id) / "memory"]
            # Archived attempts live outside the trajectory dir, so they are not
            # addressed through question_dir().
            retry_root = rollout_dir / "retry_history" / f"question_{context.question_id}"
            attempt_dirs = sorted(
                retry_root.glob("attempt_*"),
                key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
            )
            dirs.extend(p / "memory" for p in attempt_dirs)
            return dirs
        return [Path(context.runtime_config.work_dir) / context.question_id / "memory"]

    async def run(self, payload: ToolCall, context: SealMemoryContext) -> str:
        # Schema-level checks (types, required keys) run in BaseTool.pre_tool_call.
        args = payload.arguments
        memory_id = args["memory_id"]

        include_conversation = bool(args.get("include_conversation", False))
        max_messages = int(args.get("max_messages", 10))
        max_chars = int(args.get("max_chars", 8000))
        include_structured = bool(args.get("include_structured", False))
        structured_max_chars = int(args.get("structured_max_chars", 6000))

        snapshot_dirs = [p for p in self._get_snapshot_dirs(context) if p.exists()]
        if not snapshot_dirs:
            raise ReturnableToolError(f"No memory file is saved!")

        target_file = None
        for snapshot_dir in snapshot_dirs:
            target_file = await asyncio.to_thread(_find_memory_file, snapshot_dir, memory_id)
            if target_file:
                break
        if not target_file:
            raise ReturnableToolError(f"memory_id {memory_id} not found")

        def _load_memory():
            with open(target_file, "r", encoding="utf-8") as fh:
                return json.load(fh)

        try:
            data = await asyncio.to_thread(_load_memory)
        except Exception:
            raise ReturnableToolError(f"Cannot read memory {memory_id}")

        nav_state = data.get("navigation_state", {}) or {}

        response: Dict[str, Any] = {
            "memory_id": data.get("memory_id"),
            "timestamp": data.get("timestamp"),
            "stage": data.get("stage"),
            "tags": data.get("tags"),
            "comment": data.get("comment"),
            "knowledge_graph": data.get("knowledge_graph", []),
            "navigation_state": {
                "visited_summary": nav_state.get("visited_summary", ""),
                "frontier_queue": nav_state.get("frontier_queue", []),
                "dead_ends": nav_state.get("dead_ends", []),
            },
            "meta_learnings": data.get("meta_learnings", []),
            "next_step_plan": data.get("next_step_plan"),
            "task_progress": data.get("task_progress")
        }

        if include_conversation:
            response["conversation_history"] = _get_truncated_conversation(
                data.get("conversation_history", []), max_messages, max_chars
            )

        if include_structured:
            struct_text = data.get("structured_summary", "") or ""
            response["structured_summary"] = struct_text[:structured_max_chars]

        return json.dumps(response, ensure_ascii=False)
