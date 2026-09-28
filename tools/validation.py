# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Strict argument validation for `BaseTool` payloads.

The agent loop hands each tool a `payload.arguments` dict produced by the LLM.
Without checking, malformed payloads (wrong types, missing keys, bad enums,
extra fields, nested object replaced with a string, ...) only surface as
opaque crashes deep inside `run()`.

This module wraps the schema each tool already builds with `ToolSchema` and
turns it into a `jsonschema.Validator`. We use **strict** matching: the LLM
must give exactly what the schema declares, no coercion. The error formatter
produces messages with JSON-pointer-like paths plus an "expected shape" hint
for object/array failures so the LLM can self-correct on the next turn.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


_TYPE_PLACEHOLDER = {
    "string": "<string>",
    "integer": "<integer>",
    "number": "<number>",
    "boolean": "<boolean>",
    "null": "<null>",
}

_MAX_REPR_LEN = 80
# 4 covers the deepest realistic shape: top-level object → nested object →
# array → item object (e.g. seal_memory_tool's navigation_state.frontier_queue[*]).
_MAX_HINT_DEPTH = 4


def extract_parameters(schema: dict | None) -> dict:
    """Return the JSON-Schema body that describes the tool's arguments.

    `ToolSchema.build()` can produce two shapes depending on `wrap_in_function`:
        unwrapped: {"name": ..., "description": ..., "parameters": {...}}
        wrapped:   {"type": "function", "function": {"name": ..., "parameters": {...}}}

    Some tools also assign `schema = {}` (e.g. test fixtures); treat that as
    "no validation required" by returning {}.
    """
    if not schema or not isinstance(schema, dict):
        # Guards against tools that forgot to call ToolSchema.build(); those
        # schemas can't be validated, so we silently skip rather than crash.
        return {}
    if isinstance(schema.get("function"), dict):
        return schema["function"].get("parameters", {}) or {}
    return schema.get("parameters", schema) or {}


def build_validator(schema: dict | None) -> Draft202012Validator | None:
    """Compile a strict validator. Returns None when no parameters are declared."""
    params = extract_parameters(schema)
    if not params:
        return None
    return Draft202012Validator(params)


def expected_shape_hint(schema: dict | None, max_depth: int = _MAX_HINT_DEPTH) -> str:
    """Render a compact JSON skeleton showing what `arguments` should look like.

    Only required fields are emitted (otherwise the hint balloons and stops
    being useful). Nested objects/arrays recurse up to `max_depth`.
    """
    params = extract_parameters(schema)
    if not params:
        return "<any>"
    skeleton = _skeleton(params, depth=0, max_depth=max_depth)
    return json.dumps(skeleton, ensure_ascii=False)


def _skeleton(node: dict, depth: int, max_depth: int) -> Any:
    if depth >= max_depth:
        return "<...>"

    if "oneOf" in node:
        # Show the first branch; that's usually the canonical shape.
        return _skeleton(node["oneOf"][0], depth, max_depth)

    node_type = node.get("type")

    if node_type == "object":
        props = node.get("properties", {}) or {}
        required = set(node.get("required", []))
        # Required-only keeps the hint focused; fall back to all keys if none required.
        keys = [k for k in props if k in required] or list(props.keys())
        return {k: _skeleton(props[k], depth + 1, max_depth) for k in keys}

    if node_type == "array":
        item_schema = node.get("items", {}) or {}
        return [_skeleton(item_schema, depth + 1, max_depth)]

    if isinstance(node_type, str):
        return _TYPE_PLACEHOLDER.get(node_type, f"<{node_type}>")

    return "<any>"


def format_validation_errors(
    errors: Iterable[ValidationError],
    tool_name: str,
    schema: dict | None,
) -> str:
    """Render a list of jsonschema errors into an LLM-actionable message.

    Object/array `type` failures get an extra "expected shape" line generated
    from the corresponding sub-schema so the model knows exactly what nested
    structure was expected (e.g. `navigation_state` requires `visited_summary`,
    `frontier_queue`, `dead_ends`).
    """
    errors = sorted(errors, key=lambda e: list(e.absolute_path))
    if not errors:
        return ""

    lines = [f"{tool_name}: argument validation failed"]
    for err in errors:
        path = _path_to_str(err.absolute_path)
        detail = _summarize_error(err)
        lines.append(f"  - {path}: {detail}")

        hint = _maybe_shape_hint(err)
        if hint:
            lines.append(f"    expected shape: {hint}")

    return "\n".join(lines)


def _path_to_str(path) -> str:
    """`['a', 0, 'b']` -> `'$.a[0].b'`. Empty path -> `'$'`."""
    out = ["$"]
    for part in path:
        if isinstance(part, int):
            out.append(f"[{part}]")
        else:
            out.append(f".{part}")
    return "".join(out)


def _summarize_error(err: ValidationError) -> str:
    """Hand-craft a short message per validator keyword.

    The default `err.message` is fine but verbose, and for `type` failures it
    omits the actual value which is the single most useful thing to show.
    """
    validator = err.validator

    if validator == "type":
        expected = err.validator_value
        expected_str = (
            " or ".join(expected) if isinstance(expected, list) else str(expected)
        )
        return f"expected {expected_str}, got {type(err.instance).__name__} ({_short_repr(err.instance)})"

    if validator == "required":
        # err.message is like "'foo' is a required property"
        missing = err.message.split("'")[1] if "'" in err.message else err.message
        return f"required property '{missing}' is missing"

    if validator == "enum":
        return f"{_short_repr(err.instance)} is not one of {err.validator_value}"

    if validator == "additionalProperties":
        return f"unexpected extra field(s) — {err.message}"

    if validator == "oneOf":
        return f"value did not match any of the {len(err.validator_value)} allowed shapes"

    return err.message


def _short_repr(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    if len(text) > _MAX_REPR_LEN:
        text = text[: _MAX_REPR_LEN - 3] + "..."
    return text


def _maybe_shape_hint(err: ValidationError) -> str | None:
    """Attach an expected-shape hint when the failure is a structural mismatch
    (object/array `type` mismatch). Skip primitive type errors — knowing that
    `expected: string, got: integer` is already enough."""
    if err.validator != "type":
        return None
    expected = err.validator_value
    if isinstance(expected, list):
        targets = expected
    else:
        targets = [expected]
    if not any(t in ("object", "array") for t in targets):
        return None
    return expected_shape_hint(err.schema)
