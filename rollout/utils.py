# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re

import yaml

from config import GeneralLLMConfig, RuntimeConfig
from llm_clients import OpenAIClient
from tools import tool_registry


# Keys that describe the client itself (passed to the client constructor) vs.
# sampling parameters that live on GeneralLLMConfig.
_CLIENT_PAYLOAD_KEYS = ("endpoint", "api_key", "model_name")
_LLM_CONFIG_KEYS = {
    *_CLIENT_PAYLOAD_KEYS,
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "max_tokens",
    "include_thoughts",
}

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value):
    """Resolve ``${VAR}`` references so configs can be committed without secrets.

    An undefined variable expands to an empty string rather than being left
    verbatim, which lets the client fall back to its own environment defaults
    instead of authenticating with the literal text ``${OPENAI_API_KEY}``.
    """
    if not isinstance(value, str):
        return value
    return _ENV_REF.sub(lambda m: os.getenv(m.group(1), ""), value) or None


def _prepare_tools(
    config_path,
    *,
    search_tool_function=None,
    link_summary_tool_function=None,
):
    tool_cfg = {}
    if config_path:
        with open(config_path, mode="r") as f:
            tool_cfg = json.load(f)

    function_overrides = {
        "search_api": search_tool_function,
        "link_summary_tool": link_summary_tool_function,
    }
    for tool_name, function_path in function_overrides.items():
        if function_path:
            tool_cfg.setdefault(tool_name, {})["function_path"] = function_path
    tool_registry.init_tools(tool_cfg)


def _build_client(payload):
    return OpenAIClient(**payload)


def load_llm_config_from_yaml(path):
    """Build a GeneralLLMConfig from a YAML LLM definition.

    The YAML holds OpenAI-compatible connection info and sampling defaults.
    A null value means the field is left unset.
    """
    with open(path, mode="r") as f:
        cfg = yaml.safe_load(f) or {}

    unknown_keys = sorted(set(cfg) - _LLM_CONFIG_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unsupported keys in {path}: {', '.join(unknown_keys)}"
        )

    payload = {key: _expand_env(cfg.get(key)) for key in _CLIENT_PAYLOAD_KEYS}

    return GeneralLLMConfig(
        llm_client=_build_client(payload),
        temperature=cfg.get("temperature"),
        top_p=cfg.get("top_p"),
        top_k=cfg.get("top_k"),
        min_p=cfg.get("min_p"),
        presence_penalty=cfg.get("presence_penalty"),
        repetition_penalty=cfg.get("repetition_penalty"),
        max_tokens=cfg.get("max_tokens"),
        include_thoughts=cfg.get("include_thoughts", True),
    )


def load_llm_config_from_args(args, prefix):
    """Legacy CLI-flag path for scripts not yet migrated to --{prefix}_config."""
    llm_payload = {
        "api_key": getattr(args, f"{prefix}_client_api_key", None),
        "endpoint": getattr(args, f"{prefix}_client_endpoint", None),
        "model_name": getattr(args, f"{prefix}_model_name", None),
    }
    return GeneralLLMConfig(
        llm_client=_build_client(llm_payload),
        temperature=getattr(args, f"{prefix}_temperature", None),
        top_p=getattr(args, f"{prefix}_top_p", None),
        max_tokens=getattr(args, f"{prefix}_max_tokens", None),
        include_thoughts=True,
    )


def _get_llm_config(args, prefix):
    if config_path := getattr(args, f"{prefix}_config", None):
        return load_llm_config_from_yaml(config_path)
    if any(
        getattr(args, f"{prefix}_{field}", None)
        for field in ("client_endpoint", "model_name")
    ):
        return load_llm_config_from_args(args, prefix)
    raise AssertionError(
        f"Missing --{prefix}_config (path to a YAML LLM config); "
        f"OpenAI-compatible endpoint/model flags are also unset."
    )
