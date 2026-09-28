# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Observation:
    """A point-in-time view of an agent in the hierarchy at a yield boundary.

    `scope` reads outer -> inner: e.g. ("SealMemoryAgent", "BaseAgent") means
    the event was produced by the inner BaseAgent running under SealMemoryAgent.

    `context` is a live reference to the agent's local context at this level;
    outer consumers may mutate it between yields to steer the agent.
    """

    scope: tuple[str, ...]
    state: Any
    context: Any

    def under(self, name: str) -> "Observation":
        return replace(self, scope=(name, *self.scope))


@dataclass(frozen=True)
class Transition:
    """Terminal sentinel from a handler's async-generator, declaring next state.

    Every handler yields zero or more `Observation`s followed by exactly one
    `Transition`. The enclosing agent's `stream_run` consumes the `Transition`
    and does not forward it upward.
    """

    to_state: Any
