# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from enum import Enum, auto
from functools import cached_property
from typing import Any

from agents.events import Observation, Transition


class ExceptionAction(Enum):
    RAISE = auto()
    BREAK = auto()


class HandlerRegistryMixin:
    """Discovers async-generator state handlers marked with `_handles_state`."""

    @cached_property
    def handlers(self) -> dict[Any, Any]:
        registered = {}
        for cls in reversed(type(self).mro()):
            for name, attr in cls.__dict__.items():
                state = getattr(attr, "_handles_state", None)
                if state is None:
                    continue
                method = getattr(self, name)
                if not inspect.isasyncgenfunction(method):
                    raise TypeError(
                        f"Handler '{name}' must be an async generator yielding "
                        "Observations and terminating with `Transition`."
                    )
                registered[state] = method
        return registered

    def __getitem__(self, state: Any) -> Any:
        try:
            return self.handlers[state]
        except KeyError:
            raise KeyError(f"No handler registered for state: {state}") from None


class StateMachineRunner:
    """Shared event-emitting state-machine loop for nested agents."""

    def __init__(
        self,
        *,
        name: str,
        handler: Any,
        end_state: Any,
        should_stop: Callable[[Any, Any], bool] | None = None,
        on_exception: Callable[[Exception, Any, Any], ExceptionAction] | None = None,
        on_natural_end: Callable[[Any, Any], None] | None = None,
        use_state_stack: bool = True,
    ) -> None:
        self.name = name
        self.handler = handler
        self.end_state = end_state
        self.should_stop = should_stop
        self.on_exception = on_exception
        self.on_natural_end = on_natural_end
        self.use_state_stack = use_state_stack

    async def stream(self, context: Any, state: Any):
        while state != self.end_state:
            if self.should_stop is not None and self.should_stop(context, state):
                break

            current_state = state
            if self.use_state_stack:
                context.push_state(self.name, current_state)

            try:
                yield Observation(scope=(self.name,), state=current_state, context=context)

                next_state = None
                async for ev in self.handler[current_state](context):
                    if isinstance(ev, Transition):
                        next_state = ev.to_state
                        break
                    yield ev.under(self.name)

                if next_state is None:
                    raise RuntimeError(
                        f"Handler for state {current_state} did not emit a Transition"
                    )

            except Exception as exc:
                action = (
                    self.on_exception(exc, context, current_state)
                    if self.on_exception is not None
                    else ExceptionAction.RAISE
                )
                if action is ExceptionAction.BREAK:
                    break
                if action is not ExceptionAction.RAISE:
                    raise RuntimeError(f"Unknown exception action: {action!r}") from exc
                raise
            finally:
                if self.use_state_stack:
                    context.pop_state(self.name, current_state)

            state = next_state
        else:
            if self.on_natural_end is not None:
                self.on_natural_end(context, state)

        yield Observation(scope=(self.name,), state=state, context=context)
