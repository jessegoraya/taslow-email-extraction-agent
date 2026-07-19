"""Optional Microsoft Agent Framework decorators for local Inspector authoring.

The Foundry-hosted runtime uses the framework-agnostic Invocations protocol adapter and must not
depend on Agent Framework. When the optional Inspector extra is absent, these decorators remain
transparent so the production workflow executes as ordinary async Python.
"""

from collections.abc import Callable
from typing import Any

try:  # pragma: no cover - exercised only when the external package is installed.
    from agent_framework import step, workflow
except Exception:  # pragma: no cover - deterministic fallback for local authoring.

    def step(func: Callable[..., Any] | None = None, **_: Any) -> Callable[..., Any]:
        def decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
            return inner

        return decorate(func) if func is not None else decorate

    def workflow(func: Callable[..., Any] | None = None, **_: Any) -> Callable[..., Any]:
        def decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
            return inner

        return decorate(func) if func is not None else decorate
