"""Small compatibility layer around Microsoft Agent Framework.

The production dependency is `agent-framework`. Tests can still import the package when the
dependency is unavailable because these decorators degrade to transparent wrappers. This keeps
business logic testable while allowing the same functions to participate in Agent Framework
runtime execution when the package is installed.
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
