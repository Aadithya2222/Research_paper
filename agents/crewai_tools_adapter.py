"""Expose the shared ToolRegistry to CrewAI agents.

WHY THIS FILE EXISTS
--------------------
Condition D was constructed without a `tools=` argument on the CrewAI
Agent, so its agents had no tools at all. The zero tool-call count in
the first run was not the model declining to use tools; it had none to
use. That makes the D-C contrast conflate framework overhead with a
tool-availability difference.

This adapter wraps the same registry object used by Conditions A, B and
C, so all four conditions expose an identical tool inventory with
identical descriptions.

SCOPING NOTE (a real bug that shipped once)
-------------------------------------------
An earlier version wrote, inside the class body:

    description: str = description        # NameError

In a Python class body an annotated assignment makes the target a
class-local name, so the right-hand side is looked up locally and fails
before it is bound. The enclosing function parameter is shadowed. The
sibling line `name: str = tool_name` worked only because `tool_name`
was not also the assignment target.

This version binds every field from a differently-named local, and
build_crewai_tools self-tests a wrapper before returning, so the same
class of error cannot ship silently again.
"""
from __future__ import annotations

import logging
from typing import Any, List

LOG = logging.getLogger(__name__)


def _import_base_tool():
    """Locate CrewAI's tool base class across versions."""
    errors = []
    for module, attr in (("crewai.tools", "BaseTool"),
                         ("crewai_tools", "BaseTool"),
                         ("crewai.tools.base_tool", "BaseTool")):
        try:
            mod = __import__(module, fromlist=[attr])
            return getattr(mod, attr)
        except Exception as exc:
            errors.append(f"{module}.{attr}: {type(exc).__name__}: {exc}")
    LOG.error("Cannot import a CrewAI tool base class. Tried:\n  %s",
              "\n  ".join(errors))
    return None


def build_crewai_tools(registry, ledger) -> List[Any]:
    """Wrap registry tools as CrewAI tools, counting calls in the ledger.

    Returns an empty list if the base class cannot be imported, so the
    caller fails loudly rather than silently running without tools.
    Raises RuntimeError if a wrapper is built but does not carry its
    fields, which is better than finding out hours into a run.
    """
    BaseTool = _import_base_tool()
    if BaseTool is None:
        return []

    try:
        from pydantic import BaseModel, Field
    except Exception as exc:
        LOG.error("pydantic unavailable: %s", exc)
        return []

    class _Args(BaseModel):
        argument: str = Field(..., description="Input string for the tool.")

    def _make(tool_name: str, tool_desc: str) -> Any:
        """Build one wrapper.

        Every class-body field is bound from a local whose name DIFFERS
        from the field name. Naming them identically shadows the outer
        local and raises NameError at class-creation time.
        """
        captured_name = tool_name
        captured_desc = tool_desc
        captured_schema = _Args

        class _Wrapped(BaseTool):
            name: str = captured_name
            description: str = captured_desc
            args_schema: type = captured_schema

            def _run(self, argument: str = "", **kwargs: Any) -> str:
                arg = argument
                if not arg and kwargs:
                    arg = str(next(iter(kwargs.values())))
                out, elapsed, _ok = registry.call(captured_name, arg)
                # Counted exactly as in Conditions A/B/C, so tool
                # accounting is comparable across frameworks.
                ledger.tool_calls += 1
                ledger.tool_latency_seconds += elapsed
                return out

        _Wrapped.__name__ = f"CrewTool_{captured_name}"
        return _Wrapped()

    tools: List[Any] = []
    for name in registry.names:
        t = registry.get(name)
        if t is None:
            continue
        try:
            tools.append(_make(name, t.description))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build a CrewAI wrapper for tool '{name}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    # Self-test: prove a wrapper carries its fields rather than assuming.
    if tools:
        probe = tools[0]
        if not getattr(probe, "name", None):
            raise RuntimeError(
                "CrewAI tool wrapper built but its 'name' field is empty.")
        if not getattr(probe, "description", None):
            raise RuntimeError(
                "CrewAI tool wrapper built but its 'description' field "
                "is empty.")

    LOG.info("CrewAI tool adapter built %d tool(s): %s",
             len(tools), ", ".join(registry.names))
    return tools
