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
identical descriptions. Tool descriptions are read from the registry
rather than restated here, so the two frameworks cannot drift apart.

VERIFY BEFORE YOU TRUST IT
--------------------------
Binding tools does not guarantee the model calls them. Run
`verify_condition_d.py` on a single tool-requiring task and confirm a
non-zero tool-call count before committing to a full run.
"""
from __future__ import annotations

import logging
from typing import Any, List

LOG = logging.getLogger(__name__)


def build_crewai_tools(registry, ledger) -> List[Any]:
    """Wrap registry tools as CrewAI tools, counting calls in the ledger.

    Returns an empty list and logs a warning if CrewAI's tool base class
    cannot be imported, so the caller can fail loudly rather than
    silently running without tools again.
    """
    try:
        from crewai.tools import BaseTool
    except Exception:
        try:
            from crewai_tools import BaseTool  # older layouts
        except Exception as exc:
            LOG.error("Cannot import a CrewAI tool base class: %s. "
                      "Condition D would run WITHOUT tools, which is the "
                      "exact defect this adapter exists to fix.", exc)
            return []

    from pydantic import BaseModel, Field

    class _Args(BaseModel):
        argument: str = Field(...,
                              description="Input string for the tool.")

    def _make(tool_name: str, description: str) -> Any:

        class _Wrapped(BaseTool):
            name: str = tool_name
            description: str = description
            args_schema: type = _Args

            def _run(self, argument: str) -> str:
                out, elapsed, ok = registry.call(tool_name, argument)
                # Counted exactly as in Conditions A/B/C, so the tool
                # accounting is comparable across frameworks.
                ledger.tool_calls += 1
                ledger.tool_latency_seconds += elapsed
                return out

        _Wrapped.__name__ = f"CrewTool_{tool_name}"
        return _Wrapped()

    tools = []
    for name in registry.names:
        t = registry.get(name)
        if t is None:
            continue
        tools.append(_make(name, t.description))

    LOG.info("CrewAI tool adapter built %d tool(s): %s",
             len(tools), ", ".join(registry.names))
    return tools
