"""A single ReAct agent: reason -> act -> observe -> repeat.

This module is shared by Condition A, Condition B (which runs it N
times) and by the Researcher node of Condition C. Sharing one
implementation is a fairness requirement: it guarantees the loop
semantics, stopping rule and tool interface are byte-identical across
conditions, so any measured difference is attributable to the
architecture, not to two different loop implementations.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.llm import Ledger, OllamaClient
from tools.local_tools import ToolRegistry

LOG = logging.getLogger(__name__)

ACTION_RE = re.compile(r"Action\s*:\s*([a-zA-Z_]+)\s*", re.IGNORECASE)
INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(.+)", re.IGNORECASE)
FINAL_RE = re.compile(r"FINAL\s*ANSWER\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

REACT_INSTRUCTIONS = """You solve the task using a strict loop.

On each turn output EITHER a tool call:

Thought: <one sentence of reasoning>
Action: <tool name>
Action Input: <input for the tool>

OR the final result:

Thought: <one sentence of reasoning>
FINAL ANSWER: <your answer>

Rules:
- Emit exactly one Action per turn, or the FINAL ANSWER. Never both.
- Never invent tool output. Wait for the Observation.
- When you have enough information, give the FINAL ANSWER immediately.
"""


@dataclass
class AgentResult:
    final_answer: Optional[str]
    iterations: int
    transcript: List[Dict[str, str]]
    stopped_reason: str   # final_answer | max_iterations | parse_failure | repeat_loop
    parse_failures: int
    repeated_actions: int = 0
    partial_evidence: str = ""   # observations gathered before giving up


def build_system_prompt(role_charter: str, registry: ToolRegistry,
                        allow_tools: bool = True) -> str:
    """Assemble a system prompt from a role charter plus shared tool text."""
    parts = [role_charter.strip()]
    if allow_tools:
        parts.append(registry.describe())
        parts.append(REACT_INSTRUCTIONS.strip())
    else:
        parts.append("You have NO tools available. Do not emit Action lines.")
        parts.append("End your reply with: FINAL ANSWER: <answer>")
    return "\n\n".join(parts)


def run_react(client: OllamaClient, registry: ToolRegistry, ledger: Ledger,
              system_prompt: str, task_text: str, max_iterations: int,
              role_label: str = "agent", allow_tools: bool = True) -> AgentResult:
    """Execute the ReAct loop until a final answer or the iteration cap."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    transcript: List[Dict[str, str]] = []
    parse_failures = 0
    action_history: List[str] = []
    repeated_actions = 0
    observations: List[str] = []

    def _partial() -> str:
        """Evidence actually gathered, for salvage when the loop stalls."""
        return "\n".join(observations)

    for step in range(1, max_iterations + 1):
        reply = client.chat(messages, ledger, role_label=f"{role_label}:step{step}")
        transcript.append({"step": str(step), "role": "assistant", "content": reply})

        final = FINAL_RE.search(reply)
        if final:
            return AgentResult(final.group(1).strip(), step, transcript,
                               "final_answer", parse_failures,
                               repeated_actions, _partial())

        if not allow_tools:
            # No tools and no FINAL ANSWER marker: treat the whole reply
            # as the answer rather than discarding a usable output.
            return AgentResult(reply.strip(), step, transcript,
                               "final_answer", parse_failures,
                               repeated_actions, _partial())

        action = ACTION_RE.search(reply)
        arg_match = INPUT_RE.search(reply)
        if not action:
            parse_failures += 1
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": ("Observation: your reply did not contain a valid "
                            "Action or FINAL ANSWER. Reply again using the "
                            "required format."),
            })
            continue

        tool_name = action.group(1).strip()
        tool_arg = arg_match.group(1).strip() if arg_match else ""

        # Loop detection. Live runs showed agents repeating an identical
        # failing tool call until the iteration cap, which manufactures a
        # failure unrelated to the architecture being compared.
        signature = f"{tool_name}|{tool_arg}"
        action_history.append(signature)
        if action_history.count(signature) >= 3:
            repeated_actions += 1
            LOG.warning("Repeated identical action 3x (%s); stopping loop.",
                        signature[:80])
            return AgentResult(None, step, transcript, "repeat_loop",
                               parse_failures, repeated_actions, _partial())

        output, elapsed, ok = registry.call(tool_name, tool_arg)
        if ok:
            observations.append(f"{tool_name}({tool_arg}) -> {output}")
        ledger.tool_calls += 1
        ledger.tool_latency_seconds += elapsed
        transcript.append({"step": str(step), "role": "tool",
                           "content": f"{tool_name}({tool_arg}) -> {output}"})

        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation: {output}"})

    return AgentResult(None, max_iterations, transcript, "max_iterations",
                       parse_failures, repeated_actions, _partial())
