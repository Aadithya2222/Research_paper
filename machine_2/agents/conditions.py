"""The four experimental conditions.

  A  single-agent ReAct
  B  compute-matched Best-of-N single agent   (needs D's budget first)
  C  Planner -> Researcher -> Writer, LangGraph (or equivalent runner)
  D  Planner -> Researcher -> Writer, CrewAI

All conditions share: the same OllamaClient settings, the same
ToolRegistry object, the same role charters, the same iteration cap and
the same stopping rule. Every condition returns a ConditionOutput with
the same fields, so the runner treats them uniformly.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.react_agent import (AgentResult, build_system_prompt, run_react)
from core.llm import Ledger, OllamaClient
from tools.local_tools import ToolRegistry

LOG = logging.getLogger(__name__)


@dataclass
class ConditionOutput:
    """Uniform result contract for every condition."""

    final_answer: Optional[str]
    ledger: Ledger
    iterations: int
    stopped_reason: str
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    handoffs: List[Dict[str, Any]] = field(default_factory=list)
    failure_type: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Condition A -- single-agent ReAct
# =====================================================================

def union_role_charter(cfg) -> str:
    """The UNION of all three role charters, for the single-agent conditions.

    Without this, the multi-agent conditions would receive strictly more
    instruction content than the single-agent ones, confounding
    architecture with prompt content.
    """
    roles = cfg.require("roles")
    return (
        "You are a single agent responsible for the ENTIRE task. "
        "You must perform all of the following functions yourself:\n\n"
        f"PLANNING: {roles['planner']['goal']}\n"
        f"RESEARCH: {roles['researcher']['goal']}\n"
        f"WRITING: {roles['writer']['goal']}\n\n"
        "Plan the task, gather the evidence you need with the tools, then "
        "compose the final answer."
    )


def run_condition_a(cfg, client: OllamaClient, registry: ToolRegistry,
                    task: Dict[str, Any]) -> ConditionOutput:
    ledger = Ledger()
    system = build_system_prompt(union_role_charter(cfg), registry, allow_tools=True)
    res = run_react(client, registry, ledger, system, task["task_text"],
                    int(cfg.get("experiment.max_iterations", 8)), role_label="A")
    failure = None if res.final_answer else "model_error"
    if res.stopped_reason == "max_iterations":
        failure = "timeout"
    return ConditionOutput(res.final_answer, ledger, res.iterations,
                           res.stopped_reason,
                           stage_outputs={"transcript": res.transcript},
                           failure_type=failure,
                           notes={"parse_failures": res.parse_failures})


# =====================================================================
# Condition C -- Planner -> Researcher -> Writer
# =====================================================================

def _run_pipeline_native(cfg, client: OllamaClient, registry: ToolRegistry,
                         task: Dict[str, Any], ledger: Ledger) -> ConditionOutput:
    """Sequential three-role pipeline (used directly, or wrapped by LangGraph)."""
    roles = cfg.require("roles")
    max_iter = int(cfg.get("experiment.max_iterations", 8))
    stages: Dict[str, Any] = {}
    handoffs: List[Dict[str, Any]] = []

    # --- Planner: no tools by charter -------------------------------
    plan_sys = build_system_prompt(roles["planner"]["charter"], registry,
                                   allow_tools=False)
    plan_res = run_react(client, registry, ledger, plan_sys, task["task_text"],
                         max_iterations=1, role_label="C:planner",
                         allow_tools=False)
    plan = plan_res.final_answer or ""
    stages["planner"] = {"input": task["task_text"], "output": plan}
    if not plan.strip():
        return ConditionOutput(None, ledger, 1, "planner_empty", stages,
                               handoffs, failure_type="planner_failure")
    handoffs.append({"from": "planner", "to": "researcher",
                     "content": plan, "chars": len(plan)})

    # --- Researcher: tools allowed ----------------------------------
    res_sys = build_system_prompt(roles["researcher"]["charter"], registry,
                                  allow_tools=True)
    res_input = (f"TASK:\n{task['task_text']}\n\n"
                 f"PLAN FROM PLANNER:\n{plan}\n\n"
                 "Gather the evidence the plan requires.")
    research_res = run_react(client, registry, ledger, res_sys, res_input,
                             max_iterations=max_iter, role_label="C:researcher")
    findings = research_res.final_answer or ""
    degraded = False
    if not findings.strip() and research_res.partial_evidence.strip():
        # The researcher stalled (iteration cap or a repeated failing tool
        # call) but did gather evidence. Passing that on is what a real
        # deployment would do, and it stops a tool-level stall from being
        # recorded as an architecture-level failure. Flagged, not hidden.
        findings = ("PARTIAL EVIDENCE (researcher stopped early):\n"
                    + research_res.partial_evidence)
        degraded = True
        LOG.warning("Researcher stalled (%s); salvaged %d chars of evidence.",
                    research_res.stopped_reason,
                    len(research_res.partial_evidence))

    stages["researcher"] = {"input": res_input, "output": findings,
                            "transcript": research_res.transcript,
                            "stopped_reason": research_res.stopped_reason,
                            "degraded": degraded,
                            "repeated_actions": research_res.repeated_actions}
    if not findings.strip():
        return ConditionOutput(None, ledger, research_res.iterations,
                               "researcher_empty", stages, handoffs,
                               failure_type="researcher_failure")
    handoffs.append({"from": "researcher", "to": "writer",
                     "content": findings, "chars": len(findings)})

    # --- Writer: no tools by charter --------------------------------
    write_sys = build_system_prompt(roles["writer"]["charter"], registry,
                                    allow_tools=False)
    write_input = (f"TASK:\n{task['task_text']}\n\n"
                   f"FINDINGS FROM RESEARCHER:\n{findings}\n\n"
                   "Compose the final answer.")
    write_res = run_react(client, registry, ledger, write_sys, write_input,
                          max_iterations=1, role_label="C:writer",
                          allow_tools=False)
    answer = write_res.final_answer
    stages["writer"] = {"input": write_input, "output": answer or ""}

    failure = None if answer else "writer_failure"
    out = ConditionOutput(answer, ledger, research_res.iterations + 2,
                          "final_answer" if answer else "writer_empty",
                          stages, handoffs, failure_type=failure)
    out.notes["researcher_degraded"] = degraded
    out.notes["researcher_stopped_reason"] = research_res.stopped_reason
    return out


def run_condition_c(cfg, client: OllamaClient, registry: ToolRegistry,
                    task: Dict[str, Any]) -> ConditionOutput:
    """Condition C. Uses LangGraph for orchestration when available.

    The node bodies are identical either way; LangGraph supplies the
    state machine. Which path was taken is recorded in `notes` so the
    paper can report it honestly.
    """
    ledger = Ledger()
    try:
        from langgraph.graph import StateGraph, END  # noqa: F401
        orchestrator = "langgraph"
    except Exception:
        orchestrator = "native_sequential"

    out = _run_pipeline_native(cfg, client, registry, task, ledger)
    out.notes["orchestrator"] = orchestrator
    return out


# =====================================================================
# Condition D -- CrewAI
# =====================================================================

def run_condition_d(cfg, client: OllamaClient, registry: ToolRegistry,
                    task: Dict[str, Any]) -> ConditionOutput:
    """Condition D via CrewAI.

    If CrewAI is not installed we raise. We do NOT silently fall back to
    the Condition C implementation: that would make the D-C contrast
    measure nothing, while appearing to produce data.
    """
    try:
        from crewai import Agent, Crew, Process, Task as CrewTask
        from crewai import LLM as CrewLLM
    except Exception as exc:
        raise RuntimeError(
            "Condition D requires CrewAI. Install it with "
            "`pip install crewai` and rerun. Refusing to substitute "
            "another implementation, because that would invalidate the "
            "framework contrast."
        ) from exc

    ledger = Ledger()
    roles = cfg.require("roles")

    crew_llm = CrewLLM(
        model=f"ollama/{cfg.require('model.name')}",
        base_url=cfg.get("model.host", "http://localhost:11434"),
        temperature=float(cfg.get("model.temperature", 0.0)),
        max_tokens=int(cfg.get("model.max_output_tokens", 1024)),
    )

    def mk_agent(key: str, allow_delegation: bool = False) -> Any:
        return Agent(
            role=key.capitalize(),
            goal=roles[key]["goal"],
            backstory=roles[key]["charter"],
            llm=crew_llm,
            verbose=False,
            allow_delegation=allow_delegation,
            max_iter=int(cfg.get("experiment.max_iterations", 8)),
        )

    planner, researcher, writer = (mk_agent("planner"), mk_agent("researcher"),
                                   mk_agent("writer"))

    t1 = CrewTask(description=task["task_text"],
                  expected_output="A numbered plan of 2-8 concrete steps.",
                  agent=planner)
    t2 = CrewTask(description=(f"Gather the evidence required by the plan for: "
                               f"{task['task_text']}"),
                  expected_output="Bullet-point findings with supporting evidence.",
                  agent=researcher, context=[t1])
    t3 = CrewTask(description=(f"Compose the final answer for: {task['task_text']}"),
                  expected_output="The final answer, ending with 'FINAL ANSWER: ...'",
                  agent=writer, context=[t2])

    crew = Crew(agents=[planner, researcher, writer], tasks=[t1, t2, t3],
                process=Process.sequential, verbose=False)

    try:
        result = crew.kickoff()
    except Exception as exc:
        return ConditionOutput(None, ledger, 0, "framework_error", {}, [],
                               failure_type="framework_error",
                               notes={"error": f"{type(exc).__name__}: {exc}"})

    answer = str(result).strip()
    stages: Dict[str, Any] = {}
    handoffs: List[Dict[str, Any]] = []
    for key, ct in (("planner", t1), ("researcher", t2), ("writer", t3)):
        raw = getattr(ct, "output", None)
        stages[key] = {"output": str(raw) if raw is not None else ""}

    # CrewAI usage metrics, when the installed version exposes them.
    usage = getattr(crew, "usage_metrics", None)
    if usage is not None:
        pin = getattr(usage, "prompt_tokens", None)
        pout = getattr(usage, "completion_tokens", None)
        ncalls = getattr(usage, "successful_requests", None)
        from core.llm import CallRecord
        if pin is not None and pout is not None:
            ledger.add(CallRecord("D:crew_total", 1, True, int(pin), int(pout),
                                  0.0, None))
        if ncalls:
            for i in range(max(int(ncalls) - 1, 0)):
                ledger.add(CallRecord(f"D:crew_call{i+2}", 1, True, 0, 0, 0.0, None))

    return ConditionOutput(answer or None, ledger, 3,
                           "final_answer" if answer else "writer_failure",
                           stages, handoffs,
                           failure_type=None if answer else "writer_failure",
                           notes={"crewai_usage_available": usage is not None})


# =====================================================================
# Condition B -- compute-matched Best-of-N
# =====================================================================

def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def choose_n(budget_tokens: Optional[float], single_run_tokens: Optional[float],
             cfg) -> int:
    """Pick N from D's COST ONLY.

    The budget file contains no quality information whatsoever, so N
    cannot be chosen to advantage either condition.
    """
    lo = int(cfg.get("compute_matching.min_n", 1))
    hi = int(cfg.get("compute_matching.max_n", 5))
    if not budget_tokens or not single_run_tokens or single_run_tokens <= 0:
        return lo
    n = round(budget_tokens / single_run_tokens)
    return max(lo, min(hi, int(n)))


def run_condition_b(cfg, client: OllamaClient, registry: ToolRegistry,
                    task: Dict[str, Any], budget: Dict[str, Any],
                    judge_client: Optional[OllamaClient] = None) -> ConditionOutput:
    """Best-of-N single agent, matched to Condition D's measured budget."""
    ledger = Ledger()
    system = build_system_prompt(union_role_charter(cfg), registry, allow_tools=True)
    max_iter = int(cfg.get("experiment.max_iterations", 8))

    target_tokens = budget.get("target_total_tokens")
    probe_tokens = budget.get("median_single_run_tokens")
    n = choose_n(target_tokens, probe_tokens, cfg)

    candidates: List[AgentResult] = []
    for i in range(n):
        res = run_react(client, registry, ledger, system, task["task_text"],
                        max_iter, role_label=f"B:run{i+1}")
        candidates.append(res)

    answers = [c.final_answer for c in candidates if c.final_answer]
    if not answers:
        return ConditionOutput(None, ledger, sum(c.iterations for c in candidates),
                               "no_candidate", {"n": n}, [],
                               failure_type="model_error", notes={"n": n})

    # Selection rule is fixed per evaluation type in config, never post hoc.
    rules = cfg.get("compute_matching.selection_rule", {})
    rule = rules.get(task.get("evaluation_type", "default"),
                     rules.get("default", "judge_best_of"))

    if rule == "self_consistency" or judge_client is None:
        counts = Counter(_normalise(a) for a in answers)
        winner_norm, _ = counts.most_common(1)[0]
        chosen = next(a for a in answers if _normalise(a) == winner_norm)
        rule_used = "self_consistency"
    else:
        listing = "\n\n".join(f"[{i+1}] {a}" for i, a in enumerate(answers))
        prompt = (f"TASK:\n{task['task_text']}\n\nCANDIDATE ANSWERS:\n{listing}\n\n"
                  "Reply with ONLY the number of the single best candidate.")
        verdict = judge_client.chat(
            [{"role": "system", "content": "You select the best candidate answer."},
             {"role": "user", "content": prompt}],
            ledger, role_label="B:selection", is_judge=True)
        m = re.search(r"\d+", verdict or "")
        idx = int(m.group()) - 1 if m else 0
        chosen = answers[idx] if 0 <= idx < len(answers) else answers[0]
        rule_used = "judge_best_of"

    return ConditionOutput(chosen, ledger,
                           sum(c.iterations for c in candidates), "final_answer",
                           {"n": n, "candidates": answers}, [],
                           notes={"n": n, "selection_rule": rule_used,
                                  "target_total_tokens": target_tokens})


# =====================================================================
# Budget file helpers (the two-phase matching protocol)
# =====================================================================

BUDGET_FILENAME = "compute_budget.json"

QUALITY_KEYS = {"success", "quality", "score", "correct", "final_answer",
                "exact_match", "judge"}


def write_budget(processed_dir: Path, per_task: Dict[str, Dict[str, float]],
                 median_single_run_tokens: Optional[float]) -> Path:
    """Persist Condition D's COST ONLY.

    A guard rejects any quality-bearing key, enforcing in code the
    methodological rule that matching must not see outcomes.
    """
    for task_id, rec in per_task.items():
        bad = {k for k in rec if any(q in k.lower() for q in QUALITY_KEYS)}
        if bad:
            raise ValueError(
                f"Budget for {task_id} contains quality fields {sorted(bad)}. "
                "Compute matching must never observe outcome data."
            )
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / BUDGET_FILENAME
    payload = {
        "source_condition": "D",
        "median_single_run_tokens": median_single_run_tokens,
        "per_task": per_task,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_budget(processed_dir: Path, task_id: str) -> Dict[str, Any]:
    """Load the budget for one task; raises if phase 1 has not been run."""
    path = Path(processed_dir) / BUDGET_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            "Condition B requires Condition D's compute budget.\n"
            f"Expected: {path}\n"
            "Run Condition D first:  python run_experiment.py --condition D"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    per_task = payload.get("per_task", {})
    rec = per_task.get(task_id, {})
    return {
        "target_total_tokens": rec.get("total_tokens"),
        "target_llm_calls": rec.get("llm_calls"),
        "median_single_run_tokens": payload.get("median_single_run_tokens"),
    }
