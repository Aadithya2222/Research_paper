"""Evaluation, kept strictly separate from execution.

Quality is scored three ways so that no single instrument carries the
whole result:
  * exact_match          -- deterministic, no judge involved
  * gold_fact_coverage   -- deterministic, no judge involved
  * judge_rubric         -- LLM judge, different model family, blinded

Role fidelity (RFS) is a LIGHTWEIGHT PROXY. It is not claimed as a novel
metric anywhere in this codebase, and it must be validated against an
established role-adherence instrument before any metric-level claim.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.llm import Ledger, OllamaClient

LOG = logging.getLogger(__name__)

ROLE_KEYS = ("planner", "researcher", "writer")


# ---------------------------------------------------------------------
# Deterministic quality measures
# ---------------------------------------------------------------------

def _norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def exact_match(answer: Optional[str], expected: Optional[str]) -> Optional[int]:
    """1 if the expected answer appears in the response, else 0.

    Returns None when the task has no gold answer, so that "not
    applicable" is never silently scored as "wrong".
    """
    if expected is None:
        return None
    if answer is None:
        return 0
    return int(_norm(expected) in _norm(answer))


def gold_fact_coverage(answer: Optional[str],
                       gold_facts: List[str]) -> Optional[float]:
    """Fraction of gold facts present in the answer."""
    if not gold_facts:
        return None
    if answer is None:
        return 0.0
    hay = _norm(answer)
    hits = sum(1 for f in gold_facts if _norm(f) in hay)
    return round(hits / len(gold_facts), 4)


# ---------------------------------------------------------------------
# Judge-based quality
# ---------------------------------------------------------------------

RUBRIC_PROMPT = """You are grading one answer to a task.

TASK:
{task}

REFERENCE INFORMATION (may be empty):
{reference}

ANSWER:
{answer}

Score the answer from 0 to 4:
0 = does not address the task
1 = addresses the task but is largely wrong
2 = partially correct, significant gaps
3 = substantially correct, minor gaps
4 = fully correct and complete

Reply with ONLY a single digit 0-4."""


def judge_rubric(judge: OllamaClient, ledger: Ledger, task_text: str,
                 answer: Optional[str], reference: str = "") -> Optional[float]:
    """Absolute rubric score in [0,1]. Returns None if the answer is absent."""
    if answer is None:
        return None
    prompt = RUBRIC_PROMPT.format(task=task_text, reference=reference or "(none)",
                                  answer=answer)
    reply = judge.chat(
        [{"role": "system", "content": "You are a strict, impartial grader."},
         {"role": "user", "content": prompt}],
        ledger, role_label="judge:rubric", is_judge=True)
    m = re.search(r"[0-4]", reply or "")
    if not m:
        return None
    return round(int(m.group()) / 4.0, 4)


PAIRWISE_PROMPT = """Compare two answers to the same task.

TASK:
{task}

ANSWER 1:
{a1}

ANSWER 2:
{a2}

Which answer is better? Reply with ONLY "1", "2", or "TIE"."""


def judge_pairwise(judge: OllamaClient, ledger: Ledger, task_text: str,
                   answer_x: str, answer_y: str,
                   position_swap: bool = True) -> str:
    """Blinded pairwise comparison with position swap.

    Runs the comparison twice with the sides exchanged. A verdict that
    flips when positions are swapped is recorded as a TIE, which both
    measures and neutralises position bias.
    Returns "X", "Y" or "TIE".
    """
    def _ask(a1: str, a2: str) -> str:
        reply = judge.chat(
            [{"role": "system", "content": "You are an impartial judge."},
             {"role": "user", "content": PAIRWISE_PROMPT.format(
                 task=task_text, a1=a1, a2=a2)}],
            ledger, role_label="judge:pairwise", is_judge=True)
        r = (reply or "").strip().upper()
        if r.startswith("1"):
            return "first"
        if r.startswith("2"):
            return "second"
        return "tie"

    v1 = _ask(answer_x, answer_y)
    if not position_swap:
        return {"first": "X", "second": "Y", "tie": "TIE"}[v1]

    v2 = _ask(answer_y, answer_x)          # sides swapped
    if v1 == "first" and v2 == "second":
        return "X"
    if v1 == "second" and v2 == "first":
        return "Y"
    return "TIE"                            # inconsistent => position bias


# ---------------------------------------------------------------------
# Role fidelity and role bleed
# ---------------------------------------------------------------------

RFS_PROMPT = """You are auditing whether an agent stayed within its assigned role.

ROLE CHARTER:
{charter}

THE AGENT'S OUTPUT:
{output}

Score role adherence from 0 to 4:
0 = role entirely ignored
1 = major violation of the role boundary
2 = partial adherence
3 = substantial adherence, minor drift
4 = full adherence

Then state whether the output contains work belonging to ANOTHER role
(for example: a planner answering the task, a researcher writing the
final prose, or a writer gathering new evidence).

Reply in exactly this format:
SCORE: <0-4>
BLEED: <YES or NO>"""


def score_roles(judge: OllamaClient, ledger: Ledger, cfg,
                stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Score each role's output for adherence and bleed."""
    roles = cfg.require("roles")
    scale_max = int(cfg.get("evaluation.rfs.scale_max", 4))
    per_role: Dict[str, Dict[str, Any]] = {}

    for key in ROLE_KEYS:
        stage = stage_outputs.get(key) or {}
        text = (stage.get("output") or "").strip()
        if not text:
            per_role[key] = {"score": None, "bleed": None}
            continue
        reply = judge.chat(
            [{"role": "system", "content": "You are a strict role auditor."},
             {"role": "user", "content": RFS_PROMPT.format(
                 charter=roles[key]["charter"], output=text[:4000])}],
            ledger, role_label=f"judge:rfs:{key}", is_judge=True)
        sm = re.search(r"SCORE\s*:\s*([0-4])", reply or "", re.IGNORECASE)
        bm = re.search(r"BLEED\s*:\s*(YES|NO)", reply or "", re.IGNORECASE)
        per_role[key] = {
            "score": int(sm.group(1)) if sm else None,
            "bleed": (bm.group(1).upper() == "YES") if bm else None,
        }

    scores = [v["score"] for v in per_role.values() if v["score"] is not None]
    bleeds = [v["bleed"] for v in per_role.values() if v["bleed"] is not None]
    return {
        "per_role": per_role,
        # RFS = mean(role scores) / scale_max * 100
        "rfs": round(sum(scores) / len(scores) / scale_max * 100, 2) if scores else None,
        "role_bleed_rate": round(sum(bleeds) / len(bleeds) * 100, 2) if bleeds else None,
    }


# ---------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------

def classify_failure(record: Dict[str, Any]) -> Optional[str]:
    """Assign a failure type. Never returns silently-dropped records."""
    if record.get("failure_type"):
        return record["failure_type"]
    if record.get("final_answer") is None:
        stopped = record.get("stopped_reason", "")
        mapping = {
            "planner_empty": "planner_failure",
            "researcher_empty": "researcher_failure",
            "writer_empty": "writer_failure",
            "max_iterations": "timeout",
            "exception": "framework_error",
            "no_candidate": "model_error",
        }
        return mapping.get(stopped, "unknown")
    return None


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def evaluate_file(cfg, raw_path: Path, tasks_by_id: Dict[str, Any],
                  judge: Optional[OllamaClient]) -> List[Dict[str, Any]]:
    """Evaluate every raw record in one JSONL file."""
    if not raw_path.exists():
        LOG.warning("No raw file at %s -- skipping.", raw_path)
        return []

    out: List[Dict[str, Any]] = []
    with raw_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = tasks_by_id.get(rec["task_id"])
            if task is None:
                LOG.warning("Record %s references unknown task %s",
                            rec["experiment_id"], rec["task_id"])
                continue

            ledger = Ledger()   # evaluation cost is tracked but NOT charged
                                # to the experimental compute budget
            em = exact_match(rec.get("final_answer"), task.expected_answer)
            gfc = gold_fact_coverage(rec.get("final_answer"), task.gold_facts)
            rubric = None
            if judge is not None and cfg.get("judge.enabled", True):
                reference = "; ".join(task.gold_facts) if task.gold_facts else ""
                rubric = judge_rubric(judge, ledger, task.task_text,
                                      rec.get("final_answer"), reference)

            # Primary quality: deterministic where possible, judge otherwise.
            if em is not None:
                quality = float(em)
            elif gfc is not None:
                quality = float(gfc)
            else:
                quality = rubric if rubric is not None else None

            roles = None
            if (cfg.get("evaluation.rfs.enabled", True) and judge is not None
                    and rec["condition"] in ("C", "D")):
                roles = score_roles(judge, ledger, cfg,
                                    rec.get("stage_outputs") or {})

            evaluated = {
                "experiment_id": rec["experiment_id"],
                "condition": rec["condition"],
                "task_id": rec["task_id"],
                "complexity_tier": rec["complexity_tier"],
                "complexity_score": rec.get("complexity_score"),
                "run_id": rec["run_id"],
                "exact_match": em,
                "gold_fact_coverage": gfc,
                "judge_rubric": rubric,
                "quality": quality,
                "success": None if quality is None else int(quality >= 0.5),
                "rfs": (roles or {}).get("rfs"),
                "role_bleed_rate": (roles or {}).get("role_bleed_rate"),
                "role_detail": (roles or {}).get("per_role"),
                "failure_type": classify_failure(rec),
                # cost carried through for the statistics stage
                "total_tokens": rec.get("total_tokens"),
                "input_tokens": rec.get("input_tokens"),
                "output_tokens": rec.get("output_tokens"),
                "llm_calls": rec.get("llm_calls"),
                "tool_calls": rec.get("tool_calls"),
                "retries": rec.get("retries"),
                "latency_seconds": rec.get("latency_seconds"),
                "ttft_seconds": rec.get("ttft_seconds"),
                "peak_ram_mb": rec.get("peak_ram_mb"),
                "mean_ram_mb": rec.get("mean_ram_mb"),
                "peak_gpu_memory_mb": rec.get("peak_gpu_memory_mb"),
                "handoff_count": len(rec.get("handoffs") or []),
                "handoff_chars": sum(h.get("chars", 0)
                                     for h in (rec.get("handoffs") or [])),
                "evaluation_cost": ledger.summary(include_judge=True)["total_tokens"],
            }
            out.append(evaluated)
    return out
