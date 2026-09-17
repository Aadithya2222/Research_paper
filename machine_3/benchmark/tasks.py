"""Task loading and complexity scoring.

Complexity is computed from observable structural factors, not from an
author-assigned step count:

    C = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5      (Eq. 1)

Weights and tier cut points live in config.yaml and must be frozen
before the main run.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

FACTORS = ["n_tool_calls", "n_evidence_sources", "dependency_depth",
           "verification_required", "synthesis_required"]


@dataclass
class Task:
    task_id: str
    task_text: str
    evaluation_type: str                # "exact_match" | "rubric"
    expected_answer: Optional[str] = None
    gold_facts: List[str] = None        # type: ignore[assignment]
    required_tools: List[str] = None    # type: ignore[assignment]
    complexity_factors: Dict[str, float] = None  # type: ignore[assignment]
    complexity_score: Optional[float] = None
    complexity_tier: Optional[int] = None
    provenance: str = "constructed"     # "adapted" | "constructed"
    human_verified: bool = False

    def __post_init__(self) -> None:
        self.gold_facts = self.gold_facts or []
        self.required_tools = self.required_tools or []
        self.complexity_factors = self.complexity_factors or {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_complexity(factors: Dict[str, float], cfg) -> float:
    """Apply Eq. (1) with the frozen weights."""
    weights = cfg.require("complexity.weights")
    missing = [f for f in FACTORS if f not in factors]
    if missing:
        raise ValueError(f"Task is missing complexity factors: {missing}")
    return float(sum(float(weights[f]) * float(factors[f]) for f in FACTORS))


def assign_tier(score: float, cfg) -> int:
    """Map a complexity score to a tier using the frozen cut points."""
    t1 = float(cfg.require("complexity.tier_cutpoints.tier1_max"))
    t2 = float(cfg.require("complexity.tier_cutpoints.tier2_max"))
    if score <= t1:
        return 1
    if score <= t2:
        return 2
    return 3


def load_tasks(path: str | Path, cfg, tier: Optional[int] = None) -> List[Task]:
    """Load tasks, compute complexity, assign tiers, validate uniqueness."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Task file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    raw = payload.get("tasks", payload if isinstance(payload, list) else [])
    tasks: List[Task] = []
    seen: set[str] = set()
    for item in raw:
        valid_item = {k: v for k, v in item.items() if not k.startswith("_")}
        t = Task(**valid_item)
        if t.task_id in seen:
            raise ValueError(f"Duplicate task_id in {p}: {t.task_id}")
        seen.add(t.task_id)
        t.complexity_score = round(score_complexity(t.complexity_factors, cfg), 3)
        computed = assign_tier(t.complexity_score, cfg)
        if t.complexity_tier is not None and t.complexity_tier != computed:
            LOG.warning("Task %s: declared tier %s but scored tier %s "
                        "(score %.2f). Using the SCORED tier.",
                        t.task_id, t.complexity_tier, computed, t.complexity_score)
        t.complexity_tier = computed
        tasks.append(t)

    if tier is not None:
        tasks = [t for t in tasks if t.complexity_tier == tier]
    LOG.info("Loaded %d tasks from %s%s", len(tasks), p,
             f" (tier {tier})" if tier else "")
    return tasks


def tier_distribution(tasks: List[Task]) -> Dict[int, int]:
    dist: Dict[int, int] = {1: 0, 2: 0, 3: 0}
    for t in tasks:
        if t.complexity_tier in dist:
            dist[t.complexity_tier] += 1
    return dist
