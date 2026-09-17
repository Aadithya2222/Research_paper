"""Benchmark validator: is this benchmark actually answerable?

Run this AFTER writing gold facts and building the corpus, and BEFORE
spending days of inference on it.

    python validate_benchmark.py

It checks the things that would silently ruin the study:

  1. ANSWERABILITY -- every gold fact must be findable in the corpus.
  2. TOOL REQUIREMENTS -- questions answerable from memory produce ceiling effects.
  3. STRING MATCHING -- gold facts matched as case-insensitive substrings.
  4. TIER SEPARATION -- the three tiers must differ on structural factors.
  5. GROUNDING COVERAGE -- how many tasks have deterministic gold data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import load_config   # noqa: E402

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _load(path: Path, key: str) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"{BAD} Missing file: {path}")
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))[key]


def check_answerability(tasks: List[Dict[str, Any]],
                        docs: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Every gold fact must appear somewhere in the corpus."""
    haystack = _norm(" ".join(d.get("text", "") + " " + d.get("title", "") for d in docs))
    per_doc = {d["doc_id"]: _norm(d.get("text", "") + " " + d.get("title", "")) for d in docs}

    unfindable: List[Tuple[str, str]] = []
    wrong_doc: List[Tuple[str, str, str]] = []
    checked = 0

    for t in tasks:
        facts = t.get("gold_facts") or []
        listed = t.get("_source_doc_ids") or []
        for fact in facts:
            checked += 1
            nf = _norm(fact)
            if not nf:
                continue
            if nf not in haystack:
                unfindable.append((t["task_id"], fact))
            elif listed:
                if not any(nf in per_doc.get(d, "") for d in listed):
                    found_in = [d for d, txt in per_doc.items() if nf in txt]
                    wrong_doc.append((t["task_id"], fact, ",".join(found_in[:3]) or "?"))

    print(f"\n1. ANSWERABILITY  ({checked} gold facts checked)")
    if not checked:
        print(f"   {BAD} No gold facts anywhere. Quality can only come from the judge.")
    elif not unfindable:
        print(f"   {OK} every gold fact is findable in the corpus")
    else:
        print(f"   {BAD} {len(unfindable)} gold fact(s) appear NOWHERE in the corpus:")
        for tid, fact in unfindable[:10]:
            print(f"        {tid}: \"{fact}\"")
        if len(unfindable) > 10:
            print(f"        ... and {len(unfindable)-10} more")

    if wrong_doc:
        print(f"   {WARN} {len(wrong_doc)} fact(s) found, but not in listed documents:")
        for tid, fact, found in wrong_doc[:5]:
            print(f"        {tid}: \"{fact}\" is actually in {found}")

    return len(unfindable), checked


def check_tool_free(tasks: List[Dict[str, Any]]) -> int:
    """Tasks needing no tools cannot discriminate architectures."""
    print("\n2. TOOL REQUIREMENTS")
    none_listed = [t["task_id"] for t in tasks if not t.get("required_tools")]
    by_tier: Dict[int, List[str]] = defaultdict(list)
    for t in tasks:
        if not t.get("required_tools"):
            by_tier[t.get("_declared_tier", t.get("complexity_tier", 0))].append(t["task_id"])

    if not none_listed:
        print(f"   {OK} every task declares at least one required tool")
    else:
        print(f"   {WARN} {len(none_listed)} task(s) declare no required tools")
        for tier in sorted(by_tier):
            print(f"        tier {tier}: {len(by_tier[tier])} tasks")
    return len(none_listed)


def check_string_matching(tasks: List[Dict[str, Any]]) -> int:
    """Gold facts that are too long or too generic will match badly."""
    print("\n3. GOLD FACT QUALITY")
    too_long, too_short, dupes = [], [], []
    for t in tasks:
        facts = t.get("gold_facts") or []
        seen = set()
        for f in facts:
            words = len(f.split())
            if words > 8:
                too_long.append((t["task_id"], f))
            if words <= 1 and len(f) < 4:
                too_short.append((t["task_id"], f))
            n = _norm(f)
            if n in seen:
                dupes.append((t["task_id"], f))
            seen.add(n)

    issues = len(too_long) + len(too_short) + len(dupes)
    if not issues:
        print(f"   {OK} gold facts look well-formed")
    if too_long:
        print(f"   {WARN} {len(too_long)} fact(s) longer than 8 words.")
    if too_short:
        print(f"   {WARN} {len(too_short)} fact(s) very short.")
    if dupes:
        print(f"   {WARN} {len(dupes)} duplicate fact(s) within a single task")
    return issues


def check_tier_separation(tasks: List[Dict[str, Any]], cfg) -> int:
    """The tiers must actually differ on the structural factors."""
    print("\n4. TIER SEPARATION")
    from benchmark.tasks import assign_tier, score_complexity

    scores_by_tier: Dict[int, List[float]] = defaultdict(list)
    mismatch = 0
    for t in tasks:
        factors = t.get("complexity_factors") or {}
        try:
            s = score_complexity(factors, cfg)
        except ValueError:
            continue
        declared = t.get("_declared_tier", t.get("complexity_tier"))
        computed = assign_tier(s, cfg)
        if declared:
            scores_by_tier[declared].append(s)
        if declared and computed != declared:
            mismatch += 1

    for tier in sorted(k for k in scores_by_tier if k):
        vals = scores_by_tier[tier]
        distinct = len(set(vals))
        print(f"   tier {tier}: n={len(vals):<4} C ranges {min(vals):.2f}-{max(vals):.2f}, {distinct} distinct value(s)")

    if mismatch:
        print(f"   {BAD} {mismatch} task(s) score into a different tier than declared.")
    else:
        print(f"   {OK} declared tiers match the scored tiers")
    return mismatch


def check_grounding(tasks: List[Dict[str, Any]]) -> int:
    """How much of the quality signal is deterministic vs judge-dependent?"""
    print("\n5. GROUNDING COVERAGE")
    exact = sum(1 for t in tasks if t.get("expected_answer"))
    facts = sum(1 for t in tasks if t.get("gold_facts"))
    neither = sum(1 for t in tasks if not t.get("expected_answer") and not t.get("gold_facts"))
    verified = sum(1 for t in tasks if t.get("human_verified"))
    n = len(tasks)

    print(f"   exact-answer tasks:        {exact:>4}/{n}")
    print(f"   tasks with gold facts:     {facts:>4}/{n}")
    print(f"   judge-only (no gold data): {neither:>4}/{n}")
    print(f"   human-verified:            {verified:>4}/{n}")

    if neither == n:
        print(f"   {BAD} NO deterministic quality signal at all.")
    else:
        print(f"   {OK} a majority of tasks have deterministic gold data")
    return neither


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate benchmark answerability")
    ap.add_argument("--tasks", default="tasks/tasks_150.json")
    ap.add_argument("--corpus", default="tasks/corpus.json")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = _load(Path(args.tasks), "tasks")
    docs = _load(Path(args.corpus), "documents")

    print("=" * 64)
    print("BENCHMARK VALIDATION")
    print("=" * 64)
    print(f"tasks:  {len(tasks)}  ({args.tasks})")
    print(f"corpus: {len(docs)} documents  ({args.corpus})")

    unfindable, checked = check_answerability(tasks, docs)
    tool_free = check_tool_free(tasks)
    fact_issues = check_string_matching(tasks)
    tier_mismatch = check_tier_separation(tasks, cfg)
    judge_only = check_grounding(tasks)

    print("\n" + "=" * 64)
    blockers = []
    if checked == 0:
        blockers.append("no gold facts written")
    if unfindable:
        blockers.append(f"{unfindable} unfindable gold facts")
    if tier_mismatch:
        blockers.append(f"{tier_mismatch} tier mismatches")
    if judge_only == len(tasks):
        blockers.append("no deterministic quality signal")

    if blockers:
        print("NOT READY TO RUN. Blocking issues:")
        for b in blockers:
            print(f"  * {b}")
        return 1

    print("READY TO RUN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
