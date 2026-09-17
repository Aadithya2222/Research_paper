"""Generate tasks/subset_75.json containing questions 1-25, 101-125, and 201-225."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_tasks import FIELDS_AND_TOPICS, build_task_entry

def main():
    subset_tasks = []
    # Take pair IDs 1 to 25
    for pair_id, field, topic, safety_tag in FIELDS_AND_TOPICS[:25]:
        for tier in (1, 2, 3):
            entry = build_task_entry(pair_id, field, topic, safety_tag, tier)
            subset_tasks.append(entry)

    out = Path("tasks/subset_75.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tasks": subset_tasks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(subset_tasks)} tasks (25 Easy, 25 Medium, 25 Hard) -> {out}")

if __name__ == "__main__":
    main()
