"""Assemble the Mendeley Data deposit from the four machine archives.

    python build_mendeley_deposit.py \
        --machines SEND_BACK_machine_1.zip SEND_BACK_machine_2.zip \
                   SEND_BACK_machine_3.zip SEND_BACK_machine_4.zip \
        --tasks tasks/tasks.json --corpus tasks/corpus.json

ONE DEPOSIT, NOT FOUR
---------------------
The four machines did not use four different datasets. They ran disjoint
subsets of ONE 132-task benchmark, plus six calibration tasks executed
on every machine. Splitting that into four deposits would give four
DOIs, none of which is the dataset the paper analyses, and would make
the merge unreproducible.

This builds a single deposit in which machine provenance is preserved
two ways: every merged record carries a machine_id field, and each
machine's original untouched archive is included separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

CONDITIONS = ("A", "B", "C", "D")
MACHINE_CLASS = {1: "MX550", 2: "RTX3050", 3: "MX550", 4: "RTX3050"}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file, skipping lines corrupted by partial writes."""
    rows, bad = [], 0
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"    note: skipped {bad} unparseable line(s) in {path.name}")
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Mendeley deposit")
    ap.add_argument("--machines", nargs=4, required=True,
                    help="the four SEND_BACK_machine_N.zip files, in order")
    ap.add_argument("--tasks", default="tasks/tasks.json")
    ap.add_argument("--corpus", default="tasks/corpus.json")
    ap.add_argument("--calibration", default="calibration_tasks.json")
    ap.add_argument("--out", default="mendeley_deposit")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("01_benchmark", "02_merged_raw_records", "03_evaluated",
                "04_statistics", "05_per_machine_original"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. the benchmark itself -------------------------------------
    print("1. benchmark")
    tasks_path = Path(args.tasks)
    if tasks_path.exists():
        tasks = json.loads(tasks_path.read_text(encoding="utf-8"))["tasks"]
        (out / "01_benchmark" / "benchmark_tasks.json").write_text(
            json.dumps({"tasks": tasks}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        tiers = Counter(t.get("complexity_tier") for t in tasks)
        print(f"   {len(tasks)} tasks, tiers {dict(sorted(tiers.items()))}")
    else:
        tasks = []
        print(f"   WARNING: {tasks_path} not found; benchmark not included")

    for src, name in ((args.corpus, "local_corpus.json"),
                      (args.calibration, "calibration_tasks.json")):
        p = Path(src)
        if p.exists():
            shutil.copy2(p, out / "01_benchmark" / name)
            print(f"   included {name}")

    # ---- 2. merge the raw records ------------------------------------
    print("\n2. merging raw records")
    merged: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONDITIONS}
    seen = set()
    per_machine_counts: Dict[str, Counter] = {}

    with tempfile.TemporaryDirectory() as tmp:
        for idx, zpath in enumerate(args.machines, start=1):
            zp = Path(zpath)
            if not zp.exists():
                print(f"   ERROR: {zp} not found")
                return 1
            work = Path(tmp) / f"m{idx}"
            work.mkdir()
            with zipfile.ZipFile(zp) as z:
                z.extractall(work)

            # keep the original archive verbatim, as provenance
            shutil.copy2(zp, out / "05_per_machine_original" / zp.name)

            raw_dir = next((p for p in work.rglob("raw") if p.is_dir()), None)
            if raw_dir is None:
                print(f"   ERROR: no raw/ directory inside {zp.name}")
                return 1

            counts = Counter()
            for cond in CONDITIONS:
                f = raw_dir / f"condition_{cond}" / f"condition_{cond}.jsonl"
                for rec in _read_jsonl(f):
                    key = (rec.get("task_id"), cond,
                           rec.get("run_id", 1), idx)
                    if key in seen:
                        continue          # drop within-machine reruns
                    seen.add(key)
                    rec["machine_id"] = f"machine_{idx}"
                    rec["machine_class"] = MACHINE_CLASS.get(idx, "unknown")
                    merged[cond].append(rec)
                    counts[cond] += 1
            per_machine_counts[f"machine_{idx}"] = counts
            print(f"   machine_{idx}: {sum(counts.values())} records "
                  f"{dict(sorted(counts.items()))}")

    for cond in CONDITIONS:
        d = out / "02_merged_raw_records" / f"condition_{cond}"
        d.mkdir(parents=True, exist_ok=True)
        with (d / f"condition_{cond}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in merged[cond]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = sum(len(v) for v in merged.values())
    all_tasks = {r["task_id"] for v in merged.values() for r in v}
    print(f"   merged total: {total} records across {len(all_tasks)} tasks")

    # ---- 3. integrity checks -----------------------------------------
    print("\n3. integrity")
    problems = []
    task_machines: Dict[str, set] = {}
    for v in merged.values():
        for r in v:
            task_machines.setdefault(r["task_id"], set()).add(r["machine_id"])

    cal_ids = set()
    cal_p = Path(args.calibration)
    if cal_p.exists():
        cal_ids = {t["task_id"] for t in
                   json.loads(cal_p.read_text(encoding="utf-8"))["tasks"]}

    split = [t for t, ms in task_machines.items()
             if len(ms) > 1 and t not in cal_ids]
    if split:
        problems.append(f"{len(split)} non-calibration task(s) span machines")
        print(f"   FAIL: {len(split)} task(s) on more than one machine")
    else:
        print("   OK: no non-calibration task spans machines")

    hashes = {r.get("config_hash") for v in merged.values() for r in v}
    if len(hashes) == 1:
        print(f"   OK: single configuration hash {hashes.pop()}")
    else:
        problems.append(f"multiple config hashes: {hashes}")
        print(f"   FAIL: multiple config hashes {hashes}")

    if tasks:
        missing = {t["task_id"] for t in tasks} - all_tasks
        print(f"   task coverage: {len(all_tasks)}/{len(tasks)}"
              + (f", missing {sorted(missing)}" if missing else ""))

    # ---- 4. checksums and manifest -----------------------------------
    print("\n4. manifest")
    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p.relative_to(out)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": _sha256(p),
            })

    manifest = {
        "title": ("Single-Agent versus Multi-Agent LLM Orchestration: "
                  "Compute-Matched, Framework-Controlled Benchmark Data"),
        "n_tasks": len(tasks) if tasks else len(all_tasks),
        "n_merged_records": total,
        "conditions": {c: len(merged[c]) for c in CONDITIONS},
        "records_per_machine": {k: dict(v) for k, v in
                                per_machine_counts.items()},
        "machine_classes": MACHINE_CLASS,
        "calibration_task_ids": sorted(cal_ids),
        "integrity_problems": problems,
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")
    print(f"   {len(files)} files, checksums recorded")

    print("\n" + "=" * 62)
    if problems:
        print("  BUILT WITH PROBLEMS - fix before depositing:")
        for p in problems:
            print(f"    * {p}")
    else:
        print("  DEPOSIT READY")
    print(f"  {out.resolve()}")
    print("\n  Next: write README.md (template provided), then upload the")
    print("  whole folder to Mendeley Data as ONE dataset.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
