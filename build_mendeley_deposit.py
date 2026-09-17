"""Build Mendeley Data Deposit Package

Usage:
    python build_mendeley_deposit.py \
        --machines SEND_BACK_machine_1.zip SEND_BACK_machine_2.zip \
                   SEND_BACK_machine_3.zip SEND_BACK_machine_4.zip \
        --tasks tasks/tasks.json \
        --corpus tasks/corpus.json \
        --calibration calibration_tasks.json \
        --out mendeley_deposit
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble Mendeley Data deposit.")
    parser.add_argument(
        "--machines",
        nargs="+",
        required=True,
        help="List of SEND_BACK_machine_N.zip files from all machines",
    )
    parser.add_argument("--tasks", default="tasks/tasks.json", help="Path to benchmark tasks.json")
    parser.add_argument("--corpus", default="tasks/corpus.json", help="Path to corpus.json")
    parser.add_argument(
        "--calibration", default="calibration_tasks.json", help="Path to calibration_tasks.json"
    )
    parser.add_argument(
        "--out", default="mendeley_deposit", help="Output directory for deposit"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = out_dir / "raw_records"
    raw_dir.mkdir(parents=True, exist_ok=True)

    task_dir = out_dir / "benchmark_data"
    task_dir.mkdir(parents=True, exist_ok=True)

    stats_dir = out_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process machine zips
    total_records = 0
    records_per_cond = {"A": 0, "B": 0, "C": 0, "D": 0}

    print("=" * 60)
    print("  BUILDING MENDELEY DATA DEPOSIT PACKAGE")
    print("=" * 60)

    for zip_path_str in args.machines:
        zp = Path(zip_path_str)
        if not zp.exists():
            print(f"Error: Archive {zp} not found!")
            return 1

        print(f"--> Extracting {zp.name}...")
        with zipfile.ZipFile(zp, "r") as z:
            for item in z.infolist():
                if item.filename.endswith(".jsonl"):
                    filename = Path(item.filename).name
                    content = z.read(item).decode("utf-8")
                    lines = [l for l in content.splitlines() if l.strip()]
                    
                    cond = None
                    for c in ("A", "B", "C", "D"):
                        if f"condition_{c}" in item.filename:
                            cond = c
                            break
                    
                    if cond:
                        records_per_cond[cond] += len(lines)
                        total_records += len(lines)
                        target_file = raw_dir / f"condition_{cond}_pooled.jsonl"
                        with target_file.open("a", encoding="utf-8") as out_f:
                            out_f.write("\n".join(lines) + "\n")
                elif "statistics" in item.filename and item.filename.endswith(".json"):
                    dest = stats_dir / Path(item.filename).name
                    with dest.open("wb") as f_out:
                        f_out.write(z.read(item))

    # 2. Copy benchmark dataset files
    for src_path, target_name in [
        (args.tasks, "tasks.json"),
        (args.corpus, "corpus.json"),
        (args.calibration, "calibration_tasks.json"),
    ]:
        sp = Path(src_path)
        if sp.exists():
            shutil.copy(sp, task_dir / target_name)
            print(f"--> Copied {sp} to benchmark_data/{target_name}")
        else:
            print(f"Warning: {sp} does not exist.")

    # 3. Copy documentation files if present
    for doc in ["DATA_DICTIONARY.md", "README_TEMPLATE.md", "README.md", "config.yaml"]:
        dp = Path(doc)
        if dp.exists():
            shutil.copy(dp, out_dir / dp.name)
            print(f"--> Copied {dp.name} into deposit root.")

    # 4. Create Deposit Manifest
    manifest = {
        "deposit_title": "Coordination Tax Benchmark: Multi-Agent vs Compute-Matched Single Agent Raw Execution & Evaluation Dataset",
        "total_raw_records": total_records,
        "records_per_condition": records_per_cond,
        "machines_aggregated": len(args.machines),
    }
    with (out_dir / "DEPOSIT_MANIFEST.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    # 5. Create deposit zip archive
    zip_output = Path(f"{args.out}.zip")
    with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zout:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zout.write(p, arcname=str(p.relative_to(out_dir)))

    print("=" * 60)
    print(f"  DEPOSIT PACKAGE READY: {zip_output} ({zip_output.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Total pooled records: {total_records}")
    print(f"  Condition breakdown: {records_per_cond}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
