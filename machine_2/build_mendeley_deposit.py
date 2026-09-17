"""Build the Mendeley Data deposit archive from machine zip files and benchmark assets.

Usage:
    python build_mendeley_deposit.py \
        --machines SEND_BACK_machine_1.zip SEND_BACK_machine_2.zip \
                   SEND_BACK_machine_3.zip SEND_BACK_machine_4.zip \
        --tasks tasks/tasks.json \
        --corpus tasks/corpus.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Mendeley Data deposit package from machine result archives."
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        help="List of SEND_BACK_machine_N.zip files or directories",
        default=["SEND_BACK_machine_1.zip", "SEND_BACK_machine_2.zip",
                 "SEND_BACK_machine_3.zip", "SEND_BACK_machine_4.zip"],
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("tasks/tasks.json"),
        help="Path to tasks.json benchmark task definitions",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("tasks/corpus.json"),
        help="Path to corpus.json retrieval corpus",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mendeley_deposit"),
        help="Output directory for Mendeley Data package",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create mendeley_deposit.zip archive",
    )

    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_dir / "raw_data"
    raw_out.mkdir(parents=True, exist_ok=True)
    tasks_out = out_dir / "tasks"
    tasks_out.mkdir(parents=True, exist_ok=True)
    stats_out = out_dir / "statistics"
    stats_out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  BUILDING MENDELEY DATA DEPOSIT PACKAGE")
    print("=" * 65)

    # 1. Copy benchmark task & corpus files
    if args.tasks.exists():
        shutil.copy(args.tasks, tasks_out / "tasks.json")
        print(f"  [OK] Copied benchmark tasks: {args.tasks}")
    else:
        print(f"  [WARNING] Task file missing: {args.tasks}")

    if args.corpus.exists():
        shutil.copy(args.corpus, tasks_out / "corpus.json")
        print(f"  [OK] Copied domain corpus: {args.corpus}")
    else:
        print(f"  [WARNING] Corpus file missing: {args.corpus}")

    calib_task = Path("calibration_tasks.json")
    if calib_task.exists():
        shutil.copy(calib_task, tasks_out / "calibration_tasks.json")
        print("  [OK] Copied calibration tasks")

    # 2. Extract and merge raw condition data from machine archives
    records_by_condition: dict[str, list[dict]] = {"A": [], "B": [], "C": [], "D": []}
    config_hashes: set[str] = set()
    machines_found: set[str] = set()

    for m_path in args.machines:
        p = Path(m_path)
        if not p.exists():
            print(f"  [SKIP] Machine archive not found: {m_path}")
            continue

        if p.is_file() and p.suffix == ".zip":
            with zipfile.ZipFile(p, "r") as z:
                for name in z.namelist():
                    if name.endswith("MANIFEST.json"):
                        try:
                            manifest = json.loads(z.read(name))
                            config_hashes.add(manifest.get("config_hash", "unknown"))
                            machines_found.add(manifest.get("machine_id", "unknown"))
                        except Exception:
                            pass
                    for cond in ("A", "B", "C", "D"):
                        if f"condition_{cond}.jsonl" in name:
                            content = z.read(name).decode("utf-8")
                            for line in content.splitlines():
                                if line.strip():
                                    rec = json.loads(line)
                                    records_by_condition[cond].append(rec)
                                    if "config_hash" in rec:
                                        config_hashes.add(rec["config_hash"])
                                    if "machine_id" in rec:
                                        machines_found.add(rec["machine_id"])

    # Write merged condition records
    total_records = 0
    for cond, recs in records_by_condition.items():
        cond_file = raw_out / f"condition_{cond}.jsonl"
        with cond_file.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        count = len(recs)
        total_records += count
        print(f"  [OK] Merged Condition {cond}: {count:>5} records -> {cond_file}")

    # 3. Copy statistics reports
    stats_dir = Path("results/statistics")
    if stats_dir.exists():
        for stat_file in stats_dir.glob("*.json"):
            shutil.copy(stat_file, stats_out / stat_file.name)
            print(f"  [OK] Copied statistical report: {stat_file.name}")

    # 4. Copy documentation files
    readme_template = Path("README_TEMPLATE.md")
    if readme_template.exists():
        shutil.copy(readme_template, out_dir / "README.md")
        print("  [OK] Created dataset README.md")

    data_dict = Path("DATA_DICTIONARY.md")
    if data_dict.exists():
        shutil.copy(data_dict, out_dir / "DATA_DICTIONARY.md")
        print("  [OK] Created dataset DATA_DICTIONARY.md")

    # 5. Output Summary & Validation Checks
    machines_str_list = sorted([str(m) for m in machines_found])
    print("-" * 65)
    print(f"  TOTAL RECORDS COLLECTED : {total_records} / 600")
    print(f"  MACHINES INCLUDED       : {', '.join(machines_str_list) or 'None'}")
    print(f"  CONFIG HASHES VERIFIED  : {', '.join(sorted([str(h) for h in config_hashes])) or 'None'}")

    if len(config_hashes) > 1:
        print("  [WARNING] Multiple config hashes detected across machines!")
    elif len(config_hashes) == 1:
        print("  [VERIFIED] Single consistent configuration hash across all records.")

    if args.zip or total_records > 0:
        zip_path = Path("mendeley_deposit.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for item in out_dir.rglob("*"):
                if item.is_file():
                    z.write(item, arcname=str(item.relative_to(out_dir)))
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"\n  Created deposit archive: {zip_path} ({size_mb:.2f} MB)")

    print("\n  Mendeley Deposit Package build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
