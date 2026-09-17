"""Package this machine's results to send back to the coordinator.

    python send_back.py

Creates SEND_BACK_machine_N.zip containing the RAW RECORDS.

Why raw records and not the PDF
-------------------------------
The PDF is for you to read and check that your run went sensibly. It
cannot be merged. All the statistics -- paired differences, confidence
intervals, effect sizes -- have to be computed on the pooled raw data
from every machine at once. Averaging four PDFs would give the wrong
numbers, because a paired test is not the average of four separate
paired tests.

So: keep your PDF, send the zip.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import load_config   # noqa: E402


def main() -> int:
    cfg = load_config("config.yaml")
    machine_id = cfg.get("machine.id", "unknown")
    results = Path(cfg.get("experiment.output_directory", "results"))

    raw = results / "raw"
    if not raw.exists():
        print("No results/raw/ folder. Have you run the experiment yet?")
        print("  python pipeline.py --full --task-set my_tasks.json")
        return 1

    counts = {}
    total = 0
    for cond in ("A", "B", "C", "D"):
        f = raw / f"condition_{cond}" / f"condition_{cond}.jsonl"
        n = sum(1 for line in f.open(encoding="utf-8") if line.strip()) \
            if f.exists() else 0
        counts[cond] = n
        total += n

    print("=" * 60)
    print(f"  PACKAGING RESULTS -- MACHINE {machine_id}")
    print("=" * 60)
    print(f"  config hash : {cfg.config_hash}")
    for cond, n in counts.items():
        mark = "ok " if n else "MISSING"
        print(f"  condition {cond} : {n:>5} records  {mark}")

    missing = [c for c, n in counts.items() if n == 0]
    if missing:
        print(f"\n  WARNING: no records for condition(s) {', '.join(missing)}.")
        print("  The coordinator cannot compute contrasts that need them.")
        print("  If your run stopped early, finish it before sending.")

    out = Path(f"SEND_BACK_machine_{machine_id}.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for pattern in ("raw/**/*.jsonl", "processed/*.jsonl",
                        "processed/*.json", "statistics/*.json",
                        "environment.json", "*.log"):
            for p in results.glob(pattern):
                if p.is_file():
                    z.write(p, arcname=str(p.relative_to(results.parent)))
        manifest = {
            "machine_id": machine_id,
            "config_hash": cfg.config_hash,
            "model": cfg.get("model.name"),
            "judge": cfg.get("judge.name"),
            "records_per_condition": counts,
            "total_records": total,
        }
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

    size_mb = out.stat().st_size / (1024 ** 2)
    print(f"\n  Created: {out}  ({size_mb:.1f} MB, {total} records)")
    print("\n  Send this file to the coordinator.")
    print("  Keep your PDF in results/reports/ for your own reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
