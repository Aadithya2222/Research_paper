"""Run Condition D then Condition B, continuously, crash-safe.

    python rerun_auto.py              verify, then D, then B, end to end
    python rerun_auto.py --status     progress, safe any time
    python rerun_auto.py --skip-verify   only if verification already passed

CRASH SAFETY
------------
Every finished task is written to disk the moment it completes. If the
machine sleeps, crashes or is switched off, rerun the SAME command: it
resumes from the next unfinished task and repeats nothing.

Archiving of the old Condition D records happens EXACTLY ONCE, recorded
in .rerun_state.json. A resume never re-archives, so a crash midway
through cannot destroy the work already done. This is the one thing that
would silently ruin a resume, so it is tracked in state rather than left
to a command-line flag.

ORDERING
--------
Condition B's N is matched to Condition D's measured token budget, so B
cannot start until D is complete. The chain enforces this: if D is
unfinished, B is not attempted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.runner import (_emit_budget, _raw_path, completed_keys,
                              load_existing, run_one)            # noqa: E402
from benchmark.tasks import load_tasks                           # noqa: E402
from core.config import detect_hardware, load_config, setup_logging  # noqa: E402
from core.llm import client_from_config                          # noqa: E402
from tools.local_tools import build_registry                     # noqa: E402
from validation.checks import validate_config                    # noqa: E402

OK, BAD, WARN, INFO = "[ OK ]", "[FAIL]", "[WARN]", "[INFO]"
STATE = Path(".rerun_state.json")
FAILFAST_AFTER = 5
COOLDOWN_SECONDS = 900          # 15 min between 2-hour batches
BATCH_HOURS = 2.0


# ---------------------------------------------------------------------
# state
# ---------------------------------------------------------------------

def load_state() -> Dict[str, Any]:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"archived": {}, "verified": False, "started": None,
            "tool_calls": {}, "aborted": None}


def save_state(st: Dict[str, Any]) -> None:
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def archive_once(cfg, condition: str, st: Dict[str, Any]) -> None:
    """Archive the old records, but only the first time.

    A resume after a crash must NOT archive again: doing so would move
    the partially-completed new records aside and restart from zero.
    """
    if st["archived"].get(condition):
        print(f"{INFO} Condition {condition} already archived earlier "
              f"({st['archived'][condition]}); not archiving again.")
        return
    path = _raw_path(cfg, condition)
    if not path.exists():
        st["archived"][condition] = "nothing to archive"
        save_state(st)
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = path.parent.parent / f"condition_{condition}_OLD_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dest / path.name))
    st["archived"][condition] = dest.name
    save_state(st)
    print(f"{OK} archived old Condition {condition} -> {dest.name}")
    print("      Kept as evidence. Do not delete.")


# ---------------------------------------------------------------------
# one condition, batched, resumable
# ---------------------------------------------------------------------

def run_condition_batched(cfg, condition: str, tasks: List[Any],
                          client, registry, judge, hardware,
                          st: Dict[str, Any],
                          failfast: bool = True) -> int:
    """Run one condition to completion in cooled batches.

    Returns 0 complete, 4 aborted on zero tool calls, 5 finished with
    zero tool calls.
    """
    total = len(tasks)
    tool_total = int(st["tool_calls"].get(condition, 0))
    batch = 0

    while True:
        already = completed_keys(cfg, condition)
        queue = [t for t in tasks if (t.task_id, 1) not in already]
        if not queue:
            print(f"\n{OK} Condition {condition} complete "
                  f"({len(already)}/{total}).")
            break

        batch += 1
        print("\n" + "=" * 70)
        print(f"  CONDITION {condition}  |  batch {batch}  |  "
              f"{len(already)}/{total} done, {len(queue)} remaining")
        print("=" * 70)

        deadline = time.time() + BATCH_HOURS * 3600
        t0 = time.time()
        done_this_batch = 0
        zero_streak = 0

        for t in queue:
            if time.time() >= deadline:
                print(f"\n{INFO} 2-hour batch limit reached. "
                      f"Cooling for {COOLDOWN_SECONDS // 60} minutes, "
                      f"then continuing automatically.")
                break

            rec = run_one(cfg, condition, t, 1, client, registry, judge,
                          hardware)
            done_this_batch += 1
            tc = rec.get("tool_calls") or 0
            tool_total += tc
            st["tool_calls"][condition] = tool_total
            save_state(st)

            zero_streak = 0 if tc > 0 else zero_streak + 1
            elapsed = time.time() - t0
            eta = ((len(queue) - done_this_batch)
                   * (elapsed / done_this_batch) / 3600)
            status = ("ok" if rec["success_execution"]
                      else f"FAIL:{rec['failure_type']}")
            flag = "   <-- NO TOOLS" if tc == 0 and condition == "D" else ""
            print(f"  [{len(already)+done_this_batch}/{total}] {t.task_id} "
                  f"T{t.complexity_tier}  {status:<20} tools={tc:<3} "
                  f"tok={rec['total_tokens']}  {rec['latency_seconds']:.0f}s  "
                  f"eta {eta:.1f}h{flag}", flush=True)

            # fail fast: tools bound but never used
            n_done_total = len(already) + done_this_batch
            if (condition == "D" and failfast
                    and n_done_total >= FAILFAST_AFTER and tool_total == 0):
                st["aborted"] = "zero_tool_calls"
                save_state(st)
                print("\n" + "=" * 70)
                print(f"{WARN} ABORTED: {n_done_total} tasks done, "
                      f"0 tool calls in total.")
                print("=" * 70)
                print("  Tools are bound but the model is not invoking them.")
                print("  Continuing for hours would prove nothing.")
                print()
                print("  This may be a genuine finding rather than a bug:")
                print("  a small model declining tools under CrewAI's")
                print("  prompting is a reportable result.")
                print()
                print("  Send this whole output to the coordinator.")
                print("  DO NOT edit any code. DO NOT restart.")
                return 4

            if condition == "D" and zero_streak == 3:
                print(f"{WARN} 3 consecutive tasks with no tool calls. "
                      f"Continuing, but report this.")

        if condition == "D":
            _emit_budget(cfg, load_existing(cfg, "D"))

        remaining = [t for t in tasks
                     if (t.task_id, 1) not in completed_keys(cfg, condition)]
        if not remaining:
            continue
        print(f"\n{INFO} cooling {COOLDOWN_SECONDS // 60} min before the "
              f"next batch. Safe to leave unattended.")
        try:
            time.sleep(COOLDOWN_SECONDS)
        except KeyboardInterrupt:
            print(f"\n{INFO} interrupted during cooldown. Nothing lost. "
                  f"Rerun the same command to continue.")
            return 6

    if condition == "D" and tool_total == 0:
        st["aborted"] = "completed_zero_tool_calls"
        save_state(st)
        print(f"\n{WARN} Condition D finished with ZERO tool calls overall.")
        print("  Do NOT rerun Condition B: the existing B records remain")
        print("  valid against the original D budget. Report this instead.")
        return 5
    return 0


# ---------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------

def cmd_status(cfg, tasks) -> int:
    st = load_state()
    print("=" * 70)
    print("  RERUN STATUS")
    print("=" * 70)
    if st.get("started"):
        print(f"  started: {st['started']}")
    if st.get("aborted"):
        print(f"  {WARN} previously aborted: {st['aborted']}")
    for c in ["A", "C", "D", "B"]:
        done = len(completed_keys(cfg, c))
        recs = load_existing(cfg, c)
        tools = sum(r.get("tool_calls") or 0 for r in recs)
        withtools = sum(1 for r in recs if (r.get("tool_calls") or 0) > 0)
        pct = 100 * done / max(len(tasks), 1)
        bar = "#" * int(26 * done / max(len(tasks), 1))
        note = ""
        if c in ("A", "C"):
            note = "  (not being rerun)"
        elif st["archived"].get(c):
            note = f"  (RERUN; old archived: {st['archived'][c]})"
        else:
            note = "  (OLD records - not yet rerun)"
        print(f"  {c}: [{bar:<26}] {done:>3}/{len(tasks)} {pct:5.1f}%  "
              f"tools={tools:<5} ({withtools} tasks){note}")
    print()
    d_arch = bool(st["archived"].get("D"))
    b_arch = bool(st["archived"].get("B"))
    d_done = len(completed_keys(cfg, "D"))
    b_done = len(completed_keys(cfg, "B"))
    if not d_arch:
        print("  Condition D has NOT been rerun yet. The counts above are")
        print("  the ORIGINAL records.")
        print("  Next: python rerun_auto.py")
    elif d_done < len(tasks):
        print(f"  Condition D rerun in progress ({d_done}/{len(tasks)}).")
        print("  Next: python rerun_auto.py    (resumes where it stopped)")
    elif not b_arch or b_done < len(tasks):
        print(f"  Condition D rerun complete. Condition B "
              f"{'in progress' if b_arch else 'not started'} "
              f"({b_done}/{len(tasks)}).")
        print("  Next: python rerun_auto.py    (continues with B)")
    else:
        print("  Both conditions rerun and complete.")
        print("  Next: python send_back.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run Condition D then B, continuously and crash-safe")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="only if verification already passed")
    ap.add_argument("--task-set", default="my_tasks.json")
    ap.add_argument("--no-failfast", action="store_true",
                    help="do not abort on zero tool calls (ask first)")
    args = ap.parse_args()

    ts = Path(args.task_set)
    if not ts.exists():
        for alt in ("my_tasks.json", "tasks/tasks.json"):
            if Path(alt).exists():
                ts = Path(alt)
                break

    cfg = load_config("config.yaml")
    out_dir = Path(cfg.get("experiment.output_directory", "results"))
    setup_logging("INFO", True, out_dir / "rerun_auto.log")
    tasks = load_tasks(ts, cfg)

    if args.status:
        return cmd_status(cfg, tasks)

    st = load_state()
    if st.get("aborted") == "zero_tool_calls":
        print(f"{WARN} A previous run aborted because no tool calls were")
        print("  observed. That has not been resolved. Contact the")
        print("  coordinator before running again. Nothing has been lost.")
        return 4

    print("=" * 70)
    print("  CONDITION D + B AUTOMATIC RERUN")
    print("=" * 70)
    print(f"  machine   : {cfg.get('machine.id', 'unknown')}")
    print(f"  tasks     : {len(tasks)}  ({ts})")
    print(f"  config    : {cfg.config_hash}")
    print(f"  batching  : {BATCH_HOURS:.0f}h run, "
          f"{COOLDOWN_SECONDS // 60}min cooldown, repeating")
    print("  crash-safe: rerun this same command to resume; nothing repeats")
    print()

    validate_config(cfg).raise_if_failed()

    client = client_from_config(cfg)
    if not client.is_available():
        print(f"{BAD} Ollama unreachable. Start it: ollama serve")
        return 2
    if not client.model_present():
        print(f"{BAD} model missing. Run: ollama pull {client.model}")
        return 2
    judge = None
    if cfg.get("judge.enabled", True):
        j = client_from_config(cfg, judge=True)
        judge = j if j.model_present() else None
    registry = build_registry(cfg)
    hardware = detect_hardware()

    if not st.get("started"):
        st["started"] = datetime.now().isoformat(timespec="seconds")
        save_state(st)

    # ---- verification gate -------------------------------------------
    if not args.skip_verify and not st.get("verified"):
        print("=" * 70)
        print("  STEP 1  VERIFICATION")
        print("=" * 70)
        from verify_condition_d import main as verify_main
        rc = verify_main()
        if rc != 0:
            print(f"\n{BAD} Verification did not pass (exit {rc}). "
                  f"Stopping before any rerun.")
            print("  Send this output to the coordinator. Do not edit code.")
            return rc
        st["verified"] = True
        save_state(st)
        print(f"\n{OK} Verification passed. Continuing automatically.\n")
    else:
        print(f"{INFO} verification already passed; skipping to the rerun.\n")

    # ---- Condition D --------------------------------------------------
    # Archive on the FIRST invocation, regardless of how complete the old
    # records are. A rerun by definition starts from a complete previous
    # run, so gating this on incompleteness would archive nothing, rerun
    # nothing, and exit claiming success.
    archive_once(cfg, "D", st)
    rc = run_condition_batched(cfg, "D", tasks, client, registry, judge,
                               hardware, st, failfast=not args.no_failfast)
    if rc != 0:
        return rc

    # ---- Condition B --------------------------------------------------
    d_done = len(completed_keys(cfg, "D"))
    if d_done < len(tasks):
        print(f"{BAD} Condition D is {d_done}/{len(tasks)}. B cannot start.")
        return 3
    print("\n" + "=" * 70)
    print("  CONDITION D COMPLETE. STARTING CONDITION B AUTOMATICALLY.")
    print("=" * 70)
    print("  B's N is matched to D's measured token budget, which has just")
    print("  been regenerated. The old B records are now stale and are")
    print("  being archived and replaced.")
    archive_once(cfg, "B", st)
    rc = run_condition_batched(cfg, "B", tasks, client, registry, judge,
                               hardware, st, failfast=False)
    if rc != 0:
        return rc

    # ---- summary ------------------------------------------------------
    print("\n" + "=" * 70)
    print("  BOTH CONDITIONS COMPLETE")
    print("=" * 70)
    for c in ["D", "B"]:
        recs = load_existing(cfg, c)
        tools = sum(r.get("tool_calls") or 0 for r in recs)
        withtools = sum(1 for r in recs if (r.get("tool_calls") or 0) > 0)
        fails = sum(1 for r in recs if r.get("failure_type"))
        print(f"  {c}: {len(recs)} records, {tools} tool calls "
              f"({withtools} tasks), {fails} failures")
    print(f"\n  archived: {st['archived']}")
    print("\n  Next:  python send_back.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
