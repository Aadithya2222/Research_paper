"""Decide whether CrewAI's provider prefix is why tools go uncalled.

    python diagnose_tools.py

ONE question, answered by measurement rather than argument:

    Does Condition D make tool calls with "ollama_chat/" when it made
    none with "ollama/"?

CrewAI routes through litellm. litellm's "ollama/" provider has weak
function-calling support; "ollama_chat/" is the provider documented for
tool use. If that is the cause, tools are bound but never offered to the
model, producing answers with zero tool calls.

This runs ONE tool-requiring task under each prefix and prints both
counts side by side. It writes to results_diag/ and cannot touch your
real data.

Takes about 5 minutes. Whatever the outcome, it is decisive:

  ollama_chat > 0, ollama = 0
      The prefix was the cause. Rerun D and B with the patch.

  both 0
      The prefix is NOT the cause. Stop. Report the zero-tool-call
      behaviour as a framework finding and keep your existing data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.tasks import load_tasks                  # noqa: E402
from core.config import load_config, setup_logging      # noqa: E402
from core.llm import client_from_config                 # noqa: E402
from tools.local_tools import build_registry            # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def run_with_provider(cfg, client, registry, task, provider: str):
    """Run one task under Condition D with a forced provider prefix."""
    # Override the default read by run_condition_d. config.yaml itself is
    # never written, so the configuration hash is unaffected.
    cfg.data.setdefault("model", {})["crewai_provider"] = provider
    from agents.conditions import run_condition_d
    try:
        out = run_condition_d(cfg, client, registry, task.to_dict())
    except Exception as exc:
        return {"provider": provider, "error": f"{type(exc).__name__}: {exc}"}
    s = out.ledger.summary()
    return {
        "provider": provider,
        "tool_calls": s["tool_calls"],
        "total_tokens": s["total_tokens"],
        "llm_calls": s["llm_calls"],
        "answered": out.final_answer is not None,
        "tools_bound": out.notes.get("crew_tools_bound"),
    }


def main() -> int:
    print("=" * 68)
    print("  PROVIDER PREFIX DIAGNOSTIC")
    print("=" * 68)

    cfg = load_config("config.yaml")
    cfg.data["experiment"]["output_directory"] = "results_diag"
    cfg.data["logging"]["raw_directory"] = "results_diag/raw"
    setup_logging("WARNING", console=True)
    print(f"{OK} config hash {cfg.config_hash} (unchanged; not written to)")

    try:
        import crewai
        print(f"{OK} crewai {getattr(crewai, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{BAD} crewai unavailable: {exc}")
        return 1

    client = client_from_config(cfg)
    if not client.is_available() or not client.model_present():
        print(f"{BAD} Ollama or the model is unavailable.")
        return 1
    registry = build_registry(cfg)

    tasks = load_tasks("my_tasks.json", cfg)
    tool_tasks = [t for t in tasks if t.required_tools] or tasks
    task = tool_tasks[0]
    print(f"{OK} test task: {task.task_id} (tier {task.complexity_tier})")
    print(f"      requires: {', '.join(task.required_tools) or 'unspecified'}")
    print("\nRunning the SAME task under both prefixes. ~5 minutes.\n")

    results = []
    for provider in ("ollama", "ollama_chat"):
        print(f"  {provider}/ ...", flush=True)
        r = run_with_provider(cfg, client, registry, task, provider)
        results.append(r)
        if "error" in r:
            print(f"      ERROR: {r['error']}")
        else:
            print(f"      tool calls: {r['tool_calls']}   "
                  f"tokens: {r['total_tokens']}   "
                  f"answered: {r['answered']}   "
                  f"tools bound: {r['tools_bound']}")

    print("\n" + "=" * 68)
    old = next((r for r in results if r["provider"] == "ollama"), {})
    new = next((r for r in results if r["provider"] == "ollama_chat"), {})
    old_tc = old.get("tool_calls", 0) or 0
    new_tc = new.get("tool_calls", 0) or 0

    if new.get("error"):
        print(f"{BAD} ollama_chat/ raised: {new['error']}")
        print("  The prefix is not usable with this CrewAI/litellm version.")
        print("  STOP. Report the zero-tool-call behaviour as a finding.")
        return 3

    if new_tc > 0 and old_tc == 0:
        print(f"{OK} DECISIVE: ollama_chat/ made {new_tc} tool call(s); "
              f"ollama/ made {old_tc}.")
        print()
        print("  The provider prefix WAS the cause. The patched")
        print("  agents/conditions.py already defaults to ollama_chat/.")
        print()
        print("  Proceed with the rerun:")
        print("      python rerun_auto.py")
        print()
        print("  Report in the paper that CrewAI's default litellm")
        print("  provider string did not surface bound tools, and that")
        print("  ollama_chat/ was required. That is a real framework")
        print("  observation and belongs in Threats to Validity.")
        return 0

    if new_tc > 0 and old_tc > 0:
        print(f"{WARN} Both prefixes made tool calls "
              f"(ollama={old_tc}, ollama_chat={new_tc}).")
        print("  The earlier zero-tool result may have been task-specific.")
        print("  Send this output to the coordinator before rerunning.")
        return 2

    print(f"{WARN} NOT THE CAUSE: both prefixes made 0 tool calls.")
    print()
    print("  Tools are bound and functional, and the model declines to")
    print("  invoke them under CrewAI regardless of provider string.")
    print()
    print("  STOP HERE. Do not rerun Condition D.")
    print("  Your existing data stands. Report this as a framework")
    print("  finding: CrewAI's orchestration of an 8B model did not")
    print("  elicit tool use, while the same model used tools readily")
    print("  in Conditions A, B and C.")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
