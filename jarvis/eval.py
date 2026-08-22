"""
eval.py — a small, fixed benchmark for JARVIS's reasoning/tool-use/routing
behavior. From the 2026-08-22 systems audit (P2): every fix in this project
had been verified by hand, once, against its own specific repro — nothing
caught a regression in a *different* capability a change didn't directly
touch. This is not a replacement for tests/ (which checks code correctness);
it checks *behavior* — does JARVIS still route, answer, and use tools the
way it did last time this ran.

Two tiers, run separately:
  FAST cases  — skill routing/priority, deterministic, no live model or
                Ollama needed. Safe to run constantly.
  LIVE cases  — the actual reasoning model and/or MCP tool-calling loop.
                Needs `ollama serve` running. Slower, and LLM output varies
                run to run, so checks are keyword/property-based ("contains
                'paris'"), not exact-match — the LLM is not deterministic
                and pretending otherwise would make every case flaky.

Usage:
    python -m jarvis.eval            # fast cases only
    python -m jarvis.eval --live     # fast + live cases (needs Ollama)

Deliberately not a general eval framework — ~15 fixed cases, one file, no
database, no historical trend storage. That's the right scope for a
personal single-user project; see the audit's Evaluation System section
for why the heavier version isn't warranted here. Extend EVAL_CASES
directly when a new capability is worth pinning a regression check to.
"""

from __future__ import annotations

import sys
import time

try:
    from .modules.base import Registry
    from .modules.builtin import CalculatorSkill
    from .modules.os_control import OSControlSkill
    from .modules.health import HealthCheckSkill
    from .modules.memory import MemorySkill
    from .modules.fileread import FileReadSkill
    from .modules.market_analysis import MarketAnalysisSkill
    from .modules.reasoning import ReasoningSkill
except ImportError:  # pragma: no cover - legacy direct execution
    from modules.base import Registry
    from modules.builtin import CalculatorSkill
    from modules.os_control import OSControlSkill
    from modules.health import HealthCheckSkill
    from modules.memory import MemorySkill
    from modules.fileread import FileReadSkill
    from modules.market_analysis import MarketAnalysisSkill
    from modules.reasoning import ReasoningSkill


class EvalCase:
    def __init__(self, name: str, prompt: str, check, live: bool = False):
        self.name = name
        self.prompt = prompt
        self.check = check  # (reply: str) -> bool
        self.live = live


def _fast_registry(tmp_memory_path: str) -> Registry:
    reg = Registry()
    reg.register(CalculatorSkill())
    reg.register(OSControlSkill())  # ungated calls will just deny -- fine, we're checking routing not access
    reg.register(HealthCheckSkill())
    reg.register(MemorySkill(path=tmp_memory_path))
    reg.register(FileReadSkill())
    reg.register(MarketAnalysisSkill())
    return reg


FAST_CASES = [
    EvalCase("calculator_beats_nothing_else", "what is 15 * 4",
              lambda r: r.strip() == "60"),
    EvalCase("calculator_handles_parens", "calculate (3 + 2) * 4",
              lambda r: r.strip() == "20"),
    EvalCase("os_control_list_processes_routes_correctly", "list processes",
              lambda r: "processes running" in r),
    EvalCase("os_control_open_requires_auth", "open notepad",
              lambda r: "Denied" in r),  # nothing enrolled in an eval run -- correct to deny
    EvalCase("os_control_ignores_ordinary_sentences", "start describing the weather patterns",
              lambda r: r is None),  # None = no skill claimed it (regression guard for the trigger-overreach fix)
    EvalCase("health_check_routes_and_reports", "system health",
              lambda r: r.startswith("System health:")),
    EvalCase("fileread_reads_real_project_file", "read file jarvis/README.md",
              lambda r: len(r) > 200),
    EvalCase("fileread_denies_security_data", "read file jarvis/data/security/passphrase.hash",
              lambda r: "excluded" in r),
    EvalCase("market_analysis_disclaimer_always_present", "backtest AAPL",
              lambda r: "not investment advice" in r),
]

LIVE_CASES = [
    EvalCase("reasoning_answers_a_simple_fact", "What is the capital of France? Answer in one word.",
              lambda r: "paris" in r.lower(), live=True),
    EvalCase("reasoning_does_basic_arithmetic_in_prose", "If I have 3 apples and get 4 more, how many do I have?",
              lambda r: "7" in r, live=True),
    EvalCase("reasoning_stays_on_topic_briefly", "In one short sentence, what does a compiler do?",
              lambda r: len(r) < 400 and "compil" in r.lower(), live=True),
]


def run_fast(memory_path: str) -> list[dict]:
    reg = _fast_registry(memory_path)
    results = []
    for case in FAST_CASES:
        skill = reg.find_skill(case.prompt)
        reply = skill.handle(case.prompt) if skill is not None else None
        passed = case.check(reply)
        results.append({"name": case.name, "prompt": case.prompt, "reply": reply, "passed": passed})
    return results


def run_live() -> list[dict]:
    sk = ReasoningSkill()
    if not sk.available:
        return [{"name": "(skipped — Ollama not reachable)", "prompt": "", "reply": None, "passed": None}]
    results = []
    for case in LIVE_CASES:
        t0 = time.monotonic()
        reply = sk.handle(case.prompt)
        elapsed = time.monotonic() - t0
        passed = case.check(reply)
        results.append({"name": case.name, "prompt": case.prompt, "reply": reply,
                         "passed": passed, "elapsed_s": round(elapsed, 1)})
    return results


def _report(results: list[dict]) -> bool:
    all_ok = True
    for r in results:
        if r["passed"] is None:
            print(f"  SKIP  {r['name']}")
            continue
        status = "PASS" if r["passed"] else "FAIL"
        all_ok = all_ok and r["passed"]
        extra = f" ({r['elapsed_s']}s)" if "elapsed_s" in r else ""
        print(f"  {status}  {r['name']}{extra}")
        if not r["passed"]:
            print(f"        prompt: {r['prompt']!r}")
            print(f"        reply:  {str(r['reply'])[:200]!r}")
    return all_ok


def main() -> int:
    import argparse
    import tempfile
    ap = argparse.ArgumentParser(description="JARVIS behavior eval — fixed regression checks")
    ap.add_argument("--live", action="store_true", help="also run LIVE_CASES against the real reasoning model")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        print("Fast cases (skill routing, no live model):")
        fast_ok = _report(run_fast(f"{tmp}/eval_memory.json"))

    live_ok = True
    if args.live:
        print("\nLive cases (real reasoning model, needs `ollama serve`):")
        live_ok = _report(run_live())

    total_ok = fast_ok and live_ok
    print(f"\n{'ALL PASSED' if total_ok else 'SOME FAILED'}")
    return 0 if total_ok else 1


if __name__ == "__main__":
    sys.exit(main())
