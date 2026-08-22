"""
telemetry.py — structured event log: one JSONL line per meaningful event
(skill invocation, tool call, self-modify action), each with a timestamp,
component, severity, and outcome. From the 2026-08-22 systems audit (P1).

Complements, doesn't replace, the two logging pieces already in this
project: self_modify.py's issues.jsonl (tracks *problems* worth drafting a
fix for) and the full-session transcript log (everything printed, for
pasting into a chat). This is a third, narrower thing — a queryable record
of what actually ran, how long it took, and whether it succeeded, without
re-parsing free-text console output.

Deliberately NOT a metrics/tracing stack. One JSONL file under data/logs/,
no server, no aggregation, no dashboards — that's the right scope for one
machine and one user; see the audit's Observability Plan for why the
heavier version isn't warranted here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_EVENTS_PATH = os.path.join(_PKG_DIR, "data", "logs", "events.jsonl")


def log_event(component: str, event: str, severity: str = "info",
              duration_ms: float | None = None, outcome: str | None = None,
              path: str | None = None, **detail) -> None:
    """Appends one structured event. Never raises — a telemetry failure
    must never be the reason a real request fails; same reasoning as
    self_modify.log_issue()."""
    path = path or _EVENTS_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "event": event,
            "severity": severity,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "outcome": outcome,
        }
        if detail:
            record["detail"] = detail
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def read_events(path: str | None = None, limit: int | None = None) -> list[dict]:
    """Most recent events first. Returns [] if nothing's been logged yet —
    never raises on a missing/corrupt file, same "never break the caller
    over telemetry" reasoning as log_event()."""
    path = path or _EVENTS_PATH
    if not os.path.exists(path):
        return []
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    events.reverse()
    return events[:limit] if limit else events
