import json
import os

from jarvis import telemetry


def test_log_event_writes_a_jsonl_record(tmp_path):
    path = str(tmp_path / "events.jsonl")
    telemetry.log_event("calculator", "handle_turn", duration_ms=12.345, outcome="ok", path=path)

    with open(path, encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["component"] == "calculator"
    assert record["event"] == "handle_turn"
    assert record["severity"] == "info"
    assert record["duration_ms"] == 12.3  # rounded to 1 decimal
    assert record["outcome"] == "ok"
    assert "time" in record


def test_log_event_includes_extra_detail_fields(tmp_path):
    path = str(tmp_path / "events.jsonl")
    telemetry.log_event("reasoning", "handle_turn", severity="error",
                         outcome="error", path=path, error="ConnectionError")

    with open(path, encoding="utf-8") as f:
        record = json.loads(f.readline())
    assert record["severity"] == "error"
    assert record["detail"] == {"error": "ConnectionError"}


def test_log_event_never_raises_on_bad_path():
    # a directory that can't plausibly be created/written to
    telemetry.log_event("x", "y", path="\0invalid\0path")  # must not raise


def test_read_events_returns_empty_list_when_file_missing(tmp_path):
    path = str(tmp_path / "nonexistent.jsonl")
    assert telemetry.read_events(path) == []


def test_read_events_returns_most_recent_first(tmp_path):
    path = str(tmp_path / "events.jsonl")
    telemetry.log_event("a", "first", path=path)
    telemetry.log_event("b", "second", path=path)
    telemetry.log_event("c", "third", path=path)

    events = telemetry.read_events(path)
    assert [e["event"] for e in events] == ["third", "second", "first"]


def test_read_events_respects_limit(tmp_path):
    path = str(tmp_path / "events.jsonl")
    for i in range(5):
        telemetry.log_event("x", f"event-{i}", path=path)

    events = telemetry.read_events(path, limit=2)
    assert len(events) == 2
    assert events[0]["event"] == "event-4"


def test_read_events_skips_corrupt_lines(tmp_path):
    path = str(tmp_path / "events.jsonl")
    telemetry.log_event("a", "good-one", path=path)
    with open(path, "a", encoding="utf-8") as f:
        f.write("not valid json\n")
    telemetry.log_event("b", "good-two", path=path)

    events = telemetry.read_events(path)
    assert [e["event"] for e in events] == ["good-two", "good-one"]
