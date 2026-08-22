import json

from jarvis import self_modify


def test_is_protected_flags_security_and_self_modify():
    assert self_modify.is_protected("jarvis/security.py") is True
    assert self_modify.is_protected("jarvis/self_modify.py") is True


def test_is_protected_allows_ordinary_module():
    assert self_modify.is_protected("jarvis/modules/os_control.py") is False


def test_extract_patch_well_formed():
    raw = "<<<FILE>>>\ndef f():\n    return 1\n<<<END>>>"
    extracted = self_modify._extract_patch(raw)
    assert extracted == "def f():\n    return 1"


def test_extract_patch_handles_real_malformed_model_output():
    # Actual qwen2.5:3b output observed during development: one-bracket-short
    # start marker, "###\nEND" instead of a real end marker, then garbage
    # repeated NO_FIX tokens appended after a genuinely correct fix.
    raw = ('<<<FILE>\ndef add_one(n):\n    """docstring"""\n    return n + 1\n'
           '###\nEND\n<<<NO_FIX>><<<<NO_FIX>>>')
    extracted = self_modify._extract_patch(raw)
    assert extracted == 'def add_one(n):\n    """docstring"""\n    return n + 1'
    compile(extracted, "test", "exec")  # must be valid Python


def test_extract_patch_returns_none_with_no_file_marker():
    assert self_modify._extract_patch("I can't fix this, sorry.") is None


def test_extract_patch_strips_markdown_fence():
    raw = "<<<FILE>>>\n```python\ndef f():\n    return 1\n```\n<<<END>>>"
    extracted = self_modify._extract_patch(raw)
    assert extracted == "def f():\n    return 1"


def test_issue_signature_is_deterministic_and_source_sensitive():
    a = {"source": "turn:web_search", "message": "TimeoutError: x"}
    b = {"source": "turn:web_search", "message": "TimeoutError: x"}
    c = {"source": "turn:reasoning", "message": "TimeoutError: x"}
    assert self_modify._issue_signature(a) == self_modify._issue_signature(b)
    assert self_modify._issue_signature(a) != self_modify._issue_signature(c)


def test_log_issue_and_scan_round_trip(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(self_modify, "_LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(self_modify, "_ISSUES_LOG", str(logs_dir / "issues.jsonl"))
    monkeypatch.setattr(self_modify, "_ADDRESSED_PATH", str(logs_dir / "addressed.json"))

    self_modify.log_issue("test:source", "something broke", file="jarvis/modules/web.py")
    issues = self_modify.scan_recent_issues()
    assert len(issues) == 1
    assert issues[0]["message"] == "something broke"
    assert issues[0]["file"] == "jarvis/modules/web.py"

    # marking it addressed removes it from future scans
    self_modify._mark_addressed(issues[0]["_signature"])
    assert self_modify.scan_recent_issues() == []


def test_scan_dedupes_repeated_identical_issues(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    monkeypatch.setattr(self_modify, "_LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(self_modify, "_ISSUES_LOG", str(logs_dir / "issues.jsonl"))
    monkeypatch.setattr(self_modify, "_ADDRESSED_PATH", str(logs_dir / "addressed.json"))

    for _ in range(5):
        self_modify.log_issue("test:source", "same error every time", file="jarvis/modules/web.py")
    issues = self_modify.scan_recent_issues()
    assert len(issues) == 1  # 5 identical occurrences collapse to one signature


def test_propose_patch_refuses_protected_target():
    result = self_modify.propose_patch("jarvis/security.py", "make it worse")
    assert result["ok"] is False
    assert "protected" in result["reason"]


def test_approve_denies_without_authorization(tmp_path, monkeypatch):
    proposals_dir = tmp_path / "proposals"
    monkeypatch.setattr(self_modify, "_PROPOSALS_DIR", str(proposals_dir))
    proposal = {"id": "abc123", "status": "pending", "target_path": "jarvis/modules/web.py",
                "proposed": "x = 1", "diff": "", "instruction": "test", "source": "on_demand",
                "created": "now"}
    self_modify._save_proposal(proposal)

    result = self_modify.approve("abc123", security_ref=None, is_admin_ref=None)
    assert "Denied" in result
    assert self_modify.load_proposal("abc123")["status"] == "pending"  # unchanged
