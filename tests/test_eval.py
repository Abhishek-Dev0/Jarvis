from jarvis.eval import EvalCase, _report, run_fast


def test_run_fast_returns_expected_case_count(tmp_path):
    results = run_fast(str(tmp_path / "eval_memory.json"))
    from jarvis.eval import FAST_CASES
    assert len(results) == len(FAST_CASES)


def test_run_fast_all_cases_pass_against_real_skills(tmp_path):
    # This is the actual regression guard the audit asked for: if a future
    # change to routing/priority/gating breaks one of these, this fails.
    results = run_fast(str(tmp_path / "eval_memory.json"))
    failures = [r for r in results if not r["passed"]]
    assert failures == [], f"eval regressions: {[f['name'] for f in failures]}"


def test_report_returns_true_when_everything_passes(capsys):
    results = [{"name": "a", "prompt": "x", "reply": "y", "passed": True}]
    assert _report(results) is True


def test_report_returns_false_on_any_failure(capsys):
    results = [
        {"name": "a", "prompt": "x", "reply": "y", "passed": True},
        {"name": "b", "prompt": "x", "reply": "wrong", "passed": False},
    ]
    assert _report(results) is False
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "b" in out


def test_report_handles_skipped_cases_without_affecting_pass_fail(capsys):
    results = [{"name": "skipped", "prompt": "", "reply": None, "passed": None}]
    assert _report(results) is True  # a skip alone shouldn't fail the run
    out = capsys.readouterr().out
    assert "SKIP" in out


def test_eval_case_holds_its_fields():
    case = EvalCase("name", "prompt", lambda r: True, live=True)
    assert case.name == "name"
    assert case.prompt == "prompt"
    assert case.live is True
    assert case.check("anything") is True
