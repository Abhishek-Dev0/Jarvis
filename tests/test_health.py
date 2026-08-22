from types import SimpleNamespace

from jarvis.modules.health import HealthCheckSkill


def test_matches_known_trigger_phrases():
    sk = HealthCheckSkill()
    assert sk.matches("system health") is True
    assert sk.matches("Health Check") is True
    assert sk.matches("tell me a joke") is False


def test_handle_without_any_refs_still_returns_a_report():
    sk = HealthCheckSkill()
    report = sk.handle("system health")
    assert report.startswith("System health:")
    assert "core model: unknown" in report
    assert "mcp: not registered" in report
    assert "hardware:" in report


def test_reports_model_loaded_state_from_jarvis_ref():
    fake_jarvis = SimpleNamespace(model=object(), is_admin=False)
    sk = HealthCheckSkill(jarvis_ref=lambda: fake_jarvis)
    report = sk.handle("system health")
    assert "core model: loaded" in report


def test_reports_model_not_loaded():
    fake_jarvis = SimpleNamespace(model=None, is_admin=False)
    sk = HealthCheckSkill(jarvis_ref=lambda: fake_jarvis)
    report = sk.handle("system health")
    assert "core model: NOT loaded" in report


def test_reports_admin_session_state():
    fake_jarvis = SimpleNamespace(model=None, is_admin=True)
    sk = HealthCheckSkill(jarvis_ref=lambda: fake_jarvis)
    report = sk.handle("system health")
    assert "admin this session" in report


def test_reports_connected_mcp_servers():
    fake_mcp = SimpleNamespace(list_servers=lambda: ["filesystem", "web"])
    sk = HealthCheckSkill(mcp_ref=lambda: fake_mcp)
    report = sk.handle("system health")
    assert "mcp servers: 2 connected (filesystem, web)" in report


def test_reports_no_mcp_servers_connected():
    fake_mcp = SimpleNamespace(list_servers=lambda: [])
    sk = HealthCheckSkill(mcp_ref=lambda: fake_mcp)
    report = sk.handle("system health")
    assert "mcp servers: none connected" in report


def test_ollama_check_reports_unreachable_when_nothing_is_listening(monkeypatch):
    def fake_get(url, timeout):
        raise ConnectionError("nope")
    monkeypatch.setattr("requests.get", fake_get)
    sk = HealthCheckSkill()
    report = sk.handle("system health")
    assert "ollama" in report and "NOT reachable" in report


def test_never_raises_even_if_every_subsystem_check_fails(monkeypatch):
    def raise_everywhere(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr("requests.get", raise_everywhere)

    def broken_jarvis_ref():
        raise RuntimeError("jarvis is gone")

    sk = HealthCheckSkill(jarvis_ref=broken_jarvis_ref)
    report = sk.handle("system health")  # must not raise
    assert "jarvis state: check failed" in report
    assert "System health:" in report  # rest of the report still assembled
