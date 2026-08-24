import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # headless-safe before any Qt import

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from jarvis.gui.io_adapter import GuiOutput
from jarvis.gui.main_window import MainWindow
from jarvis.gui.theme import STYLESHEET
from jarvis.modules.base import Registry, SkillModule


@pytest.fixture(scope="module")
def qapp():
    # Applies the real theme stylesheet -- not cosmetic here: a prior real
    # bug (QListWidget::item's global padding silently shrinking the
    # sidebar's custom row widgets below their own sizeHint, clipping
    # descriptions) only reproduces with the actual stylesheet applied, the
    # same way it only showed up in a real screenshot, not a plain widget
    # inspection. See test_sidebar_row_widgets_are_not_clipped_by_item_padding.
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLESHEET)
    yield app


class _FakeSkill(SkillModule):
    def __init__(self, name, description, priority=0, reply=None):
        self.name = name
        self.description = description
        self.priority = priority
        self._reply = reply or f"{name} handled"

    def matches(self, text):
        return text.strip().lower() == self.name

    def handle(self, text):
        return self._reply


class _FakeSecurity:
    def authorize(self, reason, passphrase_provider=None):
        return False


class _FakeJarvis:
    def __init__(self, skills):
        self.registry = Registry()
        for s in skills:
            self.registry.register(s)
        self.whisper_engine = None
        self.persona_engine = None
        self.security = _FakeSecurity()
        self.is_admin = False
        self.responses = {}

    def respond(self, text, stream=True):
        return self.responses.get(text, f"echo: {text}")

    def _switch_persona(self, persona):
        self.last_persona = persona


def _pump(ms=800):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def test_main_window_builds_sidebar_from_registered_skills(qapp, tmp_path):
    j = _FakeJarvis([_FakeSkill("calculator", "does math"),
                      _FakeSkill("health", "status check")])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))

    assert set(window._skill_checkboxes) == {"calculator", "health"}
    assert window._skill_checkboxes["calculator"].isChecked() is True


def test_toggling_sidebar_checkbox_disables_the_skill_live(qapp, tmp_path):
    calc = _FakeSkill("calculator", "does math")
    j = _FakeJarvis([calc])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))

    assert j.registry.find_skill("calculator") is calc
    window._skill_checkboxes["calculator"].setChecked(False)
    assert calc.enabled is False
    assert j.registry.find_skill("calculator") is None


def test_toggle_state_persists_to_prefs_file(qapp, tmp_path):
    calc = _FakeSkill("calculator", "does math")
    j = _FakeJarvis([calc])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    prefs_path = str(tmp_path / "prefs.json")
    window = MainWindow(j, gui_output, prefs_path)

    window._skill_checkboxes["calculator"].setChecked(False)
    with open(prefs_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["skills"]["calculator"] is False


def test_saved_prefs_are_applied_on_next_launch(qapp, tmp_path):
    prefs_path = str(tmp_path / "prefs.json")
    with open(prefs_path, "w", encoding="utf-8") as f:
        json.dump({"skills": {"calculator": False}}, f)

    calc = _FakeSkill("calculator", "does math")
    j = _FakeJarvis([calc])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    MainWindow(j, gui_output, prefs_path)

    assert calc.enabled is False


def test_chat_round_trip_shows_user_and_reply(qapp, tmp_path):
    j = _FakeJarvis([])
    j.responses["hello"] = "hi there"
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))

    window.chat_input.setText("hello")
    window._on_send_clicked()
    _pump()

    text = window.chat_view.toPlainText()
    assert "hello" in text
    assert "hi there" in text


def test_status_tab_renders_health_skill_output(qapp, tmp_path):
    health = _FakeSkill("health", "status check", reply="all systems nominal")
    j = _FakeJarvis([health])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))

    window._refresh_status()
    assert "all systems nominal" in window.status_view.toPlainText()


def test_sidebar_row_widgets_are_not_clipped_by_item_padding(qapp, tmp_path):
    # Regression test for a real bug found via an actual screenshot: the
    # theme's global "QListWidget::item { padding: 6px 8px; }" rule was
    # stacking on top of each sidebar row's own margins, shrinking the
    # widget below what item.setSizeHint() had reported and clipping the
    # last line of any multi-line description (e.g. self_modify's). Fixed
    # via the #SkillList::item override in theme.py. Assert the actual
    # allocated widget height matches the item's slot height (within the
    # 1px divider border), for a description long enough to wrap 3+ lines.
    long_desc = ("drafts, sandbox-tests, and (with approval) applies code changes to JARVIS "
                 "itself, then (with a second approval) commits and pushes them to GitHub")
    skill = _FakeSkill("self_modify", long_desc)
    j = _FakeJarvis([skill])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))
    window.show()
    _pump(200)

    item = window.skill_list.item(0)
    row = window.skill_list.itemWidget(item)
    allocated_height = window.skill_list.visualItemRect(item).height()
    assert row.height() >= allocated_height - 2  # within the 1px border


def test_mic_button_disabled_without_a_whisper_engine(qapp, tmp_path):
    j = _FakeJarvis([])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))

    assert window.mic_button.isEnabled() is False


def _make_window_with_proposal(qapp, tmp_path, monkeypatch, status="pending"):
    from jarvis import self_modify
    proposals_dir = tmp_path / "proposals"
    monkeypatch.setattr(self_modify, "_PROPOSALS_DIR", str(proposals_dir))
    proposal = {"id": "abc123", "status": status, "target_path": "some_file.py",
                "instruction": "test change", "source": "on_demand", "created": "now",
                "diff": "--- a\n+++ b\n"}
    self_modify._save_proposal(proposal)

    j = _FakeJarvis([])
    gui_output = GuiOutput()
    j.registry.outputs.append(gui_output)
    window = MainWindow(j, gui_output, str(tmp_path / "prefs.json"))
    window.proposal_list.setCurrentRow(0)
    return window


def test_self_modify_tab_lists_and_selects_a_proposal(qapp, tmp_path, monkeypatch):
    window = _make_window_with_proposal(qapp, tmp_path, monkeypatch)
    assert window.proposal_list.count() == 1
    assert "test change" in window.proposal_diff.toPlainText()
    assert "--- a" in window.proposal_diff.toPlainText()


def test_approve_click_calls_self_modify_approve_with_selected_id(qapp, tmp_path, monkeypatch):
    from jarvis import self_modify
    window = _make_window_with_proposal(qapp, tmp_path, monkeypatch)
    monkeypatch.setattr("jarvis.gui.main_window.QMessageBox.information", lambda *a, **k: None)

    calls = []

    def fake_approve(pid, security_ref, is_admin_ref, passphrase_provider=None):
        calls.append(pid)
        return "Applied."

    monkeypatch.setattr(self_modify, "approve", fake_approve)
    window._on_approve_clicked()
    assert calls == ["abc123"]


def test_reject_click_calls_self_modify_reject_with_selected_id(qapp, tmp_path, monkeypatch):
    from jarvis import self_modify
    window = _make_window_with_proposal(qapp, tmp_path, monkeypatch)
    monkeypatch.setattr("jarvis.gui.main_window.QMessageBox.information", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(self_modify, "reject", lambda pid, reason="": calls.append(pid) or "Rejected.")
    window._on_reject_clicked()
    assert calls == ["abc123"]


def test_commit_click_refuses_a_proposal_that_isnt_applied_yet(qapp, tmp_path, monkeypatch):
    from jarvis import self_modify
    window = _make_window_with_proposal(qapp, tmp_path, monkeypatch, status="pending")
    monkeypatch.setattr("jarvis.gui.main_window.QMessageBox.warning", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(self_modify, "commit_and_push",
                         lambda pid, *a, **k: calls.append(pid) or "Committed.")
    window._on_commit_clicked()
    assert calls == []  # never called -- proposal wasn't "applied"


def test_commit_click_calls_self_modify_commit_and_push_once_applied(qapp, tmp_path, monkeypatch):
    from jarvis import self_modify
    window = _make_window_with_proposal(qapp, tmp_path, monkeypatch, status="applied")
    monkeypatch.setattr("jarvis.gui.main_window.QMessageBox.information", lambda *a, **k: None)

    calls = []
    monkeypatch.setattr(self_modify, "commit_and_push",
                         lambda pid, *a, **k: calls.append(pid) or "Committed and pushed.")
    window._on_commit_clicked()
    assert calls == ["abc123"]
