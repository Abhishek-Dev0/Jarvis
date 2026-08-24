"""main_window.py — the JARVIS desktop window: persona switch, a sidebar of
live on/off capability toggles, and four tabs (Chat, Terminal, Self-Modify,
Status). Every action here drives the same Jarvis instance / self_modify
module functions the console frontend and chat commands already use —
nothing here re-implements JARVIS logic, it's a window on top of it."""

from __future__ import annotations

import html
import json
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSplitter, QTabWidget, QTextEdit, QToolBar, QVBoxLayout,
    QWidget,
)

from .workers import CommandWorker, ListenWorker, RespondWorker, SpeakWorker

try:
    from .. import self_modify
except ImportError:  # pragma: no cover - legacy direct execution
    import self_modify

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_GUI_DIR)
_REPO_ROOT = os.path.dirname(_PKG_DIR)

_ROLE_COLOR = {"you": "#4fc1ff", "jarvis": "#4ec9b0", "system": "#8a8a8a", "error": "#f14c4c"}


def _load_prefs(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_prefs(path: str, prefs: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)


class MainWindow(QMainWindow):
    def __init__(self, jarvis, gui_output, prefs_path: str):
        super().__init__()
        self.jarvis = jarvis
        self.gui_output = gui_output
        self.prefs_path = prefs_path
        self.prefs = _load_prefs(prefs_path)

        self._respond_worker = None
        self._listen_worker = None
        self._speak_worker = None
        self._command_worker = None
        self._streaming_open = False
        self._auto_speak = self.jarvis.persona_engine is not None

        self._apply_saved_skill_toggles()

        self.setWindowTitle("Jarvis")
        self.resize(1150, 760)
        self._build_ui()
        self._wire_output_signals()

    # ------------------------------------------------------------- layout

    def _build_ui(self):
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        title = QLabel("JARVIS")
        title.setObjectName("TitleLabel")
        toolbar.addWidget(title)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Persona:"))
        self.persona_combo = QComboBox()
        self.persona_combo.addItems(["Jarvis", "Eve"])
        self.persona_combo.currentTextChanged.connect(self._on_persona_changed)
        toolbar.addWidget(self.persona_combo)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        splitter.addWidget(self._build_sidebar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chat_tab(), "Chat")
        self.tabs.addTab(self._build_terminal_tab(), "Terminal")
        self.tabs.addTab(self._build_self_modify_tab(), "Self-Modify")
        self.tabs.addTab(self._build_status_tab(), "Status")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 870])

        self.statusBar().showMessage("Ready")

    def _build_sidebar(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Capabilities")
        header.setStyleSheet("font-weight:600; padding:8px;")
        layout.addWidget(header)

        self.skill_list = QListWidget()
        self.skill_list.setObjectName("SkillList")  # see theme.py's #SkillList::item override
        layout.addWidget(self.skill_list)
        self._skill_checkboxes = {}
        # Matches the sidebar's initial splitter width (280px, set in
        # _build_ui) minus its margins/scrollbar. A QLabel's sizeHint() for
        # wrapped text is only accurate once it knows how wide it'll
        # actually render -- an unconstrained label sizeHints itself as if
        # on one line. Pinning the width makes heightForWidth resolve
        # deterministically before setSizeHint() below reads it.
        _DESC_WIDTH = 240
        for skill in self.jarvis.registry.skills:
            item = QListWidgetItem(self.skill_list)
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(8, 8, 8, 10)
            row_layout.setSpacing(4)
            cb = QCheckBox(skill.name)
            cb.setChecked(skill.enabled)
            cb.toggled.connect(lambda checked, s=skill: self._on_skill_toggled(s, checked))
            row_layout.addWidget(cb)
            desc = QLabel(skill.description)
            desc.setObjectName("SkillDescription")
            desc.setWordWrap(True)
            desc.setFixedWidth(_DESC_WIDTH)
            row_layout.addWidget(desc)
            item.setSizeHint(row.sizeHint())
            self.skill_list.addItem(item)
            self.skill_list.setItemWidget(item, row)
            self._skill_checkboxes[skill.name] = cb
        return container

    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        layout.addWidget(self.chat_view)

        row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Message Jarvis...")
        self.chat_input.returnPressed.connect(self._on_send_clicked)
        row.addWidget(self.chat_input)

        self.mic_button = QPushButton("Mic")
        self.mic_button.setObjectName("MicButton")
        self.mic_button.setFixedWidth(56)
        self.mic_button.clicked.connect(self._on_mic_clicked)
        self.mic_button.setEnabled(self.jarvis.whisper_engine is not None)
        if self.jarvis.whisper_engine is None:
            self.mic_button.setToolTip("Voice input unavailable on this launch")
        row.addWidget(self.mic_button)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send_clicked)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        return widget

    def _build_terminal_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Cascadia Mono", 10))
        layout.addWidget(self.terminal_output)

        row = QHBoxLayout()
        self.terminal_input = QLineEdit()
        self.terminal_input.setPlaceholderText(f"Command, runs in {_REPO_ROOT}...")
        self.terminal_input.returnPressed.connect(self._on_run_command)
        row.addWidget(self.terminal_input)
        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self._on_run_command)
        row.addWidget(run_btn)
        layout.addLayout(row)
        return widget

    def _build_self_modify_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        split = QSplitter(Qt.Orientation.Vertical)
        self.proposal_list = QListWidget()
        self.proposal_list.currentItemChanged.connect(self._on_proposal_selected)
        split.addWidget(self.proposal_list)
        self.proposal_diff = QTextEdit()
        self.proposal_diff.setReadOnly(True)
        self.proposal_diff.setFont(QFont("Cascadia Mono", 10))
        split.addWidget(self.proposal_diff)
        split.setSizes([160, 400])
        layout.addWidget(split)

        row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_proposals)
        row.addWidget(refresh_btn)
        row.addStretch()
        self.approve_btn = QPushButton("Approve")
        self.approve_btn.clicked.connect(self._on_approve_clicked)
        row.addWidget(self.approve_btn)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self._on_reject_clicked)
        row.addWidget(self.reject_btn)
        self.commit_btn = QPushButton("Commit && Push to GitHub")
        self.commit_btn.clicked.connect(self._on_commit_clicked)
        row.addWidget(self.commit_btn)
        layout.addLayout(row)

        self._refresh_proposals()
        return widget

    def _build_status_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.status_view = QTextEdit()
        self.status_view.setReadOnly(True)
        self.status_view.setFont(QFont("Cascadia Mono", 10))
        layout.addWidget(self.status_view)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_status)
        layout.addWidget(refresh_btn)
        self._refresh_status()
        return widget

    # ------------------------------------------------------------ wiring

    def _wire_output_signals(self):
        self.gui_output.message_ready.connect(self._on_side_message)
        self.gui_output.chunk_ready.connect(self._on_chunk)
        self.gui_output.stream_finished.connect(self._on_stream_finished)

    # ------------------------------------------------------- skill toggles

    def _apply_saved_skill_toggles(self):
        saved = self.prefs.get("skills", {})
        for skill in self.jarvis.registry.skills:
            if skill.name in saved:
                skill.enabled = bool(saved[skill.name])

    def _on_skill_toggled(self, skill, checked: bool):
        skill.enabled = checked
        self.prefs.setdefault("skills", {})[skill.name] = checked
        _save_prefs(self.prefs_path, self.prefs)

    # ------------------------------------------------------------ persona

    def _on_persona_changed(self, text: str):
        persona = text.strip().lower()
        if hasattr(self.jarvis, "_switch_persona"):
            self.jarvis._switch_persona(persona)

    # --------------------------------------------------------------- chat

    def _append_message(self, role: str, text: str):
        color = _ROLE_COLOR.get(role, "#cccccc")
        label = {"you": "You", "jarvis": self.persona_combo.currentText(),
                  "system": "System", "error": "Error"}.get(role, role)
        safe = html.escape(text).replace("\n", "<br>")
        self.chat_view.append(f'<p><b style="color:{color}">{label}:</b> {safe}</p>')

    def _on_side_message(self, text: str):
        self._append_message("system", text)

    def _on_chunk(self, chunk: str):
        if not self._streaming_open:
            self._streaming_open = True
            label = self.persona_combo.currentText()
            color = _ROLE_COLOR["jarvis"]
            self.chat_view.append(f'<b style="color:{color}">{label}:</b> ')
        cursor = self.chat_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(chunk)
        self.chat_view.setTextCursor(cursor)
        self.chat_view.ensureCursorVisible()

    def _on_stream_finished(self):
        self._streaming_open = False

    def _set_chat_busy(self, busy: bool):
        self.send_button.setEnabled(not busy)
        self.chat_input.setEnabled(not busy)
        self.statusBar().showMessage("Thinking..." if busy else "Ready")

    def _on_send_clicked(self):
        text = self.chat_input.text().strip()
        if not text or self._respond_worker is not None:
            return
        self.chat_input.clear()
        self._append_message("you", text)
        self._set_chat_busy(True)
        self._streaming_open = False

        worker = RespondWorker(self.jarvis, text)
        worker.finished_ok.connect(self._on_respond_finished)
        worker.failed.connect(self._on_respond_failed)
        worker.finished.connect(lambda: self._clear_worker("respond"))
        self._respond_worker = worker
        worker.start()

    def _on_respond_finished(self, reply: str):
        if not self._streaming_open and reply:
            self._append_message("jarvis", reply)
        self._streaming_open = False
        self._set_chat_busy(False)
        if self._auto_speak and reply:
            self._speak(reply)

    def _on_respond_failed(self, error: str):
        self._streaming_open = False
        self._append_message("error", error)
        self._set_chat_busy(False)

    def _clear_worker(self, kind: str):
        setattr(self, f"_{kind}_worker", None)

    # --------------------------------------------------------------- mic

    def _on_mic_clicked(self):
        if self._listen_worker is not None or self.jarvis.whisper_engine is None:
            return
        self.mic_button.setProperty("recording", True)
        self.mic_button.style().unpolish(self.mic_button)
        self.mic_button.style().polish(self.mic_button)
        self.mic_button.setEnabled(False)
        self.statusBar().showMessage("Listening...")

        worker = ListenWorker(self.jarvis.whisper_engine)
        worker.transcribed.connect(self._on_transcribed)
        worker.failed.connect(self._on_listen_failed)
        worker.finished.connect(lambda: self._clear_worker("listen"))
        self._listen_worker = worker
        worker.start()

    def _reset_mic_button(self):
        self.mic_button.setProperty("recording", False)
        self.mic_button.style().unpolish(self.mic_button)
        self.mic_button.style().polish(self.mic_button)
        self.mic_button.setEnabled(self.jarvis.whisper_engine is not None)
        self.statusBar().showMessage("Ready")

    def _on_transcribed(self, text: str):
        self._reset_mic_button()
        text = text.strip()
        if not text:
            return
        self.chat_input.setText(text)
        self._on_send_clicked()

    def _on_listen_failed(self, error: str):
        self._reset_mic_button()
        self._append_message("error", f"mic: {error}")

    def _speak(self, text: str):
        if self.jarvis.persona_engine is None:
            return
        worker = SpeakWorker(self.jarvis.persona_engine, text)
        worker.finished.connect(lambda: self._clear_worker("speak"))
        self._speak_worker = worker
        worker.start()

    # ---------------------------------------------------------- terminal

    def _on_run_command(self):
        cmd = self.terminal_input.text().strip()
        if not cmd or self._command_worker is not None:
            return
        self.terminal_output.appendPlainText(f"> {cmd}")
        self.terminal_input.clear()

        worker = CommandWorker(cmd, cwd=_REPO_ROOT)
        worker.line_ready.connect(self.terminal_output.appendPlainText)
        worker.finished_ok.connect(lambda code: self.terminal_output.appendPlainText(f"[exit {code}]"))
        worker.failed.connect(lambda err: self.terminal_output.appendPlainText(f"[error] {err}"))
        worker.finished.connect(lambda: self._clear_worker("command"))
        self._command_worker = worker
        worker.start()

    # ------------------------------------------------------- self-modify

    def _refresh_proposals(self):
        self.proposal_list.clear()
        for p in self_modify.list_proposals():
            item = QListWidgetItem(f"[{p['status']}] {p['id']} — {p['target_path']}")
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            self.proposal_list.addItem(item)

    def _selected_proposal(self):
        item = self.proposal_list.currentItem()
        if item is None:
            return None
        return self_modify.load_proposal(item.data(Qt.ItemDataRole.UserRole))

    def _on_proposal_selected(self, *_args):
        p = self._selected_proposal()
        if p is None:
            self.proposal_diff.clear()
            return
        lines = [f"Proposal {p['id']} [{p['status']}] — {p['target_path']}",
                 f"Instruction: {p['instruction']}", ""]
        if p.get("diff"):
            lines.append(p["diff"])
        if p.get("reason"):
            lines.append(f"Reason: {p['reason']}")
        if p.get("test_output"):
            lines.append("--- test output ---")
            lines.append(p["test_output"])
        self.proposal_diff.setPlainText("\n".join(lines))

    def _prompt_passphrase(self, reason: str) -> str:
        text, ok = QInputDialog.getText(self, "Admin verification", f"Passphrase for: {reason}",
                                         QLineEdit.EchoMode.Password)
        return text if ok else ""

    def _security_refs(self, reason: str):
        return (lambda: self.jarvis.security, lambda: self.jarvis.is_admin,
                lambda: self._prompt_passphrase(reason))

    def _on_approve_clicked(self):
        p = self._selected_proposal()
        if p is None:
            return
        security_ref, is_admin_ref, provider = self._security_refs(f"apply proposal {p['id']}")
        result = self_modify.approve(p["id"], security_ref, is_admin_ref, passphrase_provider=provider)
        QMessageBox.information(self, "Approve", result)
        self._refresh_proposals()

    def _on_reject_clicked(self):
        p = self._selected_proposal()
        if p is None:
            return
        result = self_modify.reject(p["id"])
        QMessageBox.information(self, "Reject", result)
        self._refresh_proposals()

    def _on_commit_clicked(self):
        p = self._selected_proposal()
        if p is None:
            return
        if p["status"] != "applied":
            QMessageBox.warning(self, "Commit & Push", "Approve this proposal first.")
            return
        security_ref, is_admin_ref, provider = self._security_refs(f"commit proposal {p['id']} to GitHub")
        result = self_modify.commit_and_push(p["id"], security_ref, is_admin_ref, passphrase_provider=provider)
        QMessageBox.information(self, "Commit & Push", result)
        self._refresh_proposals()

    # ------------------------------------------------------------- status

    def _refresh_status(self):
        health = None
        for skill in self.jarvis.registry.skills:
            if skill.name == "health":
                health = skill
                break
        if health is None:
            self.status_view.setPlainText("Health-check skill not loaded.")
            return
        self.status_view.setPlainText(health.handle("system status"))
