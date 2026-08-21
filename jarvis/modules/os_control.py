"""
os_control.py — OS-level agentic control: open/close applications, list what's
running. This is the actual "OS-agentic control" item from the backlog (see
the JARVIS roadmap notes) — the part of Abi's 2026-08-22 ask that means
"automate my own machine," not the "hack into it" phrasing he also used,
which was explicitly scoped down to that same meaning, not unauthorized
access to anything he doesn't control.

Every action that changes system state (launching or killing a process) goes
through SecurityGate.authorize() first, same as shutdown — this is exactly
the hook point security.py was built for and left unused until now. Read-only
actions (listing what's running) are not gated; there's nothing to protect
against there.

Windows-only (os.startfile, taskkill-equivalent via psutil) — same platform
assumption as the rest of this repo.
"""

from __future__ import annotations

import os
import re

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

# Friendly name -> real executable. Anything not listed here is tried as
# typed (with .exe appended if missing) via os.startfile, which resolves
# through Windows' "App Paths" registry the same way the Start menu does —
# works for most installed apps without needing every one hardcoded.
_APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "files": "explorer.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "vscode": "code.exe",
    "vs code": "code.exe",
    "visual studio code": "code.exe",
    "code": "code.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "powershell": "powershell.exe",
    "terminal": "wt.exe",
    "task manager": "taskmgr.exe",
    "control panel": "control.exe",
    "word": "winword.exe",
    "excel": "excel.exe",
    "spotify": "spotify.exe",
    "discord": "discord.exe",
    "steam": "steam.exe",
}

_LAUNCH_TRIGGERS = ("open ", "launch ", "start ", "run ")
_CLOSE_TRIGGERS = ("close ", "kill ", "terminate ", "quit ")
_LIST_TRIGGERS = {
    "list processes", "list running processes", "list running apps",
    "what's running", "whats running", "what is running",
    "show running processes", "show running apps",
}

_MAX_LISTED = 40


def _resolve_target(app: str) -> str:
    key = app.strip().lower()
    if key in _APP_ALIASES:
        return _APP_ALIASES[key]
    # bare word, no spaces, no extension -> assume it's already an exe stem
    if re.fullmatch(r"[\w\-]+", key) and not key.endswith(".exe"):
        return f"{key}.exe"
    return app.strip()


class OSControlSkill(SkillModule):
    """open/close/list applications on this machine — gated by SecurityGate
    for anything that isn't read-only."""

    name = "os_control"
    description = "opens, closes, and lists running applications (security-gated)"
    priority = 9  # below calculator (10); above web search/reasoning

    def __init__(self, security_ref=None, is_admin_ref=None):
        # Callables, not direct references — mirrors ReasoningSkill's
        # history_ref: main() may still swap Jarvis.security (e.g. once
        # --voice wires up a mic-backed gate) after this module is
        # constructed, so a stale direct reference would authorize against
        # the wrong gate. security_ref: () -> SecurityGate.
        # is_admin_ref: () -> bool, so an already-verified admin session
        # (see Jarvis.run()'s wake_challenge) doesn't have to re-authorize
        # every single action, same behavior as shutdown.
        self.security_ref = security_ref
        self.is_admin_ref = is_admin_ref

    def _authorized(self, reason: str) -> bool:
        if self.is_admin_ref is not None and self.is_admin_ref():
            return True
        if self.security_ref is None:
            return False
        return self.security_ref().authorize(reason)

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        if t in _LIST_TRIGGERS:
            return True
        return any(t.startswith(p) for p in (*_LAUNCH_TRIGGERS, *_CLOSE_TRIGGERS))

    def _strip_trigger(self, text: str, triggers: tuple[str, ...]) -> str | None:
        t = text.strip().lower()
        for p in triggers:
            if t.startswith(p):
                return text.strip()[len(p):].strip()
        return None

    def handle(self, text: str) -> str:
        t = text.strip().lower()

        if t in _LIST_TRIGGERS:
            return self._list_processes()

        app = self._strip_trigger(text, _LAUNCH_TRIGGERS)
        if app is not None:
            return self._launch(app)

        app = self._strip_trigger(text, _CLOSE_TRIGGERS)
        if app is not None:
            return self._close(app)

        return "I didn't catch which application you mean."

    # ------------------------------------------------------------------ list

    def _list_processes(self) -> str:
        import psutil
        names = sorted({p.info["name"] for p in psutil.process_iter(["name"]) if p.info["name"]})
        if not names:
            return "Couldn't read the process list."
        shown = names[:_MAX_LISTED]
        lines = [f"{len(names)} processes running" +
                 (f" (showing first {_MAX_LISTED}):" if len(names) > _MAX_LISTED else ":")]
        lines.extend(f"  - {n}" for n in shown)
        return "\n".join(lines)

    # ---------------------------------------------------------------- launch

    def _launch(self, app: str) -> str:
        if not app:
            return "Open what?"
        if not self._authorized(f"open {app}"):
            return "Denied — couldn't verify you for launching an application."
        target = _resolve_target(app)
        try:
            os.startfile(target)  # noqa: S606 - deliberate, gated above
            return f"Opened {app}."
        except OSError as e:
            return f"Couldn't open '{app}' ({e}). Is it installed, and spelled the way Windows knows it?"

    # ----------------------------------------------------------------- close

    def _close(self, app: str) -> str:
        if not app:
            return "Close what?"
        if not self._authorized(f"close {app}"):
            return "Denied — couldn't verify you for closing an application."
        import psutil
        needle = app.strip().lower().removesuffix(".exe")
        killed = []
        for proc in psutil.process_iter(["pid", "name"]):
            name = (proc.info["name"] or "").lower().removesuffix(".exe")
            if needle and needle in name:
                try:
                    proc.terminate()
                    killed.append(proc.info["name"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        if not killed:
            return f"Nothing running matched '{app}'."
        return f"Closed: {', '.join(sorted(set(killed)))}."
