"""paths.py — where JARVIS writes its own runtime data (memory, prefs,
security config, logs, caches, etc).

Real bug found in the packaged app: every module that persists state
independently computed its own path as `<jarvis package dir>/data/...`.
That's fine running from source, but once installed the package dir is
`C:\\Program Files\\Jarvis\\_internal\\jarvis\\`, which a standard
(non-admin) user cannot write to — the app crashed on first launch with
PermissionError trying to save gui_prefs.json, and would have hit the
same wall the moment memory, security/admin auth, self-modify, MCP
config, market watchlist, search index, design-engine output, or vision
events were touched.

user_data_dir()/user_data_path() keep the existing in-repo jarvis/data/
layout untouched when running from source (same path as before, nothing
for dev/tests to adjust to) and only redirect to a real per-user writable
directory (%LOCALAPPDATA%\\Jarvis on Windows, ~/.jarvis elsewhere) when
frozen by PyInstaller.
"""

from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _root() -> str:
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Jarvis")
    return os.path.join(_PKG_DIR, "data")


def user_data_dir(*parts: str) -> str:
    """A writable directory under the user data root; created if missing."""
    path = os.path.join(_root(), *parts)
    os.makedirs(path, exist_ok=True)
    return path


def user_data_path(*parts: str) -> str:
    """A writable file path under the user data root; parent dir created if missing."""
    path = os.path.join(_root(), *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
