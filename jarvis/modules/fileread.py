"""
fileread.py — read-only access to this project's own files. From the
2026-08-22 systems audit (P1): self_modify.py only reads the one file it's
actively drafting a patch for; there was no general "read/list files in
the repo" capability for a human (or, via the reasoning model's tool
loop, JARVIS itself) to just look at something.

Read-only, scoped to the repo root, and explicitly excludes anything under
_EXCLUDED_PREFIXES even though those paths would resolve fine — not
because path traversal into them would leak something unreadable (the
security files are DPAPI-encrypted, the vendor/ directory is just large
binaries), but because "the AI can casually read anything in the project"
and "the AI can read your credential store" are different guarantees, and
this keeps them different on purpose, matching self_modify.py's
PROTECTED_PATHS reasoning.
"""

from __future__ import annotations

import fnmatch
import os

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_PKG_DIR)

_EXCLUDED_PREFIXES = (
    "jarvis/data/security",
    "jarvis/data/memory.json",
    "jarvis/data/logs",
    "vendor",
    "venv",
    ".venv",
    ".git",
)

_MAX_READ_CHARS = 12_000
_MAX_LISTED = 200


def _norm_repo_path(path: str) -> str:
    abspath = os.path.abspath(os.path.join(_REPO_ROOT, path)) if not os.path.isabs(path) \
        else os.path.abspath(path)
    rel = os.path.relpath(abspath, _REPO_ROOT)
    return rel.replace("\\", "/")


def _is_excluded(rel_path: str) -> bool:
    return any(rel_path == p or rel_path.startswith(p + "/") for p in _EXCLUDED_PREFIXES)


def read_file(path: str, max_chars: int = _MAX_READ_CHARS) -> dict:
    """Returns {ok, path, content, reason}. Never raises — callers (a
    human via the skill, or the model via a future tool binding) get a
    clear reason string instead of a traceback."""
    rel = _norm_repo_path(path)
    if rel.startswith(".."):
        return {"ok": False, "path": rel, "reason": "path escapes the repository"}
    if _is_excluded(rel):
        return {"ok": False, "path": rel, "reason": f"'{rel}' is excluded from file reads"}
    abspath = os.path.join(_REPO_ROOT, rel)
    if not os.path.isfile(abspath):
        return {"ok": False, "path": rel, "reason": "no such file"}
    try:
        with open(abspath, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
    except Exception as e:
        return {"ok": False, "path": rel, "reason": str(e)}
    truncated = len(content) > max_chars
    return {"ok": True, "path": rel, "content": content[:max_chars], "truncated": truncated}


def list_files(pattern: str = "**/*.py", max_results: int = _MAX_LISTED) -> list[str]:
    """Repo-relative paths matching `pattern` (fnmatch-style, ** for any
    depth), excluding _EXCLUDED_PREFIXES. Capped at max_results — silently
    truncated results would misrepresent completeness, so the skill layer
    reports when the cap was hit rather than pretending this is exhaustive."""
    matches = []
    for root, dirs, files in os.walk(_REPO_ROOT):
        rel_root = _norm_repo_path(root)
        if rel_root != "." and _is_excluded(rel_root):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if not _is_excluded(_norm_repo_path(os.path.join(root, d)))]
        for fname in files:
            rel = _norm_repo_path(os.path.join(root, fname))
            if fnmatch.fnmatch(rel, pattern):
                matches.append(rel)
                if len(matches) >= max_results:
                    return matches
    return matches


_READ_PREFIXES = ("read file", "show me file", "show me")
_LIST_PREFIXES = ("list files matching", "list files")


class FileReadSkill(SkillModule):
    """"read file <path>" / "list files matching <pattern>" — ungated,
    read-only, repo-scoped. See the module docstring for what's excluded
    and why."""

    name = "fileread"
    description = "reads a project file or lists files by pattern (read-only, repo-scoped)"
    priority = 8

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) for p in (*_READ_PREFIXES, *_LIST_PREFIXES))

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()

        for prefix in sorted(_LIST_PREFIXES, key=len, reverse=True):
            if low.startswith(prefix):
                pattern = t[len(prefix):].strip() or "**/*.py"
                files = list_files(pattern)
                if not files:
                    return f"No files matching '{pattern}'."
                hit_cap = len(files) >= _MAX_LISTED
                lines = [f"{len(files)}{'+' if hit_cap else ''} file(s) matching '{pattern}'"
                         + (f" (showing first {_MAX_LISTED})" if hit_cap else "") + ":"]
                lines.extend(f"  - {f}" for f in files)
                return "\n".join(lines)

        for prefix in sorted(_READ_PREFIXES, key=len, reverse=True):
            if low.startswith(prefix):
                path = t[len(prefix):].strip()
                if not path:
                    return "Read which file?"
                result = read_file(path)
                if not result["ok"]:
                    return f"Couldn't read '{result['path']}': {result['reason']}"
                header = f"{result['path']}" + (" (truncated)" if result["truncated"] else "") + ":\n"
                return header + result["content"]

        return "I didn't catch which file(s) you meant."
