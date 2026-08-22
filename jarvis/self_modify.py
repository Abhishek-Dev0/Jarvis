"""
self_modify.py — the self-modify pipeline: JARVIS drafts a patch, tests it
in an isolated sandbox, and stops. Nothing here ever writes to the live
working tree or commits to git without an explicit, gated human approval.

This implements the design agreed on 2026-08-15 and reaffirmed since (see
the JARVIS roadmap notes — full autonomy was explicitly declined, more than
once): scan for a problem -> draft a fix in an isolated environment -> run
the real test suite there -> only if it passes, surface it as a reviewable
proposal -> a human (gated by SecurityGate.authorize(), the same mechanism
every other irreversible action in this project uses) decides whether it
gets applied. Even on approval, this only stages the change into the real
working tree — it does NOT `git commit` or push. That's still a separate,
deliberate act, same as this codebase's own git-safety rules for me.

Scoped 2026-08-22 with Abi directly: autonomous scanning is in (a
background thread periodically checks jarvis/data/logs/issues.jsonl for new
problems and drafts proposals unattended — but "drafts and tests," never
"applies"), and PROTECTED_PATHS is non-negotiable — a self-modify system
that could rewrite the file that gates it, or its own approval logic,
would make the whole human-gated design meaningless. That's a structural
requirement for this feature to mean what it says, not an extra
restriction layered on top.

Known limitation, stated plainly rather than glossed over: this project's
test suite (tests/) is thin — two tests, as of this writing. Green tests
here are a real signal but not a strong one; they will not catch most
semantic regressions. Read every proposal's diff yourself before approving
it. This was flagged as a concern back when self-modify was first scoped
(2026-08-15) and is still true.

Patch generation uses the same local Ollama reasoning model as
reasoning.py (single-shot, no tool-calling — a plain "here's the file,
here's the problem, give me the corrected file back" prompt). Quality is
bounded by that model's coding ability, which is real but modest for its
size — most of the safety here comes from the sandbox+test+human-approval
pipeline, not from trusting the model to be right.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

try:
    from .modules.base import SkillModule
    from .security import authorize_action
except ImportError:  # pragma: no cover - legacy direct execution
    from modules.base import SkillModule
    from security import authorize_action

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
_LOGS_DIR = os.path.join(_PKG_DIR, "data", "logs")
_ISSUES_LOG = os.path.join(_LOGS_DIR, "issues.jsonl")
_ADDRESSED_PATH = os.path.join(_LOGS_DIR, "self_modify_addressed.json")
_PROPOSALS_DIR = os.path.join(_PKG_DIR, "data", "self_modify", "proposals")

# Repo-relative paths a generated patch may never target. A self-modify
# system that could edit the file gating its own merges, or the pipeline
# script itself, defeats the human-gated-merge design entirely — checked
# before a patch is even drafted, not just before applying it.
PROTECTED_PATHS = {"jarvis/security.py", "jarvis/self_modify.py"}


def _norm_repo_path(path: str) -> str:
    """Repo-relative, forward-slash form, for stable comparison against
    PROTECTED_PATHS and stable use as a dict/filename key."""
    abspath = os.path.abspath(os.path.join(_REPO_ROOT, path)) if not os.path.isabs(path) \
        else os.path.abspath(path)
    rel = os.path.relpath(abspath, _REPO_ROOT)
    return rel.replace("\\", "/")


def is_protected(path: str) -> bool:
    return _norm_repo_path(path) in PROTECTED_PATHS


# ------------------------------------------------------------------- logging

def log_issue(source: str, message: str, file: str | None = None, detail: str | None = None) -> None:
    """Append one issue record. `source` is where it came from ("runtime",
    "skill:web_search", ...); `file`, when known, is the repo-relative path
    most responsible — scan_recent_issues() uses it to pick propose_patch's
    target. Safe to call from anywhere; never raises (a logging failure
    should never crash whatever was already failing)."""
    try:
        os.makedirs(_LOGS_DIR, exist_ok=True)
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "message": message,
            "file": _norm_repo_path(file) if file else None,
            "detail": detail,
        }
        with open(_ISSUES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def log_exception(source: str, exc: BaseException) -> None:
    """Convenience over log_issue() for an actual caught exception: finds
    the deepest traceback frame inside this package (skipping stdlib/
    site-packages frames) and uses that as the issue's `file`, so
    autonomous scanning has a real target to draft a patch against."""
    import traceback
    tb = traceback.extract_tb(exc.__traceback__)
    file = None
    for frame in reversed(tb):
        if frame.filename.startswith(_PKG_DIR):
            file = frame.filename
            break
    log_issue(source, f"{type(exc).__name__}: {exc}", file=file,
              detail="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])


def _issue_signature(issue: dict) -> str:
    return hashlib.sha256(f"{issue.get('source')}|{issue.get('message')}".encode()).hexdigest()[:16]


def _load_addressed() -> set:
    if not os.path.exists(_ADDRESSED_PATH):
        return set()
    try:
        with open(_ADDRESSED_PATH, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def _mark_addressed(sig: str) -> None:
    addressed = _load_addressed()
    addressed.add(sig)
    os.makedirs(_LOGS_DIR, exist_ok=True)
    with open(_ADDRESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(addressed), f)


def scan_recent_issues() -> list[dict]:
    """Every logged issue not already turned into a proposal (pass or
    fail), deduped by (source, message) signature — a recurring error from
    the same place only needs one proposal attempt, not one per
    occurrence."""
    if not os.path.exists(_ISSUES_LOG):
        return []
    addressed = _load_addressed()
    seen_this_scan = set()
    out = []
    with open(_ISSUES_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                issue = json.loads(line)
            except json.JSONDecodeError:
                continue
            sig = _issue_signature(issue)
            if sig in addressed or sig in seen_this_scan:
                continue
            seen_this_scan.add(sig)
            issue["_signature"] = sig
            out.append(issue)
    return out


# --------------------------------------------------------------- patch drafting

_PATCH_SYSTEM_PROMPT = (
    "You are a careful Python code-fixing assistant. You will be given one "
    "complete Python file and a description of a problem with it. Reply with "
    "ONLY the complete corrected file content, wrapped exactly like this:\n"
    "<<<FILE>>>\n"
    "...the full corrected file, nothing omitted...\n"
    "<<<END>>>\n"
    "Do not explain your changes. Do not omit or abbreviate any part of the "
    "file (no '... rest unchanged ...'). If you cannot confidently fix the "
    "problem, reply with exactly: <<<NO_FIX>>>"
)


def _call_ollama(prompt: str, model: str, host: str, timeout: float = 120.0) -> str:
    import requests
    r = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={"model": model, "stream": False, "messages": [
            {"role": "system", "content": _PATCH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]


_FILE_START = re.compile(r"<{2,}\s*FILE\s*>{0,}\s*\n?", re.IGNORECASE)
# Any of these, wherever they occur first after the start marker, ends the
# captured content — a 3B model is not reliable about the exact literal
# "<<<END>>>"/"<<<NO_FIX>>>" delimiters asked for in the prompt; observed
# failure modes include "<<<FILE>" (one bracket short), "###\nEND" instead
# of a real end marker, and trailing garbage repeats of "<<<NO_FIX>>>"
# after an otherwise-correct fix. Terminator search is lenient on purpose;
# compile() below and, ultimately, Abi reading the diff before approving
# are the real safety net — not perfect parsing of a small model's output.
_TERMINATORS = re.compile(
    r"<{2,}\s*(?:END|NO_FIX)\s*>{0,}|^\s*#{2,}\s*$\n^\s*END\s*$|^\s*END\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TRAILING_JUNK_LINE = re.compile(r"^\s*(#{2,}|END|<{2,}.*|>{2,}.*)\s*$")


def _extract_patch(model_output: str) -> str | None:
    start = _FILE_START.search(model_output)
    if start is None:
        # No FILE marker at all — only treat as a real decline if NO_FIX
        # actually appears; otherwise there's nothing usable to extract.
        return None
    rest = model_output[start.end():]
    end = _TERMINATORS.search(rest)
    content = rest[:end.start()] if end else rest

    # Strip a markdown code fence the model may have wrapped the file in
    # despite being asked not to.
    content = re.sub(r"^```(?:python)?\s*\n", "", content)
    content = re.sub(r"\n```\s*$", "", content)

    # Trim trailing marker-fragment lines the terminator search didn't
    # cleanly separate out (e.g. "###" / "END" left dangling on their own
    # lines right at the cut point).
    lines = content.splitlines()
    while lines and _TRAILING_JUNK_LINE.match(lines[-1]):
        lines.pop()
    content = "\n".join(lines).strip("\n")
    return content or None


def propose_patch(target_path: str, instruction: str,
                   model: str = "qwen2.5:3b", host: str = "http://localhost:11434") -> dict:
    """Drafts a full-file replacement for target_path addressing
    `instruction`. Returns {ok, target_path, original, proposed, diff,
    reason}. Never writes anything — pure drafting."""
    rel = _norm_repo_path(target_path)
    abspath = os.path.join(_REPO_ROOT, rel)

    if is_protected(rel):
        return {"ok": False, "target_path": rel, "reason": f"'{rel}' is protected — refusing to draft a patch for it"}
    if not os.path.exists(abspath):
        return {"ok": False, "target_path": rel, "reason": f"no such file: {rel}"}

    with open(abspath, encoding="utf-8") as f:
        original = f.read()

    prompt = (f"File: {rel}\n\nProblem: {instruction}\n\n"
              f"Current file content:\n{original}")
    try:
        raw = _call_ollama(prompt, model, host)
    except Exception as e:
        return {"ok": False, "target_path": rel, "reason": f"reasoning model unavailable: {e}"}

    proposed = _extract_patch(raw)
    if proposed is None:
        return {"ok": False, "target_path": rel, "reason": "model declined or gave no parseable fix"}

    # Sanity-check before anything touches a sandbox: must at least be
    # syntactically valid Python, and must not be a no-op.
    try:
        compile(proposed, rel, "exec")
    except SyntaxError as e:
        return {"ok": False, "target_path": rel, "reason": f"proposed content doesn't compile: {e}"}
    if proposed.strip() == original.strip():
        return {"ok": False, "target_path": rel, "reason": "proposed content is unchanged from the original"}

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), proposed.splitlines(keepends=True),
        fromfile=f"a/{rel}", tofile=f"b/{rel}"))
    return {"ok": True, "target_path": rel, "original": original, "proposed": proposed, "diff": diff}


# --------------------------------------------------------------- sandbox test

def test_in_sandbox(target_path: str, proposed_content: str, timeout: float = 120.0) -> dict:
    """Applies proposed_content to target_path inside a throwaway git
    worktree (never the live checkout) and runs the real test suite there.
    Returns {passed, output}. The worktree is always removed before
    returning, pass or fail."""
    rel = _norm_repo_path(target_path)
    worktree_dir = os.path.join(_REPO_ROOT, ".self_modify_sandbox_" + hashlib.sha256(
        (rel + str(time.time())).encode()).hexdigest()[:10])
    branch = "self-modify-sandbox-" + hashlib.sha256(worktree_dir.encode()).hexdigest()[:8]

    try:
        subprocess.run(["git", "worktree", "add", "-b", branch, worktree_dir, "HEAD"],
                        cwd=_REPO_ROOT, check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as e:
        return {"passed": False, "output": f"couldn't create sandbox worktree: {e.stderr}"}

    try:
        target_in_worktree = os.path.join(worktree_dir, rel)
        with open(target_in_worktree, "w", encoding="utf-8") as f:
            f.write(proposed_content)

        import sys
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q"],
            cwd=worktree_dir, capture_output=True, text=True, timeout=timeout,
        )
        return {"passed": proc.returncode == 0, "output": (proc.stdout + proc.stderr)[-4000:]}
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "test run timed out"}
    finally:
        try:
            subprocess.run(["git", "worktree", "remove", "--force", worktree_dir],
                            cwd=_REPO_ROOT, check=True, capture_output=True, text=True, timeout=30)
        except subprocess.CalledProcessError:
            shutil.rmtree(worktree_dir, ignore_errors=True)
        subprocess.run(["git", "branch", "-D", branch], cwd=_REPO_ROOT,
                        capture_output=True, text=True, timeout=10)


# ------------------------------------------------------------------ proposals

def _proposal_path(proposal_id: str) -> str:
    return os.path.join(_PROPOSALS_DIR, f"{proposal_id}.json")


def _save_proposal(proposal: dict) -> None:
    os.makedirs(_PROPOSALS_DIR, exist_ok=True)
    with open(_proposal_path(proposal["id"]), "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2)


def load_proposal(proposal_id: str) -> dict | None:
    path = _proposal_path(proposal_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_proposals(status: str | None = None) -> list[dict]:
    if not os.path.isdir(_PROPOSALS_DIR):
        return []
    out = []
    for fname in sorted(os.listdir(_PROPOSALS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(_PROPOSALS_DIR, fname), encoding="utf-8") as f:
            p = json.load(f)
        if status is None or p.get("status") == status:
            out.append(p)
    return out


def create_proposal(target_path: str, instruction: str, source: str = "on_demand",
                     model: str = "qwen2.5:3b", host: str = "http://localhost:11434") -> dict:
    """The full pipeline: draft -> sandbox-test -> save. Only reaches
    status "pending" (actionable, awaiting a human) if the draft compiled
    AND the real test suite passed against it in an isolated worktree.
    Anything else is saved too (so autonomous scanning doesn't retry it
    every cycle) but marked so it's never presented as something to
    approve."""
    draft = propose_patch(target_path, instruction, model=model, host=host)
    pid = hashlib.sha256(f"{target_path}|{instruction}|{time.time()}".encode()).hexdigest()[:12]
    proposal = {
        "id": pid, "created": datetime.now(timezone.utc).isoformat(),
        "source": source, "target_path": _norm_repo_path(target_path), "instruction": instruction,
    }

    if not draft["ok"]:
        proposal.update(status="draft_failed", reason=draft["reason"])
        _save_proposal(proposal)
        return proposal

    test_result = test_in_sandbox(draft["target_path"], draft["proposed"])
    proposal.update(diff=draft["diff"], original=draft["original"], proposed=draft["proposed"],
                     test_output=test_result["output"])
    proposal["status"] = "pending" if test_result["passed"] else "test_failed"
    _save_proposal(proposal)
    return proposal


def approve(proposal_id: str, security_ref=None, is_admin_ref=None) -> str:
    """The gated merge step. Stages the proposed content into the REAL
    working tree file — does not `git add`/`git commit`/push. That stays a
    separate, deliberate act for Abi, same as this repo's own git rules."""
    proposal = load_proposal(proposal_id)
    if proposal is None:
        return f"No proposal '{proposal_id}'."
    if proposal["status"] != "pending":
        return f"Proposal '{proposal_id}' is '{proposal['status']}', not pending — nothing to approve."
    if is_protected(proposal["target_path"]):
        return f"Refusing — '{proposal['target_path']}' is protected."  # should be unreachable, defense in depth
    if not authorize_action(f"apply self-modify proposal {proposal_id} to {proposal['target_path']}",
                             security_ref, is_admin_ref):
        return "Denied — couldn't verify you for applying this proposal."

    abspath = os.path.join(_REPO_ROOT, proposal["target_path"])
    with open(abspath, "w", encoding="utf-8") as f:
        f.write(proposal["proposed"])
    proposal["status"] = "applied"
    _save_proposal(proposal)
    return (f"Applied to {proposal['target_path']} — staged in your working tree, NOT committed. "
            f"Review with `git diff` and commit yourself when you're satisfied.")


def reject(proposal_id: str, reason: str = "") -> str:
    proposal = load_proposal(proposal_id)
    if proposal is None:
        return f"No proposal '{proposal_id}'."
    proposal["status"] = "rejected"
    if reason:
        proposal["reject_reason"] = reason
    _save_proposal(proposal)
    return f"Rejected proposal '{proposal_id}'."


# --------------------------------------------------------------- autonomous

class AutonomousScanner:
    """Background thread: every `interval_seconds`, scans for unaddressed
    issues and drafts+tests a proposal for each (never applies). Opt-in via
    runtime/jarvis.py's --self-modify-autoscan, same as every other
    continuously-resource-consuming toggle in this app (--voice,
    --chat-mode) defaults off — this is an ordinary "don't run background
    work nobody asked to have running right now" default, not a
    reconsideration of Abi's decision to have this capability exist."""

    def __init__(self, interval_seconds: int = 1800,
                 model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.interval_seconds = interval_seconds
        self.model = model
        self.host = host
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for issue in scan_recent_issues():
                    if issue.get("file") is None:
                        _mark_addressed(issue["_signature"])
                        continue
                    proposal = create_proposal(
                        issue["file"], f"Recurring issue from {issue['source']}: {issue['message']}",
                        source="autonomous", model=self.model, host=self.host)
                    _mark_addressed(issue["_signature"])
                    status = proposal["status"]
                    print(f"[self_modify] autonomous scan: {issue['file']} -> proposal "
                          f"{proposal['id']} ({status})")
            except Exception as e:
                print(f"[self_modify] autonomous scan error: {e}")
            self._stop.wait(self.interval_seconds)


# --------------------------------------------------------------------- skill

_PROPOSE_PREFIX = "propose fix "
_LIST_TRIGGERS = {"list proposals", "list self-modify proposals", "list pending proposals"}
_SHOW_PREFIX = "show proposal "
_APPROVE_PREFIX = "approve proposal "
_REJECT_PREFIX = "reject proposal "


class SelfModifySkill(SkillModule):
    """Conversational surface. "propose fix <path>: <instruction>",
    "list proposals", "show proposal <id>", "approve proposal <id>"
    (gated), "reject proposal <id>"."""

    name = "self_modify"
    description = "drafts, sandbox-tests, and (with approval) applies code changes to JARVIS itself"
    priority = 9

    def __init__(self, security_ref=None, is_admin_ref=None,
                 model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
        self.security_ref = security_ref
        self.is_admin_ref = is_admin_ref
        self.model = model
        self.host = host

    @property
    def available(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.host.rstrip('/')}/api/version", timeout=2).ok
        except Exception:
            return False

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        if t in _LIST_TRIGGERS:
            return True
        return any(t.startswith(p) for p in
                   (_PROPOSE_PREFIX, _SHOW_PREFIX, _APPROVE_PREFIX, _REJECT_PREFIX))

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()

        if low in _LIST_TRIGGERS:
            proposals = list_proposals()
            if not proposals:
                return "No self-modify proposals yet."
            lines = ["Self-modify proposals:"]
            for p in proposals:
                lines.append(f"  - {p['id']}  [{p['status']}]  {p['target_path']}  ({p['source']})")
            return "\n".join(lines)

        if low.startswith(_SHOW_PREFIX):
            pid = t[len(_SHOW_PREFIX):].strip()
            p = load_proposal(pid)
            if p is None:
                return f"No proposal '{pid}'."
            lines = [f"Proposal {p['id']} [{p['status']}] — {p['target_path']}",
                     f"Instruction: {p['instruction']}"]
            if p.get("diff"):
                lines.append(p["diff"])
            if p.get("reason"):
                lines.append(f"Reason: {p['reason']}")
            return "\n".join(lines)

        if low.startswith(_APPROVE_PREFIX):
            pid = t[len(_APPROVE_PREFIX):].strip()
            return approve(pid, self.security_ref, self.is_admin_ref)

        if low.startswith(_REJECT_PREFIX):
            pid = t[len(_REJECT_PREFIX):].strip()
            return reject(pid)

        if low.startswith(_PROPOSE_PREFIX):
            rest = t[len(_PROPOSE_PREFIX):].strip()
            if ":" not in rest:
                return "Usage: propose fix <path/to/file.py>: <what's wrong>"
            path, instruction = rest.split(":", 1)
            proposal = create_proposal(path.strip(), instruction.strip(),
                                        source="on_demand", model=self.model, host=self.host)
            if proposal["status"] == "pending":
                return (f"Proposal {proposal['id']} ready — tests passed in sandbox. "
                        f"\"show proposal {proposal['id']}\" to see the diff, "
                        f"\"approve proposal {proposal['id']}\" to apply it.")
            if proposal["status"] == "test_failed":
                return f"Drafted a patch but it failed the test suite in sandbox (proposal {proposal['id']})."
            return f"Couldn't draft a fix: {proposal.get('reason', 'unknown error')}"

        return "I didn't catch that self-modify command."
