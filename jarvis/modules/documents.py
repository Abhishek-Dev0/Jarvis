"""
documents.py — text extraction from PDF and Word documents. From the
2026-08-22 systems audit (P2): web.py already strips HTML to text, but
nothing in this project could read a PDF or a .docx — a real gap for
anything that shows up as a downloaded paper, a scanned form, or a report
rather than a web page.

Deliberately narrow: extracts text, nothing else (no OCR for scanned
image-only PDFs, no table/layout reconstruction, no embedded-image
extraction). That covers the actual common case — a text-based PDF or Word
doc — without pulling in a much heavier document-AI stack for a personal
assistant that doesn't need one.

No path restriction, same reasoning as vision.py: a PDF someone actually
wants read is as likely to be in Downloads as in this repo. Read-only.
"""

from __future__ import annotations

import os

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_MAX_CHARS = 12_000
_SUPPORTED_EXTENSIONS = (".pdf", ".docx")


def extract_text(path: str, max_chars: int = _MAX_CHARS) -> dict:
    """Returns {ok, path, content, truncated, reason}. Never raises."""
    abspath = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abspath):
        return {"ok": False, "path": path, "reason": "no such file"}

    ext = os.path.splitext(abspath)[1].lower()
    try:
        if ext == ".pdf":
            text = _extract_pdf(abspath)
        elif ext == ".docx":
            text = _extract_docx(abspath)
        else:
            return {"ok": False, "path": path,
                    "reason": f"unsupported extension '{ext}' (supported: {', '.join(_SUPPORTED_EXTENSIONS)})"}
    except Exception as e:
        return {"ok": False, "path": path, "reason": f"couldn't extract text: {e}"}

    if not text.strip():
        return {"ok": False, "path": path,
                "reason": "no extractable text — likely a scanned/image-only document (OCR isn't supported)"}
    truncated = len(text) > max_chars
    return {"ok": True, "path": path, "content": text[:max_chars], "truncated": truncated}


def _extract_pdf(abspath: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(abspath)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _extract_docx(abspath: str) -> str:
    import docx
    doc = docx.Document(abspath)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(p for p in paragraphs if p.strip())


_TRIGGERS = ("read document", "read pdf", "read docx", "summarize document", "extract text from")


class DocumentSkill(SkillModule):
    """"read document <path>" — ungated, read-only. Extracts text only;
    pair with the reasoning model yourself for a summary (this skill
    doesn't call it — same "one thing, correctly" split as fileread.py
    vs. the reasoning loop that consumes its output)."""

    name = "documents"
    description = "extracts text from a local PDF or Word document (no OCR)"
    priority = 8

    @property
    def available(self) -> bool:
        try:
            import pypdf  # noqa: F401
            import docx  # noqa: F401
            return True
        except ImportError:
            return False

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) for p in _TRIGGERS)

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()
        path = None
        for prefix in sorted(_TRIGGERS, key=len, reverse=True):
            if low.startswith(prefix):
                path = t[len(prefix):].strip()
                break
        if not path:
            return "Read which document? (.pdf or .docx)"

        result = extract_text(path)
        if not result["ok"]:
            return f"Couldn't read '{result['path']}': {result['reason']}"
        header = f"{result['path']}" + (" (truncated)" if result["truncated"] else "") + ":\n"
        return header + result["content"]
