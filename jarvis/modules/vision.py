"""
vision.py — local image understanding via a vision-capable Ollama model.
From the 2026-08-22 systems audit (P2): JARVIS had no way to look at an
image, a screenshot, or a photo at all — every other perception path
(speech, hardware sensors, a single face-auth frame) existed except this
one.

Model choice measured, not guessed, same discipline as reasoning.py's
qwen2.5:3b pick: moondream (~1B params, ~1.24GB VRAM once loaded)
benchmarked on this machine's 4GB-VRAM RTX 3050 at ~2.5s/image once warm,
and gave an accurate description on the first real test image (correctly
named colors, shapes, and features of a hand-drawn icon it had never seen).
First call after a cold start is much slower (~55s) — that's Ollama
loading the model into VRAM once, same one-time cost every model in this
project pays, not a per-call cost. Small enough to sit alongside
qwen2.5:3b without evicting it (~1.24GB + ~2GB, both well under 4GB).

Swap-in/out like every other model here (--vision-model overrides);
silently unavailable if Ollama isn't running, same graceful-degradation
pattern as reasoning.py. A separate skill/model from reasoning.py on
purpose — qwen2.5:3b has no vision capability, this isn't a variant of
the same call, it's a different model entirely.

Deliberately no path restriction (unlike fileread.py's repo-scoped text
reads): a screenshot in Downloads or a photo anywhere else on disk is the
whole point of this existing. Read-only — this only ever reads image
bytes off disk and sends them to the local Ollama instance, nothing is
written, executed, or sent anywhere beyond localhost.
"""

from __future__ import annotations

import base64
import os

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_TRIGGERS = ("describe image", "describe the image", "analyze image", "analyze the image",
             "what's in image", "whats in image", "look at image")

_DEFAULT_QUESTION = "Describe this image."


class VisionSkill(SkillModule):
    """"describe image <path>" or "describe image <path>: <question>" —
    ungated, read-only. See the module docstring for the model choice and
    why paths aren't restricted to the repo the way fileread.py's are."""

    name = "vision"
    description = "describes or answers questions about a local image file (moondream via Ollama)"
    priority = 8

    def __init__(self, model="moondream", host="http://localhost:11434", timeout=90):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import requests
                self._available = requests.get(f"{self.host}/api/version", timeout=2).ok
            except Exception:
                self._available = False
        return self._available

    def setup(self) -> None:
        # Same best-effort friendliness check as reasoning.py's setup() —
        # warn now if the model isn't pulled, instead of only failing
        # later inside a live turn.
        if not self.available:
            return
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            pulled = {m.get("name") for m in resp.json().get("models", [])}
            if self.model not in pulled and f"{self.model}:latest" not in pulled:
                print(f"[vision] '{self.model}' isn't pulled yet — run: ollama pull {self.model}")
        except Exception:
            pass

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        return any(t.startswith(p) for p in _TRIGGERS)

    def handle(self, text: str) -> str:
        t = text.strip()
        low = t.lower()
        rest = None
        for prefix in sorted(_TRIGGERS, key=len, reverse=True):
            if low.startswith(prefix):
                rest = t[len(prefix):].strip()
                break
        if not rest:
            return "Describe which image? (e.g. \"describe image C:\\path\\to\\photo.jpg\")"

        # Split on ": " (colon-*space*), not a bare colon — a Windows drive
        # letter's colon ("C:\Users\...") is always immediately followed by
        # a backslash, never a space, so this cleanly separates "<path>:
        # <question>" from a plain absolute path without needing to special
        # -case drive letters.
        if ": " in rest:
            path, question = rest.split(": ", 1)
            path, question = path.strip(), question.strip() or _DEFAULT_QUESTION
        else:
            path, question = rest, _DEFAULT_QUESTION

        if not path:
            return "Describe which image?"

        abspath = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(abspath):
            return f"No such file: {path}"

        try:
            image_bytes = self._normalize_to_png(abspath)
        except Exception as e:
            return f"Couldn't read '{path}' as an image: {e}"

        b64 = base64.b64encode(image_bytes).decode()
        import requests
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "stream": False,
                      "messages": [{"role": "user", "content": question, "images": [b64]}]},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            return (f"Vision model isn't responding ({e}). Is `ollama serve` running "
                     f"and is '{self.model}' pulled?")

    @staticmethod
    def _normalize_to_png(abspath: str) -> bytes:
        """Re-encodes through PIL to PNG regardless of source format,
        rather than sending the file's raw bytes straight through. Found
        necessary by actually testing this, not assumed: Ollama's vision
        endpoint rejected a real .ico file outright ("Failed to load image
        or audio file", HTTP 400) even though it's a perfectly valid image
        PIL opens without complaint — its accepted format list is narrower
        than "anything a human would call an image file." Re-encoding here
        means this skill works with whatever the user actually has (.ico,
        .bmp, .tiff, ...), not just Ollama's native list, and costs nothing
        for formats that would have worked anyway (PNG/JPEG round-trip
        losslessly enough for a description task)."""
        import io
        from PIL import Image
        with Image.open(abspath) as img:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            return buf.getvalue()
