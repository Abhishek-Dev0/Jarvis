"""
reasoning.py — local LLM (via Ollama) for anything needing real knowledge or
coherent dialogue. The from-scratch core model can't do this at its current
scale (nano, ~1.8M params trained on 5 novels) — that's physics, not a bug,
see the JARVIS architecture notes. This is the swap-in/out module the
2026-08-15 hybrid decision planned for: it attaches from outside, same as
every other module, and never touches core/. JARVIS still works fine
(skills-only, or the raw core model) if Ollama isn't running.

Model choice matters more than it looks like it should: this machine has a
4GB-VRAM GPU. gemma4:12b-it-qat (already pulled on this machine before this
session) benchmarked at ~3m48s for one short sentence — 12B doesn't fit in
4GB, so most of it runs on CPU. qwen2.5:3b benchmarked at ~1.6s once warm —
fits in VRAM, fast enough for real conversation. Default here reflects that
measurement, not a guess; override with a bigger model if you have the VRAM
for it, or once modules/hardware.py grows a real quantization-aware picker.
"""

from __future__ import annotations

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule


class ReasoningSkill(SkillModule):
    name = "reasoning"
    description = "local LLM (Ollama) for real conversation/knowledge"
    # Lower than every other skill (calculator=10, web search=0-ish) so
    # they still get first crack at a request. matches() is a catch-all —
    # this is meant to replace "fall through to the raw core model" as the
    # default conversational path, not compete with more specific skills.
    priority = -100

    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434",
                 system_prompt=None, history_ref=None, max_history=6, timeout=60):
        self.model = model
        self.host = host.rstrip("/")
        self.system_prompt = system_prompt or (
            "You are JARVIS, a helpful local assistant. Be concise and direct.")
        # Callable returning Jarvis.history (e.g. lambda: j.history) so this
        # stays in sync with the live conversation without importing Jarvis
        # itself — modules don't reach back into the orchestrator.
        self.history_ref = history_ref
        self.max_history = max_history
        self.timeout = timeout
        self._available = None

    @property
    def available(self):
        if self._available is None:
            try:
                import requests
                self._available = requests.get(f"{self.host}/api/version", timeout=2).ok
            except Exception:
                self._available = False
        return self._available

    def setup(self):
        # Best-effort friendliness check, not a hard requirement: warn if the
        # hardware-recommended (or explicitly requested) model tag hasn't
        # actually been pulled yet, so a bad first run says why instead of
        # just failing inside handle()'s except block later.
        if not self.available:
            return
        try:
            import requests
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            pulled = {m.get("name") for m in resp.json().get("models", [])}
            if self.model not in pulled:
                print(f"[reasoning] '{self.model}' isn't pulled yet — run: ollama pull {self.model}")
        except Exception:
            pass

    def matches(self, text):
        return True

    def handle(self, text):
        import requests
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.history_ref is not None:
            for u, a in self.history_ref()[-self.max_history:]:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": text})
        try:
            r = requests.post(f"{self.host}/api/chat",
                               json={"model": self.model, "messages": messages, "stream": False},
                               timeout=self.timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            return f"My reasoning model isn't responding ({e}). Is `ollama serve` running?"
