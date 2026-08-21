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
for it, or let modules/hardware.py's recommend_reasoning_model() pick.

Optionally hands off to modules/mcp_client.py: if constructed with an
mcp_ref (see runtime/jarvis.py), tool schemas from connected MCP servers are
passed into Ollama's tool-calling API (qwen2.5 supports it) so the model can
decide, from plain conversation, that it needs to call a real tool — not a
rigid "mcp call X Y" command a human has to type correctly. Every tool call
the model requests still goes through MCPSkill.call_tool(), which is
security-gated by default; see that module's docstring for why.
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
                 system_prompt=None, history_ref=None, max_history=6, timeout=60,
                 mcp_ref=None, max_tool_turns=4):
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
        # Callable returning the live MCPSkill instance (same lambda-ref
        # pattern as history_ref/security_ref elsewhere) — deferred so this
        # doesn't care whether MCP connected before or after construction.
        self.mcp_ref = mcp_ref
        # Hard cap on model-requests-a-tool -> tool-runs -> model-answers
        # round trips per turn, so a model stuck calling tools in a loop
        # can't hang a conversation turn forever.
        self.max_tool_turns = max_tool_turns
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

        mcp = self.mcp_ref() if self.mcp_ref is not None else None
        tools = mcp.as_ollama_tools() if mcp is not None else None

        for _ in range(self.max_tool_turns):
            payload = {"model": self.model, "messages": messages, "stream": False}
            if tools:
                payload["tools"] = tools
            try:
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
                r.raise_for_status()
                msg = r.json()["message"]
            except Exception as e:
                return f"My reasoning model isn't responding ({e}). Is `ollama serve` running?"

            calls = msg.get("tool_calls")
            if not calls:
                return (msg.get("content") or "").strip()

            messages.append(msg)
            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                result = mcp.call_tool(name, args) if mcp is not None else \
                    f"No MCP connection available to call '{name}'."
                messages.append({"role": "tool", "content": str(result)})

        return "I kept needing another tool call and hit the limit for this turn — try asking again, more narrowly."
