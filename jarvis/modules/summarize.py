"""summarize.py — folds conversation turns that have aged out of the live
context window into a running text summary, via Ollama.

Why this exists: modules/reasoning.py and runtime/jarvis.py's build_prompt
both window context to the last `max_history` turns (see reasoning.py's
docstring) — necessary so a long session doesn't blow the model's context
or slow every request down re-sending the whole conversation. But that
means anything said before those N turns was simply gone: ask "what did I
tell you five minutes ago" past turn N and there was no way for the model
to know. This keeps a cheap, continuously-updated compressed recap of
everything older, so it can still be handed to the model alongside the
live window instead of being lost outright.

Deliberately session-scoped, same lifetime as Jarvis.history itself — this
is a *recap* of the current conversation, not a durable fact JARVIS was
asked to remember (that's modules/memory.py, which persists to disk and
survives a restart on purpose). Resetting the summary on restart is
correct here, not a limitation.
"""

from __future__ import annotations

_PROMPT_TEMPLATE = (
    "You maintain a short running summary of an ongoing conversation, for "
    "another assistant's own later reference. Update the summary so it "
    "also covers the new exchange below. Keep it under 150 words, written "
    "as plain prose covering the topics, facts, and decisions that matter "
    "for understanding later turns — not a transcript.\n\n"
    "Current summary: {previous_summary}\n\n"
    "New exchange:\nUser: {user_text}\nAssistant: {reply}\n\n"
    "Updated summary:"
)


def fold_turn_into_summary(previous_summary: str, user_text: str, reply: str,
                            model: str = "qwen2.5:3b", host: str = "http://localhost:11434",
                            timeout: int = 30) -> str:
    """Returns an updated running summary that also covers (user_text, reply).
    Best-effort: on any failure (Ollama not running, timeout, bad response,
    ...) returns previous_summary unchanged rather than raising —
    summarization is a nice-to-have that must never break a live turn."""
    import requests
    prompt = _PROMPT_TEMPLATE.format(
        previous_summary=previous_summary or "(none yet)",
        user_text=user_text, reply=reply,
    )
    try:
        r = requests.post(f"{host.rstrip('/')}/api/generate",
                           json={"model": model, "prompt": prompt, "stream": False},
                           timeout=timeout)
        r.raise_for_status()
        updated = (r.json().get("response") or "").strip()
        return updated or previous_summary
    except Exception:
        return previous_summary
