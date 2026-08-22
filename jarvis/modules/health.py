"""
health.py — one command, one answer: what's actually reachable right now.

From the 2026-08-22 systems audit (priority P1): a way to check model/tool/
dependency reachability in one place instead of hunting through scattered
startup print() lines or waiting to hit a failure mid-conversation.

Read-only, ungated — nothing here changes state or reveals a secret
(enrollment status is reported as a boolean, never the passphrase/
embeddings themselves), so it needs no SecurityGate check, same reasoning
as os_control.py's "list processes" or hardware_io.py's "list serial ports."
"""

from __future__ import annotations

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_TRIGGERS = {"system health", "health check", "status check", "run diagnostics", "system status"}


class HealthCheckSkill(SkillModule):
    name = "health"
    description = "reports what's actually reachable right now (model, Ollama, MCP, camera, mic, hardware)"
    priority = 8  # same tier as web_search/market_analysis — informational, read-only

    def __init__(self, jarvis_ref=None, mcp_ref=None, reasoning_host="http://localhost:11434"):
        # jarvis_ref: () -> Jarvis, so this can inspect live state (core
        # model loaded, admin session state) without importing Jarvis
        # itself — same deferred-lambda pattern as every other module here.
        self.jarvis_ref = jarvis_ref
        self.mcp_ref = mcp_ref
        self.reasoning_host = reasoning_host.rstrip("/")

    def matches(self, text: str) -> bool:
        return text.strip().lower() in _TRIGGERS

    def handle(self, text: str) -> str:
        # A health check that can itself crash on one bad subsystem would be
        # a bad joke — every section is independently guarded so a failure
        # in, say, reading jarvis_ref still lets the rest of the report
        # through instead of losing the whole thing.
        lines = ["System health:"]
        lines.extend(self._safe(self._jarvis_lines, "jarvis state"))
        lines.extend(self._safe(lambda: [f"  ollama ({self.reasoning_host}): "
                                          f"{'reachable' if self._check_ollama() else 'NOT reachable'}"],
                                 "ollama"))
        lines.extend(self._safe(self._mcp_lines, "mcp"))
        lines.extend(self._safe(lambda: [f"  camera: "
                                          f"{'available' if self._check_camera() else 'not available'}"],
                                 "camera"))
        lines.extend(self._safe(lambda: [f"  microphone: "
                                          f"{'available' if self._check_mic() else 'not available'}"],
                                 "microphone"))
        lines.extend(self._safe(self._hardware_lines, "hardware"))
        return "\n".join(lines)

    @staticmethod
    def _safe(fn, label: str) -> list[str]:
        try:
            return fn()
        except Exception as e:
            return [f"  {label}: check failed ({e})"]

    def _jarvis_lines(self) -> list[str]:
        j = self.jarvis_ref() if self.jarvis_ref is not None else None
        if j is None:
            return ["  core model: unknown (no session reference)"]
        lines = [f"  core model: {'loaded' if j.model is not None else 'NOT loaded'}"]
        try:
            try:
                from ..security import is_enrolled
            except ImportError:  # pragma: no cover - legacy direct execution
                from security import is_enrolled
            enrolled = is_enrolled()
        except Exception:
            enrolled = None
        status = "unknown" if enrolled is None else ("enrolled" if enrolled else "not enrolled")
        if j.is_admin:
            status += ", admin this session"
        lines.append(f"  security: {status}")
        return lines

    def _mcp_lines(self) -> list[str]:
        mcp = self.mcp_ref() if self.mcp_ref is not None else None
        if mcp is None:
            return ["  mcp: not registered"]
        servers = mcp.list_servers()
        if not servers:
            return ["  mcp servers: none connected"]
        return [f"  mcp servers: {len(servers)} connected ({', '.join(servers)})"]

    def _hardware_lines(self) -> list[str]:
        try:
            from . import hardware
        except ImportError:  # pragma: no cover - legacy direct execution
            import hardware
        try:
            profile = hardware.detect()
        except Exception as e:
            return [f"  hardware: couldn't detect ({e})"]
        gpu = f"{profile['gpu_name']} ({profile['vram_gb']}GB VRAM)" if profile["has_cuda"] else "no CUDA GPU"
        return [f"  hardware: {profile['cpu_cores']}c/{profile['cpu_threads']}t, "
                f"{profile['ram_gb']}GB RAM, {gpu}"]

    def _check_ollama(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.reasoning_host}/api/version", timeout=2).ok
        except Exception:
            return False

    def _check_camera(self) -> bool:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            ok = cap.isOpened()
            cap.release()
            return ok
        except Exception:
            return False

    def _check_mic(self) -> bool:
        try:
            import sounddevice as sd
            return any(d.get("max_input_channels", 0) > 0 for d in sd.query_devices())
        except Exception:
            return False
