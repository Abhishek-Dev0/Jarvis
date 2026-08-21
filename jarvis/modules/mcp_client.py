"""
mcp_client.py — connects JARVIS to external MCP (Model Context Protocol)
servers, so the reasoning model can call real tools instead of just talking
about them. This is the MCP client capability + "general agentic capability
comparable to modern AI" Abi explicitly granted scope for on 2026-08-22.

Two jobs, one class (kept together deliberately, unlike hardware_io.py's
transport/skill split — there's only one reasonable consumer of an MCP
connection, the reasoning model, so there's no separate concern to isolate):

  connection management   load data/mcp_servers.json, connect to each
                           configured server, keep sessions open for the
                           process lifetime (most MCP servers are
                           subprocesses over stdio — slow to start, meant to
                           be talked to repeatedly, not reconnected per call)

  conversational surface  "list mcp servers" / "list mcp tools" — read-only,
                           same trigger-phrase pattern as every other skill.
                           Actually *calling* a tool does not go through
                           matches()/handle() at all — it goes through
                           ReasoningSkill, which is handed this object (see
                           runtime/jarvis.py's mcp_ref) and passes
                           as_ollama_tools() into Ollama's tool-calling API.
                           The model decides when a tool is needed from
                           natural conversation; that's the actual point of
                           MCP over a rigid "mcp call X Y" command grammar.

MCP's client API is async (anyio/asyncio); everything else in this codebase
is sync. Bridged with one background thread running a persistent event loop
— sync callers dispatch onto it with run_coroutine_threadsafe() and block
for the result, same shape as any sync-wrapping-async bridge.

Every tool call is security-gated by default (see call_tool's `gate`
param) — same reasoning as os_control.py/hardware_io.py: a tool's real
side effects are whatever its (someone else's) server implementation does,
unknown to JARVIS ahead of time, so nothing here assumes a tool is safe
just because the model wants to call it.

Nothing connects until data/mcp_servers.json lists at least one server —
ships as an empty list by default. See _EXAMPLE_CONFIG below for the shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading

try:
    from .base import SkillModule
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CONFIG_PATH = os.path.join(_PKG_DIR, "data", "mcp_servers.json")

# Shown in the module docstring / README, not executed — the shape
# data/mcp_servers.json entries should follow.
_EXAMPLE_CONFIG = [
    {"name": "filesystem", "transport": "stdio",
     "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\some\\path"]},
    {"name": "remote-example", "transport": "sse", "url": "https://example.com/mcp/sse"},
]

_LIST_TRIGGERS = {
    "list mcp servers", "list mcp tools", "what mcp tools are available",
    "what tools do you have", "what tools are available",
}


class _MCPLoop:
    """One background thread running one persistent asyncio event loop, so
    MCP sessions (and their subprocess/socket state) live outside of
    whatever thread happens to call into them."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout=30):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5.0)


class MCPSkill(SkillModule):
    name = "mcp"
    description = "connects to MCP servers; tool calls happen through the reasoning model, gated"
    priority = 9  # same tier as os_control/hardware — read-only here, but the
                  # capability it exposes (tool calls, via ReasoningSkill) is not

    def __init__(self, config_path: str | None = None, security_ref=None, is_admin_ref=None):
        self.config_path = config_path or _DEFAULT_CONFIG_PATH
        self.security_ref = security_ref
        self.is_admin_ref = is_admin_ref
        self._loop: _MCPLoop | None = None
        self._sessions: dict = {}       # server name -> ClientSession
        self._tools: dict = {}          # server name -> list[mcp.types.Tool]
        # Each connection lives inside one persistent asyncio task for its
        # whole life (_serve_connection) — its AsyncExitStack is entered and
        # exited in that same task. anyio's cancel scopes (used internally
        # by stdio_client/ClientSession) are tied to the task that entered
        # them; opening in one dispatched coroutine and closing in another
        # (the first version of this code did exactly that) raises
        # "Attempted to exit cancel scope in a different task than it was
        # entered in" — caught by actually running the connect+call+close
        # round trip against a real test server, not just import-checking
        # this module. _close_events signals that task to shut down;
        # _tasks holds the concurrent.futures.Future so teardown() can wait
        # for the close to actually finish before stopping the loop.
        self._close_events: dict = {}   # server name -> asyncio.Event
        self._tasks: dict = {}          # server name -> concurrent.futures.Future

    @property
    def available(self) -> bool:
        try:
            import mcp  # noqa: F401
            return True
        except ImportError:
            return False

    def setup(self) -> None:
        if not self.available or not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, encoding="utf-8") as f:
                servers = json.load(f)
        except Exception as e:
            print(f"[mcp] couldn't read {self.config_path}: {e}")
            return
        if not servers:
            return
        self._loop = _MCPLoop()
        for entry in servers:
            name = entry.get("name", "?")
            holder: dict = {}
            ready = threading.Event()
            fut = asyncio.run_coroutine_threadsafe(
                self._serve_connection(entry, holder, ready), self._loop.loop)
            if not ready.wait(timeout=30):
                print(f"[mcp] '{name}' timed out connecting")
                continue
            if "error" in holder:
                print(f"[mcp] couldn't connect to '{name}': {holder['error']}")
                continue
            self._sessions[name] = holder["session"]
            self._tools[name] = holder["tools"]
            self._close_events[name] = holder["close_event"]
            self._tasks[name] = fut
            print(f"[mcp] connected '{name}' ({len(holder['tools'])} tools)")

    async def _serve_connection(self, entry: dict, holder: dict, ready: threading.Event) -> None:
        """Runs for the connection's entire lifetime as one task: connect,
        publish the session/tools back to the sync caller via `holder`,
        then block until told to close — so the AsyncExitStack that opened
        the connection is the same one that closes it, in the same task."""
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        close_event = asyncio.Event()
        holder["close_event"] = close_event
        try:
            async with AsyncExitStack() as stack:
                transport = entry.get("transport", "stdio")
                if transport == "stdio":
                    params = StdioServerParameters(command=entry["command"], args=entry.get("args", []),
                                                    env=entry.get("env"))
                    read, write = await stack.enter_async_context(stdio_client(params))
                elif transport == "sse":
                    from mcp.client.sse import sse_client
                    read, write = await stack.enter_async_context(sse_client(entry["url"]))
                else:
                    raise ValueError(f"unsupported transport '{transport}' (use 'stdio' or 'sse')")

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()

                holder["session"] = session
                holder["tools"] = listed.tools
                ready.set()
                await close_event.wait()
        except Exception as e:
            holder["error"] = e
            ready.set()

    def teardown(self) -> None:
        if self._loop is None:
            return
        for close_event in self._close_events.values():
            self._loop.loop.call_soon_threadsafe(close_event.set)
        for name, fut in list(self._tasks.items()):
            try:
                fut.result(timeout=10)
            except Exception as e:
                print(f"[mcp] error closing '{name}': {e}")
        self._loop.stop()
        self._loop = None

    # -------------------------------------------------------------- queries

    def list_servers(self) -> list[str]:
        return list(self._sessions.keys())

    def list_tools(self) -> dict:
        return dict(self._tools)

    def as_ollama_tools(self) -> list[dict]:
        """Flattened tool schemas in Ollama/OpenAI function-calling format.
        Tool names are '<server>__<tool>' so a call routes back to the right
        session — Ollama's tool-call protocol only gives back a flat name."""
        out = []
        for server, tools in self._tools.items():
            for t in tools:
                out.append({
                    "type": "function",
                    "function": {
                        "name": f"{server}__{t.name}",
                        "description": t.description or "",
                        "parameters": t.input_schema or {"type": "object", "properties": {}},
                    },
                })
        return out

    # ---------------------------------------------------------------- calls

    def _authorized(self, reason: str) -> bool:
        if self.is_admin_ref is not None and self.is_admin_ref():
            return True
        if self.security_ref is None:
            return False
        return self.security_ref().authorize(reason)

    def call_tool(self, qualified_name: str, arguments: dict, gate: bool = True) -> str:
        """qualified_name: '<server>__<tool>' as produced by as_ollama_tools()."""
        if "__" not in qualified_name:
            return f"Malformed tool name '{qualified_name}' (expected '<server>__<tool>')."
        server, tool = qualified_name.split("__", 1)
        if server not in self._sessions:
            return f"No connected MCP server named '{server}'."
        if gate and not self._authorized(f"call MCP tool '{tool}' on '{server}'"):
            return "Denied — couldn't verify you for this tool call."
        try:
            result = self._loop.run(self._sessions[server].call_tool(tool, arguments), timeout=60)
        except Exception as e:
            return f"Tool call failed: {e}"
        texts = [b.text for b in getattr(result, "content", []) if getattr(b, "text", None)]
        body = "\n".join(texts) if texts else "(tool returned no text content)"
        return f"[tool error] {body}" if getattr(result, "is_error", False) else body

    # -------------------------------------------------------- listing skill

    def matches(self, text: str) -> bool:
        return text.strip().lower() in _LIST_TRIGGERS

    def handle(self, text: str) -> str:
        t = text.strip().lower()
        if t == "list mcp servers":
            servers = self.list_servers()
            return ("Connected MCP servers: " + ", ".join(servers)) if servers else \
                "No MCP servers connected. Configure them in data/mcp_servers.json."
        tools = self.list_tools()
        if not tools:
            return "No MCP tools available. Configure servers in data/mcp_servers.json."
        lines = ["MCP tools:"]
        for server, ts in tools.items():
            for t_ in ts:
                lines.append(f"  - {server}.{t_.name}: {t_.description or ''}")
        return "\n".join(lines)
