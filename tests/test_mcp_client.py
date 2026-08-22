from types import SimpleNamespace

from jarvis.modules.mcp_client import MCPSkill


def _fake_tool(name, description="", input_schema=None):
    return SimpleNamespace(name=name, description=description,
                            input_schema=input_schema or {"type": "object", "properties": {}})


def test_as_ollama_tools_flattens_with_server_prefixed_names():
    sk = MCPSkill()
    sk._tools = {
        "filesystem": [_fake_tool("read_file", "reads a file")],
        "web": [_fake_tool("search", "searches the web")],
    }
    tools = sk.as_ollama_tools()
    names = {t["function"]["name"] for t in tools}
    assert names == {"filesystem__read_file", "web__search"}
    for t in tools:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_as_ollama_tools_empty_when_nothing_connected():
    sk = MCPSkill()
    assert sk.as_ollama_tools() == []


def test_call_tool_rejects_malformed_name():
    sk = MCPSkill()
    result = sk.call_tool("no_double_underscore_here", {})
    assert "Malformed" in result


def test_call_tool_reports_unknown_server():
    sk = MCPSkill()
    result = sk.call_tool("nonexistent_server__some_tool", {})
    assert "No connected MCP server" in result


def test_call_tool_gated_when_server_known_but_unauthorized():
    sk = MCPSkill(security_ref=None, is_admin_ref=None)
    sk._sessions = {"filesystem": object()}  # present, so it gets past the "unknown server" check
    result = sk.call_tool("filesystem__read_file", {}, gate=True)
    assert "Denied" in result


def test_call_tool_ungated_bypasses_authorization():
    sk = MCPSkill(security_ref=None, is_admin_ref=None)
    sk._sessions = {"filesystem": object()}

    class FakeLoop:
        def run(self, coro, timeout=30):
            return SimpleNamespace(content=[], is_error=False)
    sk._loop = FakeLoop()
    result = sk.call_tool("filesystem__read_file", {}, gate=False)
    assert "Denied" not in result


def test_list_servers_and_list_tools_empty_by_default():
    sk = MCPSkill()
    assert sk.list_servers() == []
    assert sk.list_tools() == {}


def test_matches_only_list_triggers_not_tool_calls():
    sk = MCPSkill()
    assert sk.matches("list mcp servers") is True
    assert sk.matches("list mcp tools") is True
    assert sk.matches("use the search tool") is False  # tool-calling goes through reasoning, not this skill
