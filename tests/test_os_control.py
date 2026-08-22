from jarvis.modules.os_control import OSControlSkill, _resolve_target, _looks_like_app_name


def test_looks_like_app_name_accepts_alias():
    assert _looks_like_app_name("file explorer") is True


def test_looks_like_app_name_accepts_bare_word():
    assert _looks_like_app_name("notepad") is True


def test_looks_like_app_name_rejects_sentences():
    assert _looks_like_app_name("describing the weather patterns") is False
    assert _looks_like_app_name("me through what happened yesterday") is False


def test_looks_like_app_name_rejects_empty():
    assert _looks_like_app_name("") is False
    assert _looks_like_app_name("   ") is False


def test_resolve_target_uses_alias_table():
    assert _resolve_target("notepad") == "notepad.exe"
    assert _resolve_target("VS Code") == "code.exe"


def test_resolve_target_appends_exe_to_bare_word():
    assert _resolve_target("obs") == "obs.exe"


def test_resolve_target_leaves_multiword_unlisted_name_untouched():
    # not in the alias table, not a bare word -> passed through as-is
    assert _resolve_target("some random program") == "some random program"


def test_matches_real_commands():
    sk = OSControlSkill()
    assert sk.matches("open notepad") is True
    assert sk.matches("close chrome") is True
    assert sk.matches("start explorer") is True
    assert sk.matches("open task manager") is True
    assert sk.matches("list processes") is True


def test_matches_rejects_ordinary_conversation():
    sk = OSControlSkill()
    assert sk.matches("start describing the weather patterns") is False
    assert sk.matches("run me through what happened yesterday") is False
    assert sk.matches("close the door for good this time") is False
    assert sk.matches("kill the mood") is False
    assert sk.matches("tell me a joke") is False


def test_launch_denied_without_authorization():
    sk = OSControlSkill(security_ref=None, is_admin_ref=None)
    reply = sk.handle("open notepad")
    assert "Denied" in reply


def test_close_denied_without_authorization():
    sk = OSControlSkill(security_ref=None, is_admin_ref=None)
    reply = sk.handle("close notepad")
    assert "Denied" in reply


def test_is_admin_bypasses_gate():
    sk = OSControlSkill(security_ref=None, is_admin_ref=lambda: True)
    reply = sk._launch("nonexistent-app-xyz")
    # should get past the gate and fail on the actual OS call instead
    assert "Denied" not in reply
