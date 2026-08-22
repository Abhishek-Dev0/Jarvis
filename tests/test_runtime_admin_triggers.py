from jarvis.runtime.jarvis import Jarvis, _normalize_phrase


def test_admin_trigger_phrases_normalize_correctly():
    j = Jarvis()
    assert _normalize_phrase("I'm the admin") in j.admin_trigger_phrases
    assert _normalize_phrase("I am the admin.") in j.admin_trigger_phrases


def test_face_login_phrases_present():
    j = Jarvis()
    assert _normalize_phrase("recognize me") in j.face_login_phrases
    assert _normalize_phrase("Recognize Me!") in j.face_login_phrases


def test_admin_and_face_phrases_dont_collide_with_shutdown():
    j = Jarvis()
    assert not (j.admin_trigger_phrases & j.shutdown_phrases)
    assert not (j.face_login_phrases & j.shutdown_phrases)


def test_change_passphrase_phrases_present_and_isolated():
    j = Jarvis()
    assert _normalize_phrase("change the phrase") in j.change_passphrase_phrases
    assert _normalize_phrase("Change The Passphrase!") in j.change_passphrase_phrases
    assert not (j.change_passphrase_phrases & j.shutdown_phrases)
    assert not (j.change_passphrase_phrases & j.admin_trigger_phrases)


def test_handle_change_passphrase_denies_without_admin_or_credentials(tmp_path, monkeypatch):
    from jarvis import security
    sec_dir = tmp_path / "security"
    monkeypatch.setattr(security, "_SEC_DIR", str(sec_dir))
    monkeypatch.setattr(security, "_PASS_HASH_PATH", str(sec_dir / "passphrase.hash"))
    monkeypatch.setattr(security, "_SALT_PATH", str(sec_dir / "passphrase.salt"))

    j = Jarvis()
    from jarvis.modules.builtin import ConsoleOutput
    j.register(ConsoleOutput())
    assert j.is_admin is False
    j._handle_change_passphrase()
    assert j.is_admin is False  # still denied -- nothing enrolled, no admin session


def test_refresh_multilingual_phrases_is_a_safe_no_op_without_translator():
    j = Jarvis()
    before = (set(j.admin_trigger_phrases), set(j.face_login_phrases))
    j.refresh_multilingual_phrases()  # translator is None by default
    after = (set(j.admin_trigger_phrases), set(j.face_login_phrases))
    assert before == after
