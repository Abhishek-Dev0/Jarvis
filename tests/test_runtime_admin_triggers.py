from jarvis.runtime.jarvis import Jarvis, _normalize_phrase


def test_record_turn_appends_and_does_not_summarize_by_default():
    j = Jarvis(max_history=2)
    j._record_turn("hi", "hello")
    j._record_turn("how are you", "good")
    j._record_turn("bye", "goodbye")
    assert len(j.history) == 3
    assert j.history_summary == ""  # summarize_enabled defaults to False


def test_record_turn_folds_aging_out_turns_when_summarize_enabled(monkeypatch):
    import jarvis.modules.summarize as summarize

    calls = []

    def fake_fold(previous_summary, user_text, reply, model="qwen2.5:3b",
                   host="http://localhost:11434", timeout=30):
        calls.append((previous_summary, user_text, reply))
        return f"{previous_summary}|{user_text}"

    monkeypatch.setattr(summarize, "fold_turn_into_summary", fake_fold)

    j = Jarvis(max_history=2)
    j.summarize_enabled = True
    j._record_turn("turn1 user", "turn1 reply")
    j._record_turn("turn2 user", "turn2 reply")
    assert calls == []  # window not yet full past max_history

    j._record_turn("turn3 user", "turn3 reply")
    assert len(calls) == 1
    assert calls[0][1] == "turn1 user"  # the turn that just aged out
    assert j.history_summary == "|turn1 user"

    j._record_turn("turn4 user", "turn4 reply")
    assert len(calls) == 2
    assert calls[1][1] == "turn2 user"
    assert j.history_summary == "|turn1 user|turn2 user"


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


def test_switch_persona_phrases_present_and_isolated():
    j = Jarvis()
    assert _normalize_phrase("switch to eve") in j.switch_eve_phrases
    assert _normalize_phrase("switch to jarvis") in j.switch_jarvis_phrases
    assert not (j.switch_eve_phrases & j.switch_jarvis_phrases)
    assert not (j.switch_eve_phrases & j.shutdown_phrases)


def test_switch_persona_updates_mascot_and_tts_engine_without_gating():
    from jarvis.modules.mascot import CatMascot
    from jarvis.modules.builtin import SpeechOutput

    j = Jarvis()
    j.mascot = CatMascot(enabled=False)

    class FakeTTSEngine:
        persona = "jarvis"

    engine = FakeTTSEngine()
    j.register(SpeechOutput(engine=engine, mascot=j.mascot))

    assert j.is_admin is False  # ungated -- no admin needed
    j._switch_persona("eve")
    assert engine.persona == "eve"
    assert j.mascot.persona == "eve"
    assert j.is_admin is False  # still not a security-relevant action


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
