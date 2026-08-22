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


def test_refresh_multilingual_phrases_is_a_safe_no_op_without_translator():
    j = Jarvis()
    before = (set(j.admin_trigger_phrases), set(j.face_login_phrases))
    j.refresh_multilingual_phrases()  # translator is None by default
    after = (set(j.admin_trigger_phrases), set(j.face_login_phrases))
    assert before == after
