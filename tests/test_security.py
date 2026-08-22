import os

from jarvis import security


def test_normalize_lowercases_strips_and_collapses_whitespace():
    assert security._normalize("  Correct   Horse Battery-Staple!  ") == "correct horse batterystaple"


def test_normalize_is_idempotent():
    once = security._normalize("Hello, World!")
    assert security._normalize(once) == once


def test_hash_passphrase_deterministic_for_same_input_and_salt():
    salt = b"fixed-salt-for-test"
    a = security._hash_passphrase("open sesame", salt)
    b = security._hash_passphrase("Open Sesame", salt)  # normalization should make these equal
    assert a == b


def test_hash_passphrase_differs_for_different_passphrases():
    salt = b"fixed-salt-for-test"
    a = security._hash_passphrase("open sesame", salt)
    b = security._hash_passphrase("close sesame", salt)
    assert a != b


def test_hash_passphrase_differs_for_different_salts():
    a = security._hash_passphrase("open sesame", b"salt-one")
    b = security._hash_passphrase("open sesame", b"salt-two")
    assert a != b


def _isolate_security_paths(tmp_path, monkeypatch):
    sec_dir = tmp_path / "security"
    monkeypatch.setattr(security, "_SEC_DIR", str(sec_dir))
    monkeypatch.setattr(security, "_PASS_HASH_PATH", str(sec_dir / "passphrase.hash"))
    monkeypatch.setattr(security, "_SALT_PATH", str(sec_dir / "passphrase.salt"))
    monkeypatch.setattr(security, "_VOICEPRINT_PATH", str(sec_dir / "voiceprint.npy"))


def test_not_enrolled_before_enroll_called(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    assert security.is_enrolled() is False
    assert security.has_voiceprint() is False


def test_enroll_then_correct_passphrase_verifies(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    assert security.is_enrolled() is True
    assert security.has_voiceprint() is False  # no voice_audio given

    gate = security.SecurityGate()  # no mic -> console/typed path
    assert gate._check_passphrase("correct horse battery staple") is True
    assert gate._check_passphrase("Correct Horse Battery Staple") is True  # normalized match


def test_enroll_then_wrong_passphrase_fails(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    gate = security.SecurityGate()
    assert gate._check_passphrase("wrong words entirely") is False


def test_enroll_rejects_empty_passphrase(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    try:
        security.enroll(passphrase="   ")
        assert False, "should have rejected an empty/whitespace-only passphrase"
    except ValueError:
        pass


def test_authorize_denies_when_nothing_enrolled(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    gate = security.SecurityGate()
    assert gate.authorize("do something risky") is False


def test_authorize_console_mode_retries_once_then_succeeds(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    attempts = iter(["wrong first try", "correct horse battery staple"])
    monkeypatch.setattr(security.getpass, "getpass", lambda prompt="": next(attempts))
    gate = security.SecurityGate()
    assert gate.authorize("do something risky") is True


def test_authorize_console_mode_denies_after_two_wrong_attempts(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    monkeypatch.setattr(security.getpass, "getpass", lambda prompt="": "still wrong")
    gate = security.SecurityGate()
    assert gate.authorize("do something risky") is False


def test_authorize_become_admin_uses_sci_fi_denial_wording(tmp_path, monkeypatch, capsys):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    monkeypatch.setattr(security.getpass, "getpass", lambda prompt="": "wrong every time")
    gate = security.SecurityGate()
    assert gate.authorize("become admin") is False
    out = capsys.readouterr().out
    assert "Admin denied. You're a general user." in out


def test_authorize_other_reasons_keep_generic_denial_wording(tmp_path, monkeypatch, capsys):
    _isolate_security_paths(tmp_path, monkeypatch)
    security.enroll(passphrase="correct horse battery staple")
    monkeypatch.setattr(security.getpass, "getpass", lambda prompt="": "wrong every time")
    gate = security.SecurityGate()
    assert gate.authorize("shut down JARVIS") is False
    out = capsys.readouterr().out
    assert "passphrase did not match" in out
    assert "You're a general user" not in out
