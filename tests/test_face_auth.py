import numpy as np

from jarvis import security


def _isolate_security_paths(tmp_path, monkeypatch):
    sec_dir = tmp_path / "security"
    monkeypatch.setattr(security, "_SEC_DIR", str(sec_dir))
    monkeypatch.setattr(security, "_FACE_EMBEDDING_PATH", str(sec_dir / "face_embedding.npy"))


def test_has_face_false_before_enrollment(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    assert security.has_face() is False


def test_enroll_face_rejects_frame_with_no_detected_face(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(security, "_face_embed", lambda frame: None)
    try:
        security.enroll_face(frame="not a real frame, just needs to be non-None")
        assert False, "should have raised when no face was detected"
    except ValueError:
        pass
    assert security.has_face() is False


def test_enroll_face_then_matching_embedding_logs_in(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    enrolled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(security, "_face_embed", lambda frame: enrolled)
    security.enroll_face(frame="fake-frame")
    assert security.has_face() is True

    monkeypatch.setattr(security, "capture_frame", lambda camera_index=0: "fake-live-frame")
    gate = security.SecurityGate()
    assert gate.face_login() is True


def test_face_login_denies_a_different_face(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    enrolled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(security, "_face_embed", lambda frame: enrolled)
    security.enroll_face(frame="fake-frame")

    # a very different embedding -> low cosine similarity -> below threshold
    different = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(security, "_face_embed", lambda frame: different)
    monkeypatch.setattr(security, "capture_frame", lambda camera_index=0: "fake-live-frame")
    gate = security.SecurityGate()
    assert gate.face_login() is False


def test_face_login_false_when_nothing_enrolled(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    gate = security.SecurityGate()
    assert gate.face_login() is False  # short-circuits before ever touching the camera


def test_face_login_false_when_no_face_in_live_frame(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    enrolled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(security, "_face_embed", lambda frame: enrolled)
    security.enroll_face(frame="fake-frame")

    monkeypatch.setattr(security, "_face_embed", lambda frame: None)  # no face this time
    monkeypatch.setattr(security, "capture_frame", lambda camera_index=0: "fake-live-frame")
    gate = security.SecurityGate()
    assert gate.face_login() is False


def test_face_login_handles_camera_error_gracefully(tmp_path, monkeypatch):
    _isolate_security_paths(tmp_path, monkeypatch)
    enrolled = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    monkeypatch.setattr(security, "_face_embed", lambda frame: enrolled)
    security.enroll_face(frame="fake-frame")

    def raise_camera_error(camera_index=0):
        raise RuntimeError("couldn't open camera 0")
    monkeypatch.setattr(security, "capture_frame", raise_camera_error)

    gate = security.SecurityGate()
    assert gate.face_login() is False  # never raises out to the caller
