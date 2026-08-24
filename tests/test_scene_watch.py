import json

import numpy as np
import pytest
import torch

from jarvis.modules import scene_watch
from jarvis.modules.scene_watch import (
    ChangeDetector, SceneWatchSkill, identify_faces, load_events,
    load_known_faces, log_event, vision_context,
)


class _FakeFace:
    def __init__(self, embedding, bbox=(0, 0, 100, 100)):
        self.normed_embedding = np.asarray(embedding, dtype=np.float32)
        self.bbox = bbox


class _FakeFaceApp:
    def __init__(self, faces_by_frame):
        # frame -> list[_FakeFace]; matched by identity/equality on the frame object
        self._faces_by_frame = faces_by_frame

    def get(self, frame):
        for key, faces in self._faces_by_frame:
            if key is frame or key == frame:
                return faces
        return []


# --------------------------------------------------------------- identify_faces

def test_identify_faces_matches_known_embedding_above_threshold(monkeypatch):
    known = {"abhishek": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    frame = "frame-1"
    fake_app = _FakeFaceApp([(frame, [_FakeFace([1.0, 0.0, 0.0])])])
    monkeypatch.setattr(scene_watch, "get_face_app", lambda: fake_app)

    names = identify_faces(frame, known, threshold=0.9)
    assert names == ["abhishek"]


def test_identify_faces_rejects_a_different_face_below_threshold(monkeypatch):
    known = {"abhishek": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    frame = "frame-2"
    fake_app = _FakeFaceApp([(frame, [_FakeFace([0.0, 1.0, 0.0])])])
    monkeypatch.setattr(scene_watch, "get_face_app", lambda: fake_app)

    names = identify_faces(frame, known, threshold=0.9)
    assert names == []


def test_identify_faces_handles_multiple_people_in_one_frame(monkeypatch):
    known = {
        "abhishek": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "eve": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    frame = "frame-3"
    fake_app = _FakeFaceApp([(frame, [_FakeFace([1.0, 0.0, 0.0]), _FakeFace([0.0, 1.0, 0.0])])])
    monkeypatch.setattr(scene_watch, "get_face_app", lambda: fake_app)

    names = identify_faces(frame, known, threshold=0.9)
    assert set(names) == {"abhishek", "eve"}


def test_identify_faces_empty_known_faces_short_circuits(monkeypatch):
    # Must never touch the face app at all with nothing to match against.
    def boom():
        raise AssertionError("should not be called")
    monkeypatch.setattr(scene_watch, "get_face_app", boom)
    assert identify_faces("any-frame", {}) == []


# ------------------------------------------------------------- load_known_faces

def test_load_known_faces_reads_one_embedding_per_reference_photo(tmp_path, monkeypatch):
    (tmp_path / "abhishek.jpg").write_bytes(b"not a real jpg, just needs to exist")
    (tmp_path / "eve.png").write_bytes(b"not a real png either")
    (tmp_path / "notes.txt").write_bytes(b"should be ignored, not an image extension")

    monkeypatch.setattr("cv2.imread", lambda path: path)  # frame == its own path, for identity matching
    fake_app = _FakeFaceApp([
        (str(tmp_path / "abhishek.jpg"), [_FakeFace([1.0, 0.0, 0.0])]),
        (str(tmp_path / "eve.png"), [_FakeFace([0.0, 1.0, 0.0])]),
    ])
    monkeypatch.setattr(scene_watch, "get_face_app", lambda: fake_app)

    known = load_known_faces(str(tmp_path))
    assert set(known) == {"abhishek", "eve"}
    assert known["abhishek"] == pytest.approx([1.0, 0.0, 0.0])


def test_load_known_faces_skips_photos_with_no_detected_face(tmp_path, monkeypatch):
    (tmp_path / "blurry.jpg").write_bytes(b"placeholder")
    monkeypatch.setattr("cv2.imread", lambda path: path)
    monkeypatch.setattr(scene_watch, "get_face_app", lambda: _FakeFaceApp([]))  # no face found anywhere

    known = load_known_faces(str(tmp_path))
    assert known == {}


def test_load_known_faces_empty_dir_returns_empty_dict(tmp_path):
    assert load_known_faces(str(tmp_path / "does_not_exist")) == {}


# ---------------------------------------------------------------- ChangeDetector

def test_change_detector_first_call_just_sets_baseline_no_event():
    d = ChangeDetector()
    embedding = torch.tensor([1.0, 0.0, 0.0])
    assert d.update(embedding, now=1000.0) is None


def test_change_detector_fires_on_a_large_enough_shift():
    d = ChangeDetector(threshold=0.1, cooldown_s=10.0)
    d.update(torch.tensor([1.0, 0.0, 0.0]), now=1000.0)
    event = d.update(torch.tensor([0.0, 1.0, 0.0]), now=1001.0)  # orthogonal -> distance 1.0
    assert event is not None
    assert event["type"] == "scene_change"


def test_change_detector_stays_quiet_for_a_small_shift():
    d = ChangeDetector(threshold=0.5, cooldown_s=10.0)
    d.update(torch.tensor([1.0, 0.0, 0.0]), now=1000.0)
    event = d.update(torch.tensor([0.99, 0.01, 0.0]), now=1001.0)  # nearly identical
    assert event is None


def test_change_detector_respects_cooldown():
    d = ChangeDetector(threshold=0.1, cooldown_s=20.0)
    d.update(torch.tensor([1.0, 0.0, 0.0]), now=1000.0)
    first = d.update(torch.tensor([0.0, 1.0, 0.0]), now=1001.0)
    assert first is not None
    # a second big shift immediately after -- suppressed by cooldown
    second = d.update(torch.tensor([1.0, 0.0, 0.0]), now=1002.0)
    assert second is None


# --------------------------------------------------------------------- events

def test_log_event_and_load_round_trip(tmp_path):
    path = str(tmp_path / "events.json")
    log_event({"type": "face_seen", "names": ["abhishek"]}, path)
    events = load_events(path)
    assert len(events) == 1
    assert events[0]["type"] == "face_seen"
    assert "ts" in events[0]


def test_log_event_caps_at_fifty_entries(tmp_path):
    path = str(tmp_path / "events.json")
    for i in range(60):
        log_event({"type": "scene_change", "distance": i}, path)
    events = load_events(path)
    assert len(events) == 50
    assert events[-1]["distance"] == 59  # oldest dropped, newest kept


def test_load_events_empty_when_file_absent(tmp_path):
    assert load_events(str(tmp_path / "missing.json")) == []


def test_vision_context_empty_when_nothing_logged(tmp_path):
    assert vision_context(str(tmp_path / "missing.json")) == ""


def test_vision_context_frames_events_as_background_information(tmp_path):
    path = str(tmp_path / "events.json")
    log_event({"type": "face_seen", "names": ["abhishek"]}, path)
    ctx = vision_context(path)
    assert "background information" in ctx
    assert "abhishek" in ctx


# ------------------------------------------------ scene-encoder auto-resolve

def test_scene_encoder_forced_on_regardless_of_cuda():
    sk = SceneWatchSkill(use_scene_encoder=True)
    assert sk._resolve_use_scene_encoder() is True


def test_scene_encoder_forced_off_regardless_of_cuda():
    sk = SceneWatchSkill(use_scene_encoder=False)
    assert sk._resolve_use_scene_encoder() is False


def test_scene_encoder_auto_follows_cuda_availability(monkeypatch):
    # Real finding from measuring this on the dev machine: V-JEPA2 took
    # 68-90s/clip on CPU, unusable for an ambient cadence -- auto mode must
    # not silently enable it without a GPU.
    import torch
    sk = SceneWatchSkill(use_scene_encoder=None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert sk._resolve_use_scene_encoder() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert sk._resolve_use_scene_encoder() is True


# --------------------------------------------------------------- SceneWatchSkill

def test_skill_disabled_by_default():
    sk = SceneWatchSkill()
    assert sk.enabled is False


def test_enabling_the_skill_starts_the_camera_thread(monkeypatch):
    started = []
    stopped = []

    class _FakeCamera:
        def start(self):
            started.append(True)
            return self

        def stop(self):
            stopped.append(True)

        def latest(self, timeout=1.0):
            return None

    monkeypatch.setattr(scene_watch, "CameraStream", lambda *a, **k: _FakeCamera())
    monkeypatch.setattr(scene_watch, "load_known_faces", lambda *a, **k: {})

    sk = SceneWatchSkill(use_scene_encoder=False)
    sk.enabled = True
    assert started == [True]
    assert sk.enabled is True

    sk.enabled = False
    assert stopped == [True]
    assert sk.enabled is False


def test_setting_enabled_to_its_current_value_is_a_no_op(monkeypatch):
    calls = []
    monkeypatch.setattr(SceneWatchSkill, "_start_watch", lambda self: calls.append("start"))
    monkeypatch.setattr(SceneWatchSkill, "_stop_watch", lambda self: calls.append("stop"))

    sk = SceneWatchSkill()
    sk.enabled = False  # already False -- must not call _stop_watch
    assert calls == []


def test_matches_who_is_in_the_room_and_recent_events_triggers():
    sk = SceneWatchSkill()
    assert sk.matches("who's in the room")
    assert sk.matches("recent vision events")
    assert not sk.matches("what's the weather")


def test_handle_recent_events_reads_the_log(tmp_path):
    events_path = str(tmp_path / "events.json")
    log_event({"type": "face_seen", "names": ["abhishek"]}, events_path)
    sk = SceneWatchSkill(events_path=events_path)
    reply = sk.handle("recent vision events")
    assert "abhishek" in reply


def test_handle_who_is_in_the_room_with_no_known_faces_says_so():
    sk = SceneWatchSkill(known_faces_dir=str("this/does/not/exist"))
    reply = sk.handle("who's in the room")
    assert "No known faces" in reply


def test_handle_who_is_in_the_room_uses_a_live_frame(monkeypatch, tmp_path):
    class _FakeCamera:
        def start(self):
            return self

        def latest(self, timeout=1.0):
            return "a-frame"

        def stop(self):
            pass

    monkeypatch.setattr(scene_watch, "CameraStream", lambda *a, **k: _FakeCamera())
    monkeypatch.setattr(scene_watch, "identify_faces", lambda frame, known, threshold=None: ["abhishek"])
    monkeypatch.setattr("time.sleep", lambda s: None)

    sk = SceneWatchSkill(known_faces_dir=str(tmp_path))
    sk._known_faces = {"abhishek": np.array([1.0, 0.0, 0.0])}  # skip real load_known_faces
    reply = sk.handle("who's in the room")
    assert "abhishek" in reply
