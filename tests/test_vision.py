import os

from jarvis.modules.vision import VisionSkill

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_REAL_IMAGE = os.path.join(_REPO_ROOT, "assets", "jarvis_cat.ico")


class _FakeResponse:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def test_matches_known_trigger_phrases():
    sk = VisionSkill()
    assert sk.matches("describe image foo.png") is True
    assert sk.matches("what's in image bar.jpg") is True
    assert sk.matches("tell me a joke") is False


def test_handle_with_no_path_prompts():
    sk = VisionSkill()
    reply = sk.handle("describe image")
    assert "Describe which image" in reply


def test_handle_missing_file_reports_clearly():
    sk = VisionSkill()
    reply = sk.handle("describe image definitely_not_a_real_file.png")
    assert reply.startswith("No such file:")


def test_handle_splits_path_and_question_on_colon_space(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse("ok")

    monkeypatch.setattr("requests.post", fake_post)

    sk = VisionSkill()
    reply = sk.handle(f"describe image {_REAL_IMAGE}: is there a cat")
    assert reply == "ok"
    assert captured["messages"][0]["content"] == "is there a cat"


def test_handle_sends_default_question_when_none_given(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse("a small cat icon")

    monkeypatch.setattr("requests.post", fake_post)

    sk = VisionSkill()
    reply = sk.handle(f"describe image {_REAL_IMAGE}")
    assert reply == "a small cat icon"
    assert captured["messages"][0]["content"] == "Describe this image."
    assert len(captured["messages"][0]["images"]) == 1


def test_handle_sends_custom_question(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse("yes, there is a cat")

    monkeypatch.setattr("requests.post", fake_post)

    sk = VisionSkill()
    reply = sk.handle(f"describe image {_REAL_IMAGE}: is there a cat in this?")
    assert reply == "yes, there is a cat"
    assert captured["messages"][0]["content"] == "is there a cat in this?"


def test_handle_reports_unreachable_ollama_gracefully(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("no server")

    monkeypatch.setattr("requests.post", fake_post)

    sk = VisionSkill()
    reply = sk.handle(f"describe image {_REAL_IMAGE}")
    assert "isn't responding" in reply
    assert "ollama serve" in reply


def test_windows_path_without_question_is_not_split_on_drive_colon(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse("ok")

    monkeypatch.setattr("requests.post", fake_post)

    sk = VisionSkill()
    # a bare Windows-style absolute path with no question -- must NOT be
    # mistaken for a "<path>: <question>" split on the drive-letter colon
    reply = sk.handle(f"describe image {_REAL_IMAGE}")
    assert reply == "ok"
    assert captured["messages"][0]["content"] == "Describe this image."
