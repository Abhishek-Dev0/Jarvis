from jarvis.modules.summarize import fold_turn_into_summary


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fold_turn_into_summary_returns_the_model_response(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse({"response": "Abi asked about tokenizers; JARVIS explained BPE."})

    monkeypatch.setattr("requests.post", fake_post)

    updated = fold_turn_into_summary("", "what's BPE?", "It's byte pair encoding.")
    assert updated == "Abi asked about tokenizers; JARVIS explained BPE."
    assert "what's BPE?" in captured["prompt"]
    assert "It's byte pair encoding." in captured["prompt"]
    assert "(none yet)" in captured["prompt"]


def test_fold_turn_into_summary_includes_previous_summary_in_prompt(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(json)
        return _FakeResponse({"response": "updated summary"})

    monkeypatch.setattr("requests.post", fake_post)

    fold_turn_into_summary("Abi's GPU is an RTX 3050.", "what OS am I on?", "Windows 11.")
    assert "Abi's GPU is an RTX 3050." in captured["prompt"]


def test_fold_turn_into_summary_returns_previous_unchanged_on_failure(monkeypatch):
    def fake_post(url, json, timeout):
        raise ConnectionError("no server")

    monkeypatch.setattr("requests.post", fake_post)

    updated = fold_turn_into_summary("existing summary", "hi", "hello")
    assert updated == "existing summary"


def test_fold_turn_into_summary_returns_previous_unchanged_on_empty_response(monkeypatch):
    def fake_post(url, json, timeout):
        return _FakeResponse({"response": "   "})

    monkeypatch.setattr("requests.post", fake_post)

    updated = fold_turn_into_summary("existing summary", "hi", "hello")
    assert updated == "existing summary"
