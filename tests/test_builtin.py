from jarvis.modules.builtin import SpeechInput


class _FakeEngine:
    def __init__(self, transcript="hello"):
        self.record_calls = 0
        self.transcript = transcript

    def record_until_silence(self):
        self.record_calls += 1
        return "fake-audio"

    def transcribe(self, audio):
        assert audio == "fake-audio"
        return self.transcript


class _FakeMascot:
    def __init__(self):
        self.listen_while_calls = []

    def listen_while(self, fn, *args, **kwargs):
        self.listen_while_calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)


def test_listen_prints_a_listening_indicator(capsys):
    sk = SpeechInput(engine=_FakeEngine())
    sk.listen()
    out = capsys.readouterr().out
    assert "listening" in out.lower()


def test_listen_works_without_a_mascot():
    engine = _FakeEngine(transcript="what time is it")
    sk = SpeechInput(engine=engine)
    result = sk.listen()
    assert result == "what time is it"
    assert engine.record_calls == 1


def test_listen_routes_recording_through_the_mascot_when_present():
    engine = _FakeEngine(transcript="hi")
    mascot = _FakeMascot()
    sk = SpeechInput(engine=engine, mascot=mascot)

    result = sk.listen()

    assert result == "hi"
    assert engine.record_calls == 1
    assert len(mascot.listen_while_calls) == 1
    assert mascot.listen_while_calls[0][0] == engine.record_until_silence


def test_available_reflects_whether_an_engine_is_set():
    assert SpeechInput(engine=None).available is False
    assert SpeechInput(engine=_FakeEngine()).available is True
