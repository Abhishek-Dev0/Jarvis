from jarvis.modules.mascot import CatMascot, _FRAMES


def test_defaults_to_jarvis_persona():
    m = CatMascot()
    assert m.persona == "jarvis"


def test_unknown_persona_falls_back_to_jarvis():
    m = CatMascot(persona="totally-not-a-real-persona")
    assert m.persona == "jarvis"


def test_switch_to_known_persona_succeeds():
    m = CatMascot()
    assert m.switch("eve") is True
    assert m.persona == "eve"


def test_switch_to_unknown_persona_fails_and_leaves_persona_unchanged():
    m = CatMascot()
    assert m.switch("not-a-real-one") is False
    assert m.persona == "jarvis"


def test_idle_flourish_writes_output_when_enabled(capsys):
    m = CatMascot(enabled=True)
    m.idle_flourish(cycles=1, delay=0)
    out = capsys.readouterr().out
    assert out  # something was drawn


def test_idle_flourish_silent_when_disabled(capsys):
    m = CatMascot(enabled=False)
    m.idle_flourish(cycles=1, delay=0)
    out = capsys.readouterr().out
    assert out == ""


def test_talk_while_runs_fn_and_returns_its_result():
    m = CatMascot(enabled=False)  # disabled -> no real stdout timing noise in the test
    result = m.talk_while(lambda x: x * 2, 21)
    assert result == 42


def test_talk_while_animates_during_a_slow_call(capsys):
    import time
    m = CatMascot(enabled=True)

    def slow():
        time.sleep(0.3)
        return "done"

    result = m.talk_while(slow, delay=0.05)
    assert result == "done"
    out = capsys.readouterr().out
    assert out  # the animation thread drew at least one frame during the 0.3s call


def test_listen_while_runs_fn_and_returns_its_result():
    m = CatMascot(enabled=False)  # disabled -> no real stdout timing noise in the test
    result = m.listen_while(lambda x: x * 2, 21)
    assert result == 42


def test_listen_while_animates_during_a_slow_call(capsys):
    import time
    m = CatMascot(enabled=True)

    def slow():
        time.sleep(0.3)
        return "done"

    result = m.listen_while(slow, delay=0.05)
    assert result == "done"
    out = capsys.readouterr().out
    assert out  # the animation thread drew at least one frame during the 0.3s call


def test_all_persona_frame_sets_include_idle_talk_and_listen():
    for persona, kinds in _FRAMES.items():
        assert set(kinds.keys()) == {"idle", "talk", "listen"}, persona


def test_all_frame_sets_have_consistent_line_height():
    for persona, kinds in _FRAMES.items():
        for kind, frames in kinds.items():
            for frame in frames:
                lines = frame.rstrip("\n").split("\n")
                assert len(lines) == 3, f"{persona}/{kind} frame has {len(lines)} lines, expected 3"
