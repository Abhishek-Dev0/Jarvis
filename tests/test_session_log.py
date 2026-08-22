import io
import sys

from jarvis.runtime.jarvis import _TeeLog, _LOG_REDACTIONS, _stop_session_log


def test_tee_writes_to_both_stream_and_file():
    console = io.StringIO()
    logfile = io.StringIO()
    tee = _TeeLog(console, logfile)

    tee.write("hello world\n")

    assert console.getvalue() == "hello world\n"
    assert logfile.getvalue() == "hello world\n"


def test_tee_redacts_heard_passphrase_in_file_only():
    console = io.StringIO()
    logfile = io.StringIO()
    tee = _TeeLog(console, logfile)

    line = '[security] heard: "there is no god" — say this back to verify later.\n'
    tee.write(line)

    assert console.getvalue() == line  # unredacted on the real console
    assert "there is no god" not in logfile.getvalue()
    assert "[REDACTED]" in logfile.getvalue()
    assert "say this back to verify later" in logfile.getvalue()  # rest of the line intact


def test_tee_does_not_redact_unrelated_security_lines():
    console = io.StringIO()
    logfile = io.StringIO()
    tee = _TeeLog(console, logfile)

    line = "[security] voiceprint enrolled from the same recording.\n"
    tee.write(line)

    assert logfile.getvalue() == line  # untouched -- no secret in this line


def test_tee_forwards_unknown_attributes_to_wrapped_stream():
    class FakeStream:
        def __init__(self):
            self.reconfigured_with = None

        def write(self, data):
            pass

        def reconfigure(self, **kwargs):
            self.reconfigured_with = kwargs

        def isatty(self):
            return True

    stream = FakeStream()
    tee = _TeeLog(stream, io.StringIO())

    tee.reconfigure(encoding="utf-8", errors="replace")
    assert stream.reconfigured_with == {"encoding": "utf-8", "errors": "replace"}
    assert tee.isatty() is True


def test_redaction_patterns_do_not_touch_ordinary_text():
    console = io.StringIO()
    logfile = io.StringIO()
    tee = _TeeLog(console, logfile)

    tee.write("jarvis> Systems online. How can I help you, sir?\n")
    tee.write("you> what is 2 plus 2\n")

    assert logfile.getvalue() == "jarvis> Systems online. How can I help you, sir?\nyou> what is 2 plus 2\n"


def test_at_least_one_redaction_pattern_is_registered():
    # a canary against silently losing the redaction list in a future edit
    assert len(_LOG_REDACTIONS) >= 1


def test_stop_session_log_restores_real_streams_and_closes_file(tmp_path):
    real_stdout, real_stderr = sys.stdout, sys.stderr
    logfile = open(tmp_path / "session.log", "w", encoding="utf-8")
    try:
        sys.stdout = _TeeLog(real_stdout, logfile)
        sys.stderr = _TeeLog(real_stderr, logfile)

        _stop_session_log()

        assert sys.stdout is real_stdout
        assert sys.stderr is real_stderr
        assert logfile.closed
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr
        if not logfile.closed:
            logfile.close()


def test_stop_session_log_is_a_safe_no_op_without_a_tee():
    real_stdout, real_stderr = sys.stdout, sys.stderr
    _stop_session_log()  # nothing wrapped -- must not raise
    assert sys.stdout is real_stdout
    assert sys.stderr is real_stderr
