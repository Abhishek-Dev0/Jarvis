"""workers.py — QThread wrappers for anything that blocks (model generation,
Ollama calls, mic recording, TTS playback). Nothing long-running may run on
the Qt UI thread or the window freezes; every one of these emits a Qt
signal back to the UI thread when done rather than returning a value
directly, since Qt objects aren't safe to touch cross-thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class RespondWorker(QThread):
    """Runs Jarvis.respond() off the UI thread. For a skill-handled request
    the reply arrives whole via `finished`; for a raw-model reply, tokens
    also stream through the GuiOutput registered as an output module
    (connect to that separately) while this still emits `finished` with the
    complete text once generation ends."""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, jarvis, text: str, parent=None):
        super().__init__(parent)
        self.jarvis = jarvis
        self.text = text

    def run(self):
        try:
            reply = self.jarvis.respond(self.text, stream=True)
            self.finished_ok.emit(reply)
        except Exception as e:
            self.failed.emit(str(e))


class ListenWorker(QThread):
    """Push-to-talk: records until the whisper engine's own VAD detects
    silence, then transcribes. Mirrors modules/builtin.py's SpeechInput
    but on-demand rather than in a blocking loop."""

    transcribed = Signal(str)
    failed = Signal(str)

    def __init__(self, whisper_engine, parent=None):
        super().__init__(parent)
        self.whisper_engine = whisper_engine

    def run(self):
        try:
            audio = self.whisper_engine.record_until_silence()
            text = self.whisper_engine.transcribe(audio)
            self.transcribed.emit(text or "")
        except Exception as e:
            self.failed.emit(str(e))


class SpeakWorker(QThread):
    """Speaks a reply aloud via the persona TTS engine. Fire-and-forget from
    the UI's perspective; `finished_speaking` just lets the mic button
    re-enable itself instead of overlapping recording with playback."""

    finished_speaking = Signal()
    failed = Signal(str)

    def __init__(self, persona_engine, text: str, lang: str | None = None, parent=None):
        super().__init__(parent)
        self.persona_engine = persona_engine
        self.text = text
        self.lang = lang

    def run(self):
        try:
            self.persona_engine.speak(self.text, lang=self.lang)
            self.finished_speaking.emit()
        except Exception as e:
            self.failed.emit(str(e))


class CommandWorker(QThread):
    """Runs a single shell command via subprocess and streams combined
    stdout/stderr back line-by-line. Used by the Terminal tab — a real
    command runner, not a full pty/ANSI terminal emulator (see the plan's
    scoping note)."""

    line_ready = Signal(str)
    finished_ok = Signal(int)
    failed = Signal(str)

    def __init__(self, command: str, cwd: str, parent=None):
        super().__init__(parent)
        self.command = command
        self.cwd = cwd
        self._process = None

    def run(self):
        import subprocess
        try:
            self._process = subprocess.Popen(
                self.command, cwd=self.cwd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            for line in self._process.stdout:
                self.line_ready.emit(line.rstrip("\n"))
            code = self._process.wait()
            self.finished_ok.emit(code)
        except Exception as e:
            self.failed.emit(str(e))

    def stop(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()


class MarketDataWorker(QThread):
    """Fetches historical OHLCV (yfinance, via modules/market_analysis.py's
    fetch_history) off the UI thread -- real network I/O, would freeze the
    window otherwise. Used by the GUI's Markets tab."""

    finished_ok = Signal(object)  # pandas DataFrame
    failed = Signal(str)

    def __init__(self, symbol: str, period: str, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.period = period

    def run(self):
        try:
            try:
                from ..modules.market_analysis import fetch_history
            except ImportError:  # pragma: no cover - legacy direct execution
                from modules.market_analysis import fetch_history
            df = fetch_history(self.symbol, period=self.period)
            self.finished_ok.emit(df)
        except Exception as e:
            self.failed.emit(str(e))


class NewsSearchWorker(QThread):
    """Live web search (modules/market_analysis.py's market_news(), which
    just calls modules/web.py's existing search() -- no new scraper) off
    the UI thread. Used by the GUI's Markets tab news panel."""

    finished_ok = Signal(list)  # [{title, url, snippet}]
    failed = Signal(str)

    def __init__(self, query: str, parent=None):
        super().__init__(parent)
        self.query = query

    def run(self):
        try:
            try:
                from ..modules.market_analysis import market_news
            except ImportError:  # pragma: no cover - legacy direct execution
                from modules.market_analysis import market_news
            results = market_news(self.query)
            self.finished_ok.emit(results)
        except Exception as e:
            self.failed.emit(str(e))
