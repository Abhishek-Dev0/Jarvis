"""io_adapter.py — the output sink that lets the GUI receive everything
Jarvis would otherwise print to a console: streamed model tokens during
Jarvis.respond(stream=True), and any registry.emit_all() side-channel
message (security/admin flows, "switched to eve", etc.). Qt signals cross
the worker-thread -> UI-thread boundary safely; plain callbacks would not.

Deliberately NOT a modules.base.OutputModule subclass: OutputModule's ABC
metaclass conflicts with QObject's (PySide6's Shiboken metaclass) — Python
can't build a class with both. Registry.register() special-cases the
isinstance check to route modules into inputs/outputs/skills, so this
duck-types the same emit()/emit_stream()/flush() interface and gets
appended straight into registry.outputs (see app.py) instead of going
through register()."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class GuiOutput(QObject):
    name = "gui_out"
    description = "routes Jarvis output into the chat panel"
    available = True

    message_ready = Signal(str)   # a complete, non-streamed message (emit())
    chunk_ready = Signal(str)     # one streamed token/chunk (emit_stream())
    stream_finished = Signal()    # the current streamed reply is complete (flush())

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def emit(self, text: str) -> None:
        self.message_ready.emit(text)

    def emit_stream(self, chunk: str) -> None:
        self.chunk_ready.emit(chunk)

    def flush(self) -> None:
        self.stream_finished.emit()
