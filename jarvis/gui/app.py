"""app.py — GUI entry point. Builds the same Jarvis instance the console
frontend builds (see runtime/jarvis.py's build_jarvis()), then shows the
window instead of running the console loop."""

from __future__ import annotations

import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

try:
    from ..runtime.jarvis import _build_arg_parser, build_jarvis
except ImportError:  # pragma: no cover - legacy direct execution
    from runtime.jarvis import _build_arg_parser, build_jarvis

from .io_adapter import GuiOutput
from .main_window import MainWindow
from .theme import STYLESHEET

_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_GUI_DIR)


def _icon_path() -> str | None:
    """assets/jarvis_cat.ico, resolved whether running from source (repo
    root next to jarvis/) or frozen by PyInstaller (bundled under
    sys._MEIPASS — see packaging/jarvis_gui.spec's datas)."""
    base = sys._MEIPASS if getattr(sys, "frozen", False) else os.path.dirname(_PKG_DIR)
    path = os.path.join(base, "assets", "jarvis_cat.ico")
    return path if os.path.exists(path) else None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _build_arg_parser().parse_args(argv)
    # The GUI defaults to trying voice on (a working mic is part of the
    # point) even though console mode defaults it off — falls back to
    # text-only below if the voice stack (faster-whisper/Kokoro) isn't
    # installed, rather than crashing the whole app over it.
    args.voice = True

    app = QApplication([sys.argv[0], *argv])
    app.setStyleSheet(STYLESHEET)
    icon_path = _icon_path()
    if icon_path is not None:
        app.setWindowIcon(QIcon(icon_path))

    try:
        j, scanner = build_jarvis(args, register_io=False)
    except Exception as e:
        print(f"[gui] voice engines unavailable ({e}) — continuing without voice")
        args.voice = False
        j, scanner = build_jarvis(args, register_io=False)

    gui_output = GuiOutput()
    # Appended directly rather than via j.register(): GuiOutput can't be an
    # OutputModule subclass (QObject/ABC metaclass conflict — see
    # io_adapter.py), so it skips register()'s isinstance-based dispatch.
    j.registry.outputs.append(gui_output)

    prefs_path = os.path.join(_PKG_DIR, "data", "gui_prefs.json")
    window = MainWindow(j, gui_output, prefs_path)
    window.show()

    exit_code = app.exec()
    if scanner is not None:
        scanner.stop()
    for skill in j.registry.skills:
        if skill.name == "scene_watch":
            skill.enabled = False  # releases the camera if it was toggled on
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
