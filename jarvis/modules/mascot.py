"""
mascot.py — an animated ASCII cat in the terminal: JARVIS's visual presence
alongside its voice. Idle animation (tail wag) played as a short flourish
after each reply, mouth-flap animation synced to actual TTS playback
duration while speaking, switchable between the "jarvis" and "eve"
personas via plain conversation ("switch to eve") — not gated, it's
cosmetic, not a security-relevant action, so any user can say it.

Redraw safety: every animation "session" (one idle_flourish() call, or one
talk_while() call) starts with a fresh print — no cursor movement assumed
— and only frames *within* that same session use cursor-up-and-overwrite
to redraw in place. That's what keeps this from ever corrupting unrelated
console output: between sessions, arbitrary other text may have been
printed (skill replies, "you>" prompts, module load lines), so each new
session re-anchors itself instead of blindly overwriting whatever is now
sitting where the cat last was.

Windows consoles need ANSI/VT100 escape processing explicitly enabled —
off by default on some Windows consoles even today. _enable_vt100() does
that once via SetConsoleMode; harmless no-op if it's already on, and a
silent no-op entirely on non-Windows.
"""

from __future__ import annotations

import sys
import threading
import time

_HEIGHT = 3  # lines per frame — every frame below must be exactly this many lines


def _enable_vt100() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


_FRAMES = {
    "jarvis": {
        "idle": [
            " /\\_/\\      \n( o.o )    ~\n > ^ <  ~   \n",
            " /\\_/\\   ~  \n( o.o ) ~   \n > ^ <      \n",
            " /\\_/\\  ~   \n( o.o )    ~\n > ^ <    ~ \n",
        ],
        "talk": [
            " /\\_/\\      \n( o.o )     \n > ^ <      \n",
            " /\\_/\\      \n( o.o )     \n > 3 <      \n",
        ],
    },
    "eve": {
        "idle": [
            " /\\_/\\)     \n( o.o )    ~\n > . <  ~   \n",
            " /\\_/\\)  ~  \n( o.o ) ~   \n > . <      \n",
            " /\\_/\\) ~   \n( o.o )    ~\n > . <    ~ \n",
        ],
        "talk": [
            " /\\_/\\)     \n( o.o )     \n > . <      \n",
            " /\\_/\\)     \n( o.o )     \n > 3 <      \n",
        ],
    },
}


class CatMascot:
    """persona: "jarvis" or "eve" — matches modules/voice.py's PersonaTTSEngine
    personas, kept in sync by whatever calls switch() (see runtime/jarvis.py's
    "switch to eve"/"switch to jarvis" trigger)."""

    def __init__(self, persona: str = "jarvis", enabled: bool = True):
        _enable_vt100()
        self.persona = persona if persona in _FRAMES else "jarvis"
        self.enabled = enabled
        self._lock = threading.Lock()
        self._drawn = False

    def switch(self, persona: str) -> bool:
        if persona not in _FRAMES:
            return False
        self.persona = persona
        return True

    def _draw(self, frame: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._drawn:
                sys.stdout.write(f"\033[{_HEIGHT}A")
            sys.stdout.write("\r" + frame)
            sys.stdout.flush()
            self._drawn = True

    def idle_flourish(self, cycles: int = 1, delay: float = 0.2) -> None:
        """A short tail-wag, meant to be played once right after a reply."""
        if not self.enabled:
            return
        frames = _FRAMES[self.persona]["idle"]
        self._drawn = False  # fresh session -- other output may have happened since the last draw
        for _ in range(cycles):
            for frame in frames:
                self._draw(frame)
                time.sleep(delay)

    def talk_while(self, fn, *args, delay: float = 0.15, **kwargs):
        """Runs fn(*args, **kwargs) (e.g. a blocking TTS speak() call) while
        flapping the mascot's mouth in sync with it; returns fn's result.
        Animation still starts/stops correctly even with mascot disabled
        (enabled=False) or in headless test contexts — _draw() just no-ops."""
        frames = _FRAMES[self.persona]["talk"]
        self._drawn = False
        stop = threading.Event()

        def _animate():
            i = 0
            while not stop.is_set():
                self._draw(frames[i % len(frames)])
                i += 1
                stop.wait(delay)

        thread = threading.Thread(target=_animate, daemon=True)
        thread.start()
        try:
            return fn(*args, **kwargs)
        finally:
            stop.set()
            thread.join(timeout=1.0)
            self._draw(_FRAMES[self.persona]["idle"][0])
