"""
hardware_io.py — serial link to external hardware: Arduino-class boards,
sensor breakouts, motor controllers. This is the general-purpose "talk to a
physical device" layer Abi asked for on 2026-08-22 for his robot project
("Loki", per the top-level README's suggested order of work) and any future
exoskeleton/actuator work — one transport, reused by whatever's plugged in,
same philosophy as every other module here: attaches from outside, core/
never knows it exists.

Two halves, deliberately kept apart:

  SerialLink      raw transport — connect, read lines, write lines. No
                   opinion about what the bytes mean. Reused by anything:
                   an EMG board, a motor driver, a temperature sensor.

  HardwareSkill    the conversational surface ("list serial ports", "connect
                   to arduino", "read arduino", "send X to arduino") — same
                   trigger-phrase-routing pattern as CalculatorSkill/
                   OSControlSkill. Connecting and sending are gated through
                   SecurityGate, same reasoning as os_control.py: this can
                   make a physical device do something, which is the same
                   risk category as launching/killing a process, arguably
                   higher once anything actuated is wearable. Listing ports
                   and reading the latest line are read-only, ungated.

IMPORTANT — what this module does NOT do: send_raw() has no safety clamp.
It writes exactly what you give it, unbounded. That's correct for a
conversational "send this debug command" skill where a human is in the loop
reading the reply, and wrong for anything that closes a loop autonomously
(e.g. EMG signal -> motor command with no human in between) — a control
loop like that needs its own explicit bounds/kill-switch built at the
call site, not borrowed from here. See modules/biosignal.py for the signal-
processing half (EMG envelope + activity detection) this is meant to pair
with; that module is pure math with no actuation path of its own, on
purpose, for the same reason.

Not a certified or regulated medical device. Fine for hobbyist/robotics
muscle-activity sensing (driving a prosthetic or exoskeleton actuator you
built and are testing yourself); not validated for clinical or diagnostic
use, and nothing here claims to be.
"""

from __future__ import annotations

import threading
import time

try:
    from .base import SkillModule
    from ..security import authorize_action
except ImportError:  # pragma: no cover - legacy direct execution
    from base import SkillModule
    from security import authorize_action

# Common Arduino-family USB-serial chip/board identifiers, lowercased, matched
# against pyserial's port description/manufacturer strings. Not exhaustive —
# an unrecognized board still works via --port/connect(port=...) explicitly,
# this list only drives the "just guess it for me" auto-detect path.
_ARDUINO_HINTS = (
    "arduino", "ch340", "ch341", "cp210", "ftdi", "usb-serial", "usb serial",
    "wch.cn", "silicon labs",
)


def list_serial_ports() -> list[dict]:
    """Every serial port pyserial can see right now, with a best-guess
    is_likely_arduino flag. Read-only, no connection opened."""
    import serial.tools.list_ports
    ports = []
    for p in serial.tools.list_ports.comports():
        desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
        ports.append({
            "device": p.device,
            "description": p.description,
            "manufacturer": p.manufacturer,
            "is_likely_arduino": any(h in desc for h in _ARDUINO_HINTS),
        })
    return ports


class SerialLink:
    """One serial connection. Line-oriented (reads/writes text lines,
    newline-terminated) — the common case for Arduino sketches that print
    sensor readings with Serial.println() and read commands with
    Serial.readStringUntil('\\n')."""

    def __init__(self, port: str | None = None, baud: int = 115200, timeout: float = 1.0):
        # port=None -> connect() auto-picks the first is_likely_arduino hit.
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._conn = None
        self._stream_thread: threading.Thread | None = None
        self._stream_stop = threading.Event()
        self.last_line: str | None = None

    @property
    def available(self) -> bool:
        try:
            import serial  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def connected(self) -> bool:
        return self._conn is not None and self._conn.is_open

    def connect(self) -> str:
        """Opens the link. Returns the port actually used. Raises if
        port=None and nothing Arduino-like is plugged in, or if the named
        port can't be opened."""
        import serial
        port = self.port
        if port is None:
            candidates = [p for p in list_serial_ports() if p["is_likely_arduino"]]
            if not candidates:
                raise RuntimeError("no Arduino-like serial device found — plug one in, "
                                    "or pass an explicit port")
            port = candidates[0]["device"]
        self._conn = serial.Serial(port, self.baud, timeout=self.timeout)
        self.port = port
        # Most boards reset on serial open; give the sketch a moment to boot
        # before anything is written to or read from it.
        time.sleep(2.0)
        return port

    def close(self) -> None:
        self.stop_stream()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def read_line(self) -> str | None:
        """One line, blocking up to `timeout` seconds. None on timeout or if
        not connected."""
        if not self.connected:
            return None
        raw = self._conn.readline()
        if not raw:
            return None
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            self.last_line = line
        return line or None

    def send_raw(self, text: str) -> None:
        """Writes `text` + newline, exactly as given. No validation, no
        safety clamp — see the module docstring. Raises if not connected."""
        if not self.connected:
            raise RuntimeError("not connected")
        self._conn.write((text.strip() + "\n").encode("utf-8"))

    def start_stream(self, callback) -> None:
        """Background thread: calls callback(line) for every non-empty line
        received, until stop_stream() or close(). For continuous sensor
        polling (e.g. an EMG board printing one reading per line) without
        blocking the caller."""
        if not self.connected:
            raise RuntimeError("not connected")
        self._stream_stop.clear()

        def _loop():
            while not self._stream_stop.is_set() and self.connected:
                try:
                    line = self.read_line()
                except Exception as e:
                    # e.g. the port was closed from another thread while
                    # this thread was blocked inside readline(). Exit
                    # quietly instead of dying with an unhandled exception
                    # in a daemon thread.
                    print(f"[hardware_io] stream read failed, stopping: {e}")
                    return
                if line:
                    try:
                        callback(line)
                    except Exception as e:
                        print(f"[hardware_io] stream callback raised: {e}")

        self._stream_thread = threading.Thread(target=_loop, daemon=True)
        self._stream_thread.start()

    def stop_stream(self) -> None:
        self._stream_stop.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None


# --------------------------------------------------------------------- skill

_LIST_TRIGGERS = {"list serial ports", "list ports", "what's connected", "whats connected"}
_CONNECT_TRIGGERS = ("connect to arduino", "connect to serial", "connect arduino")
_SEND_TRIGGERS = ("send to arduino", "send arduino", "arduino send")
_READ_TRIGGERS = {"read arduino", "read serial", "last arduino reading"}


class HardwareSkill(SkillModule):
    """Conversational surface over one SerialLink. Silently unavailable
    (registry skips it) if pyserial isn't installed — same graceful-
    degradation pattern as every other optional module here."""

    name = "hardware"
    description = "connects to and talks over a serial link (Arduino etc.), security-gated"
    priority = 9  # same tier as os_control — also a physical-side-effect skill

    def __init__(self, security_ref=None, is_admin_ref=None, baud: int = 115200):
        self.security_ref = security_ref
        self.is_admin_ref = is_admin_ref
        self.baud = baud
        self._link: SerialLink | None = None

    @property
    def available(self) -> bool:
        try:
            import serial  # noqa: F401
            return True
        except ImportError:
            return False

    def teardown(self) -> None:
        if self._link is not None:
            self._link.close()

    def _authorized(self, reason: str) -> bool:
        return authorize_action(reason, self.security_ref, self.is_admin_ref)

    def matches(self, text: str) -> bool:
        t = text.strip().lower()
        if t in _LIST_TRIGGERS or t in _READ_TRIGGERS:
            return True
        return any(t.startswith(p) for p in (*_CONNECT_TRIGGERS, *_SEND_TRIGGERS))

    def handle(self, text: str) -> str:
        t = text.strip().lower()

        if t in _LIST_TRIGGERS:
            return self._list()
        if t in _READ_TRIGGERS:
            return self._read()
        for p in _CONNECT_TRIGGERS:
            if t.startswith(p):
                return self._connect(text.strip()[len(p):].strip())
        for p in _SEND_TRIGGERS:
            if t.startswith(p):
                return self._send(text.strip()[len(p):].strip())
        return "I didn't catch that hardware command."

    def _list(self) -> str:
        ports = list_serial_ports()
        if not ports:
            return "No serial ports detected."
        lines = ["Serial ports:"]
        for p in ports:
            tag = " (likely Arduino)" if p["is_likely_arduino"] else ""
            lines.append(f"  - {p['device']}: {p['description']}{tag}")
        return "\n".join(lines)

    def _connect(self, port_hint: str) -> str:
        if not self._authorized("connect to a serial device"):
            return "Denied — couldn't verify you for connecting to hardware."
        if self._link is not None:
            # Close any previous connection first — otherwise its handle
            # (and, on Windows, its exclusive lock on the COM port) leaks
            # silently, and reconnecting to the same port then fails.
            self._link.close()
        port = port_hint or None
        self._link = SerialLink(port=port, baud=self.baud)
        try:
            used = self._link.connect()
            return f"Connected on {used}."
        except Exception as e:
            self._link = None
            return f"Couldn't connect ({e})."

    def _read(self) -> str:
        if self._link is None or not self._link.connected:
            return "Not connected to anything. Say \"connect to arduino\" first."
        line = self._link.read_line()
        return line if line else "No data received (timed out)."

    def _send(self, payload: str) -> str:
        if self._link is None or not self._link.connected:
            return "Not connected to anything. Say \"connect to arduino\" first."
        if not payload:
            return "Send what?"
        if not self._authorized(f"send '{payload}' to connected hardware"):
            return "Denied — couldn't verify you for sending a hardware command."
        try:
            self._link.send_raw(payload)
            return f"Sent: {payload}"
        except Exception as e:
            return f"Send failed ({e})."
