"""
biosignal.py — muscle-activity detection from a raw EMG (electromyography)
signal: the "analyze when the muscle is moving or not" ask from 2026-08-22,
for driving a prosthetic/exoskeleton actuator from surface EMG electrodes.

Pure signal processing, no hardware dependency of its own — feed it samples
from anywhere (a SerialLink in modules/hardware_io.py reading an EMG breakout
board like a MyoWare, a CSV replay, a unit test's synthetic sine wave). It
never sends a command anywhere; wiring its output to an actuator is a
deliberate separate step at the call site, same reasoning as
hardware_io.py's send_raw() having no built-in clamp — the piece that decides
"is the muscle active" should not be the same piece that decides "so move the
motor," because the second one needs its own bounds this module can't know.

Method: rectify -> moving-RMS envelope -> two-threshold (hysteresis)
classifier. This is the standard textbook approach to EMG onset detection
(e.g. surface EMG threshold detection as used in myoelectric prosthesis
control), not a novel algorithm — the two thresholds exist specifically to
stop noise near a single cutoff from flickering the output on/off every
sample.

Not a certified or regulated medical device, and not validated for clinical/
diagnostic use — this is a hobbyist/robotics tool for a signal you're
generating and wiring up yourself. If "medical" ever means an actual clinical
context rather than a personal robotics project, that is a different
regulatory category (FDA/CE class II+ medical device rules) this code makes
no attempt to satisfy.
"""

from __future__ import annotations

from collections import deque


class EnvelopeFilter:
    """Rectified moving-RMS envelope of a raw signal. Smooths a noisy raw
    EMG trace into a slower-moving "how much muscle activity right now"
    curve — the input a threshold classifier can actually work with."""

    def __init__(self, window: int = 20):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, sample: float) -> float:
        """Feed one raw sample, get back the current envelope value."""
        self._buf.append(sample * sample)
        mean_sq = sum(self._buf) / len(self._buf)
        return mean_sq ** 0.5

    def reset(self) -> None:
        self._buf.clear()


class MuscleActivityDetector:
    """Envelope + hysteresis threshold -> boolean "is the muscle active
    right now" state, debounced against flicker.

    on_threshold: envelope must rise above this to become "active".
    off_threshold: envelope must fall below this to return to "resting".
        Defaults to 70% of on_threshold if not given — the gap is what
        prevents noise sitting right at one cutoff from toggling state every
        sample. Must be < on_threshold.
    min_active_samples / min_rest_samples: require the envelope to stay past
        a threshold for this many consecutive samples before the reported
        state actually flips — filters out single-sample spikes. Defaults to
        1 (flip immediately), raise if your signal is noisy.

    Calibrate on_threshold/off_threshold from your own sensor and baseline —
    raw EMG amplitude varies enormously by electrode placement, skin contact,
    and hardware gain. There is no universal default that's correct for a
    board and setup this code has never seen.
    """

    def __init__(self, on_threshold: float, off_threshold: float | None = None,
                 window: int = 20, min_active_samples: int = 1, min_rest_samples: int = 1):
        if off_threshold is None:
            off_threshold = on_threshold * 0.7
        if off_threshold >= on_threshold:
            raise ValueError("off_threshold must be lower than on_threshold "
                              "(that gap is what prevents flicker)")
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.envelope = EnvelopeFilter(window=window)
        self.min_active_samples = max(1, min_active_samples)
        self.min_rest_samples = max(1, min_rest_samples)

        self.active = False
        self._consec = 0
        self.last_envelope = 0.0

    def update(self, raw_sample: float) -> bool:
        """Feed one raw sample, get back the current (possibly unchanged)
        active/resting state."""
        env = self.envelope.update(raw_sample)
        self.last_envelope = env

        crossed = env >= self.on_threshold if not self.active else env <= self.off_threshold
        self._consec = self._consec + 1 if crossed else 0

        needed = self.min_active_samples if not self.active else self.min_rest_samples
        if crossed and self._consec >= needed:
            self.active = not self.active
            self._consec = 0

        return self.active

    def reset(self) -> None:
        self.envelope.reset()
        self.active = False
        self._consec = 0
        self.last_envelope = 0.0
