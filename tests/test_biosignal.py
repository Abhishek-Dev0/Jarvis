import random

from jarvis.modules.biosignal import EnvelopeFilter, MuscleActivityDetector


def test_envelope_filter_rms_of_constant_signal():
    ef = EnvelopeFilter(window=5)
    for _ in range(10):
        result = ef.update(2.0)
    assert result == 2.0


def test_envelope_filter_reset_clears_state():
    ef = EnvelopeFilter(window=3)
    ef.update(5.0)
    ef.update(5.0)
    ef.reset()
    assert ef.update(1.0) == 1.0


def test_muscle_activity_detector_rejects_bad_thresholds():
    try:
        MuscleActivityDetector(on_threshold=1.0, off_threshold=1.0)
        assert False, "should have rejected off_threshold >= on_threshold"
    except ValueError:
        pass


def test_muscle_activity_detector_stays_inactive_on_low_noise():
    random.seed(0)
    det = MuscleActivityDetector(on_threshold=0.5, window=10)
    for _ in range(200):
        active = det.update(random.uniform(-0.1, 0.1))
    assert active is False


def test_muscle_activity_detector_detects_a_burst_and_returns_to_rest():
    random.seed(0)
    det = MuscleActivityDetector(on_threshold=0.5, window=10,
                                  min_active_samples=3, min_rest_samples=3)

    def noise(amp):
        return random.uniform(-amp, amp)

    states = []
    for i in range(600):
        sample = noise(1.0) if 200 <= i < 400 else noise(0.1)
        states.append(det.update(sample))

    assert not any(states[:200])          # quiet before the burst
    assert all(states[350:400])           # solidly active mid-burst
    assert not any(states[500:600])       # back to rest afterward
