import pytest

from jarvis.modules.hardware_io import HardwareSkill, SerialLink, list_serial_ports


def test_list_serial_ports_does_not_crash_and_returns_a_list():
    # Read-only enumeration -- safe to call for real, no device required.
    ports = list_serial_ports()
    assert isinstance(ports, list)
    for p in ports:
        assert "device" in p and "is_likely_arduino" in p


def test_serial_link_not_connected_by_default():
    link = SerialLink()
    assert link.connected is False


def test_serial_link_read_line_returns_none_when_not_connected():
    link = SerialLink()
    assert link.read_line() is None


def test_serial_link_send_raw_raises_when_not_connected():
    link = SerialLink()
    with pytest.raises(RuntimeError):
        link.send_raw("hello")


def test_serial_link_start_stream_raises_when_not_connected():
    link = SerialLink()
    with pytest.raises(RuntimeError):
        link.start_stream(lambda line: None)


def test_hardware_skill_matches_commands():
    sk = HardwareSkill()
    assert sk.matches("list serial ports") is True
    assert sk.matches("connect to arduino") is True
    assert sk.matches("send to arduino move 90") is True
    assert sk.matches("read arduino") is True
    assert sk.matches("tell me a joke") is False


def test_hardware_skill_connect_denied_without_authorization():
    sk = HardwareSkill(security_ref=None, is_admin_ref=None)
    reply = sk.handle("connect to arduino")
    assert "Denied" in reply


def test_hardware_skill_read_before_connect():
    sk = HardwareSkill()
    reply = sk.handle("read arduino")
    assert "Not connected" in reply


def test_hardware_skill_send_before_connect():
    sk = HardwareSkill(security_ref=None, is_admin_ref=lambda: True)  # even if authorized...
    reply = sk.handle("send to arduino ping")
    assert "Not connected" in reply  # ...there's still nothing to send to
