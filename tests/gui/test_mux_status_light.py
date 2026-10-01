"""The status lights must report the hardware, before anything is run.

Bug history (2026-09-17): the MUX light stayed red with the multiplexer plugged
in and working, and only turned green after the operator pressed a run button.
Nothing was wrong with the MUX — `connect_mux()` was only ever called at run
start, and the calibration gate is Keithley-only by design, so the port
genuinely was not open. The lights reported the application's laziness rather
than the instruments, which is exactly backwards: by the time a run has begun,
knowing the hardware is connected is too late to be useful.

Both instruments arrive as USB-to-serial adapters (MUX: CH340, Keithley:
Prolific PL2303 into its RS-232 port), so "connected" has to mean *the adapter
is still enumerated AND we hold an open handle*. `pyserial`'s `is_open` and a
PyVISA session handle both stay valid after the USB device is pulled.
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.instruments import port_status  # noqa: E402
from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402

GREEN = "#10b981"
RED = "#ef4444"


@pytest.fixture
def clean_logging():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in saved:
            root.removeHandler(h)
    for h in saved:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fresh_dirs(tmp_path, monkeypatch):
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")
    yield tmp_path
    DirectoryManager._instance = None


class _FakeSerial:
    is_open = True


class _FakeMux:
    def __init__(self):
        self.ser = _FakeSerial()

    def close(self):
        self.ser.is_open = False


def _window(app):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    return JVAnalyzerWindow("yaman3397")


def _close(app, window):
    window.close()
    window.deleteLater()
    app.processEvents()


def _run_check(app, window):
    window.check_hardware_status()
    thread = getattr(window, "_hardware_thread", None)
    if thread is not None:
        thread.wait(5000)
    app.processEvents()


# ---------------------------------------------------------------------------
# visa_to_port / presence — the pure bits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("address, expected", [
    ("ASRL3::INSTR", "COM3"),
    ("ASRL12::INSTR", "COM12"),
    ("ASRLCOM3::INSTR", "COM3"),
    ("GPIB0::24::INSTR", ""),
    ("", ""),
])
def test_visa_address_maps_to_its_com_port(address, expected):
    assert port_status.visa_to_port(address) == expected


def test_a_non_com_transport_is_never_reported_as_unplugged():
    """A GPIB or USB-TMC Keithley has no COM port to look for."""
    assert port_status.is_port_present("", set()) is True


def test_presence_is_by_enumeration():
    assert port_status.is_port_present("COM3", {"COM3", "COM4"}) is True
    assert port_status.is_port_present("COM9", {"COM3", "COM4"}) is False


# ---------------------------------------------------------------------------
# the lights
# ---------------------------------------------------------------------------

def test_mux_light_is_green_before_any_run(app, fresh_dirs, clean_logging, monkeypatch):
    """The whole point: connected hardware shows green without pressing Run."""
    from solarjv_analyzer.config import GPIB_ADDRESS, MUX_PORT

    window = _window(app)
    try:
        assert RED in window.mux_light.styleSheet(), "precondition: starts red"

        keithley_port = port_status.visa_to_port(GPIB_ADDRESS) or "COM3"
        monkeypatch.setattr(port_status, "port_names",
                            lambda: {MUX_PORT.upper(), keithley_port.upper()})
        monkeypatch.setattr(port_status, "describe", lambda name: "USB-SERIAL CH340")

        def fake_connect(simulation=False):
            window.instrument_manager.mux = _FakeMux()

        window.instrument_manager.connect_mux = fake_connect

        _run_check(app, window)

        assert GREEN in window.mux_light.styleSheet()
    finally:
        _close(app, window)


def test_light_goes_red_when_the_adapter_is_unplugged(
    app, fresh_dirs, clean_logging, monkeypatch
):
    """`is_open` stays True after the USB device is pulled — enumeration does not."""
    window = _window(app)
    try:
        window.instrument_manager.mux = _FakeMux()
        assert window.instrument_manager.is_mux_alive(), "handle still looks open"

        monkeypatch.setattr(port_status, "port_names", lambda: set())   # unplugged
        _run_check(app, window)

        assert RED in window.mux_light.styleSheet()
    finally:
        _close(app, window)


def test_the_monitor_never_touches_instruments_during_a_run(
    app, fresh_dirs, clean_logging, monkeypatch
):
    """Mid-sweep the check may look, never open, close or probe."""
    from solarjv_analyzer.config import MUX_PORT

    window = _window(app)
    try:
        monkeypatch.setattr(port_status, "port_names", lambda: {MUX_PORT.upper()})
        attempts = []
        window.instrument_manager.connect_mux = lambda **k: attempts.append(k)
        window.controller.is_busy = True

        _run_check(app, window)

        assert attempts == [], "opened a port while a measurement was running"
    finally:
        window.controller.is_busy = False
        _close(app, window)


def test_a_missing_adapter_is_not_hammered(app, fresh_dirs, clean_logging, monkeypatch):
    """No enumeration, no open attempt — cheaper and quieter than a failed open."""
    window = _window(app)
    try:
        monkeypatch.setattr(port_status, "port_names", lambda: set())
        attempts = []
        window.instrument_manager.connect_mux = lambda **k: attempts.append(k)

        _run_check(app, window)

        assert attempts == []
        assert RED in window.mux_light.styleSheet()
    finally:
        _close(app, window)


def test_a_failed_open_leaves_the_window_usable(app, fresh_dirs, clean_logging, monkeypatch):
    import serial
    from solarjv_analyzer.config import MUX_PORT

    window = _window(app)
    try:
        monkeypatch.setattr(port_status, "port_names", lambda: {MUX_PORT.upper()})

        def boom(simulation=False):
            raise serial.SerialException(f"could not open port {MUX_PORT!r}")

        window.instrument_manager.connect_mux = boom

        _run_check(app, window)          # must not raise

        assert RED in window.mux_light.styleSheet()
        assert window.isEnabled()
    finally:
        _close(app, window)


def test_the_light_says_only_whether_it_is_connected(
    app, fresh_dirs, clean_logging, monkeypatch
):
    """The indicator is deliberately dumb.

    Ports, adapter names and failure reasons belong in the session log, not on
    an operator's screen — all they need to know before starting an experiment
    is whether both instruments are ready.
    """
    from solarjv_analyzer.config import MUX_PORT

    window = _window(app)
    try:
        monkeypatch.setattr(port_status, "port_names", lambda: {MUX_PORT.upper()})
        monkeypatch.setattr(port_status, "describe", lambda name: "USB-SERIAL CH340")
        window.instrument_manager.connect_mux = lambda **k: setattr(
            window.instrument_manager, "mux", _FakeMux())

        _run_check(app, window)

        assert window.mux_light.toolTip() == "MUX: Connected"
        assert window.keithley_light.toolTip().startswith("Keithley: ")

        tips = window.mux_light.toolTip() + window.keithley_light.toolTip()
        for leak in (MUX_PORT, "CH340", "COM", "cable", "Click"):
            assert leak not in tips, f"{leak!r} leaked into the status tooltip"
    finally:
        _close(app, window)


def test_the_lights_are_indicators_not_buttons(app, fresh_dirs, clean_logging):
    """Nothing should invite the operator to click a status dot."""
    from PyQt5 import QtCore

    window = _window(app)
    try:
        for light in (window.keithley_light, window.mux_light):
            assert light.cursor().shape() != QtCore.Qt.PointingHandCursor
    finally:
        _close(app, window)


def test_a_change_of_state_is_recorded_in_the_log(
    app, fresh_dirs, clean_logging, monkeypatch, caplog
):
    """Diagnosis happens in the log, and only when something actually changes."""
    from solarjv_analyzer.config import MUX_PORT

    window = _window(app)
    try:
        monkeypatch.setattr(port_status, "port_names", lambda: {MUX_PORT.upper()})
        monkeypatch.setattr(port_status, "describe", lambda name: "USB-SERIAL CH340")
        window.instrument_manager.connect_mux = lambda **k: setattr(
            window.instrument_manager, "mux", _FakeMux())

        # JVAnalyzerWindow._setup_logging() blanket-removes the root logger's
        # handlers (audit finding A7), which takes pytest's capture handler
        # with it. Put it back now that the window has been built, or nothing
        # this window logs can be asserted on.
        logging.getLogger().addHandler(caplog.handler)

        with caplog.at_level(logging.INFO):
            _run_check(app, window)
            first = [r for r in caplog.records if "Hardware status" in r.getMessage()]
            assert first, "a state change was not logged"
            assert "CH340" in first[0].getMessage()

            caplog.clear()
            _run_check(app, window)          # nothing changed
            assert not [r for r in caplog.records
                        if "Hardware status" in r.getMessage()], \
                "an unchanged state was logged again — this would flood the log"
    finally:
        _close(app, window)


def test_monitor_starts_and_stops_cleanly(app, fresh_dirs, clean_logging, monkeypatch):
    monkeypatch.setattr(port_status, "port_names", lambda: set())
    window = _window(app)
    try:
        window.start_hardware_monitor()
        assert window._hardware_timer.isActive()
        window.start_hardware_monitor()          # idempotent
        assert window._hardware_timer.isActive()
    finally:
        _close(app, window)
        assert not window._hardware_timer.isActive(), "timer left running after close"
