# tests/gui/test_calibration_handoff.py
"""
Regression test: the calibration → main-window HAND-OFF must not tear down
the hardware.

Bug history (2026-09): `_on_proceed`/`_on_skip` emit `calibration_passed` and
then close the window. The slot (main.py's `launch_main_app`) runs
synchronously, so `CalibrationWindow.closeEvent` fired *during* the hand-off
and ran its full safety shutdown — `:OUTP OFF`, `:ABOR`, then
`disconnect_keithley()`, which calls `shutdown()` **and `close()`** on the
VISA session the main window had just been given. The main window kept a
non-None handle to that closed instrument, so:

  * `connect_keithley()` short-circuited on `if self.keithley: return`,
  * `update_instrument_lights()` showed a green "connected" dot, and
  * the first sweep died on its first `:OUTP OFF` with
    `pyvisa.errors.InvalidSession: Invalid session handle`.

The same closeEvent also called `SessionManager.end_session()`, which removed
the root-logger file handler (so the failure was never written to the session
log) and cleared `current_user` (so closing the main window bounced back to
the login dialog instead of exiting).

The ✕ / Cmd-W path must still shut down — see test_calibration_close_event.py.
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.auth.session import SessionManager  # noqa: E402
from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


class _FakeConnection:
    """Stand-in for a pyvisa resource: `session` raises once closed."""

    def __init__(self):
        self._open = True

    @property
    def session(self):
        if not self._open:
            raise RuntimeError("Invalid session handle. The resource might be closed.")
        return 1

    def close(self):
        self._open = False


class _FakeAdapter:
    def __init__(self):
        self.connection = _FakeConnection()

    def close(self):
        self.connection.close()


class _FakeKeithley:
    """Records SCPI writes and whether the session was torn down."""

    def __init__(self):
        self.writes = []
        self.adapter = _FakeAdapter()
        self.shutdown_calls = 0

    def write(self, cmd):
        # Mirror pyvisa: writing to a closed session raises.
        _ = self.adapter.connection.session
        self.writes.append(cmd)

    def shutdown(self):
        self.shutdown_calls += 1

    def close(self):
        self.adapter.close()


class _FakeSerial:
    is_open = True

    def close(self):
        type(self).is_open = False


class _FakeMux:
    def __init__(self):
        self.ser = _FakeSerial()
        self.closed = False

    def close(self):
        self.closed = True
        self.ser.close()


@pytest.fixture
def clean_logging():
    """Windows attach Qt-emitter log handlers to the root logger; snapshot and
    restore so later log calls don't hit a dead C++ Emitter."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(saved_level)


@pytest.fixture
def app():
    application = QtWidgets.QApplication.instance()
    if application is None:
        application = QtWidgets.QApplication([])
    return application


@pytest.fixture
def fresh_dirs(tmp_path, monkeypatch):
    """Isolate the DirectoryManager singleton and output tree."""
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    yield
    DirectoryManager._instance = None


@pytest.fixture
def fake_session():
    """Give SessionManager a live-looking session and restore it afterwards."""
    saved_user = SessionManager.current_user
    saved_handler = SessionManager._file_handler
    SessionManager.current_user = "pytest_user"
    SessionManager._file_handler = logging.NullHandler()
    logging.root.addHandler(SessionManager._file_handler)
    yield SessionManager
    if SessionManager._file_handler in logging.root.handlers:
        logging.root.removeHandler(SessionManager._file_handler)
    SessionManager.current_user = saved_user
    SessionManager._file_handler = saved_handler


def _build_window(monkeypatch, manager):
    from solarjv_analyzer.windows.calibration_window import CalibrationWindow

    # __init__ calls _connect_hardware(), which would open a real VISA
    # session. Stub it out — the test injects its own instruments.
    monkeypatch.setattr(CalibrationWindow, "_connect_hardware", lambda self: None)
    return CalibrationWindow("pytest_user", instrument_manager=manager)


@pytest.mark.parametrize("path", ["proceed", "skip"])
def test_handoff_leaves_the_visa_session_open(
    app, fresh_dirs, clean_logging, fake_session, monkeypatch, path
):
    """Proceed / Skip must hand a USABLE instrument to the main window."""
    from solarjv_analyzer.instruments.instrument_manager import (
        InstrumentManager, visa_session_open,
    )
    from solarjv_analyzer.windows.calibration_window import CalibrationWindow

    manager = InstrumentManager()
    k, m = _FakeKeithley(), _FakeMux()
    manager.keithley = k
    manager.mux = m

    w = _build_window(monkeypatch, manager)
    try:
        # Stand in for main.py: capture the payload, then close the window
        # from inside the slot exactly as launch_main_app does.
        received = {}

        def launch_main_app(data):
            received.update(data)
            w.close()

        w.calibration_passed.connect(launch_main_app)

        if path == "proceed":
            w._on_proceed()
        else:
            monkeypatch.setattr(
                QtWidgets.QMessageBox, "warning",
                staticmethod(lambda *a, **kw: QtWidgets.QMessageBox.Yes),
            )
            w._on_skip()

        assert received, "calibration_passed was never delivered"
        handed_over = received["instrument_manager"]
        assert handed_over is manager, "a copy was handed over, not the manager"

        # The whole point: the session survives the hand-off.
        assert visa_session_open(k), (
            "hand-off closed the VISA session — the main window would fail "
            "with InvalidSession on its first sweep"
        )
        assert handed_over.is_keithley_alive()
        assert handed_over.keithley is k, "instrument was released on hand-off"
        assert k.shutdown_calls == 0, "instrument was shut down during hand-off"
        assert not m.closed, "MUX serial port was closed during hand-off"

        # ...and the sweep-blocking SCPI is not sent either.
        assert ":OUTP OFF" not in k.writes
        assert ":ABOR" not in k.writes

        # The logging session must survive: the main window's run has to be
        # recorded, and main.py treats a cleared user as "log in again".
        assert fake_session.current_user == "pytest_user"
        assert fake_session._file_handler is not None
        assert fake_session._file_handler in logging.root.handlers
    finally:
        w.deleteLater()
        app.processEvents()


def test_second_close_after_handoff_is_still_a_noop(
    app, fresh_dirs, clean_logging, fake_session, monkeypatch
):
    """`_on_proceed` closes the window again after the slot already did.

    Qt delivers closeEvent on every close() call, so the hand-off guard has to
    hold for the repeat too — otherwise the second pass tears down the session
    the first pass carefully preserved.
    """
    from solarjv_analyzer.instruments.instrument_manager import (
        InstrumentManager, visa_session_open,
    )

    manager = InstrumentManager()
    k = _FakeKeithley()
    manager.keithley = k

    w = _build_window(monkeypatch, manager)
    try:
        w.calibration_passed.connect(lambda data: w.close())
        w._on_proceed()      # slot closes once, _on_proceed closes again
        w.close()            # and a third time for good measure

        assert visa_session_open(k)
        assert manager.keithley is k
        assert k.shutdown_calls == 0
        assert fake_session.current_user == "pytest_user"
    finally:
        w.deleteLater()
        app.processEvents()


def test_main_window_inherits_a_usable_session_end_to_end(
    app, fresh_dirs, clean_logging, fake_session, monkeypatch
):
    """The full hand-off, wired the way main.py wires it.

    This is the reproduction of the reported failure: after Proceed, the main
    window must be able to start a sweep. Before the fix its
    `instrument_manager.keithley` was a closed handle, `connect_keithley()`
    short-circuited on it, the status light was green, and the sweep raised
    `pyvisa.errors.InvalidSession` on its first `:OUTP OFF`.
    """
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    from solarjv_analyzer.instruments.instrument_manager import InstrumentManager

    manager = InstrumentManager()
    k = _FakeKeithley()
    manager.keithley = k
    manager.mux = _FakeMux()

    calib = _build_window(monkeypatch, manager)
    main_window = None
    try:
        def launch_main_app(data):
            # main.py: hand over the MANAGER, then close the calibration window.
            nonlocal main_window
            main_window = JVAnalyzerWindow(
                "pytest_user", instrument_manager=data["instrument_manager"]
            )
            main_window.update_instrument_lights()
            calib.close()

        calib.calibration_passed.connect(launch_main_app)
        calib._on_proceed()

        assert main_window is not None
        assert main_window.instrument_manager is manager
        assert main_window.instrument_manager.is_keithley_alive(), (
            "the main window inherited a closed VISA session"
        )
        # The controller's pre-flight gate must let the run start...
        assert main_window.controller._keithley_usable("test") is True
        # ...and the instrument must actually accept SCPI.
        main_window.instrument_manager.keithley.write("*CLS")
        assert "*CLS" in k.writes
    finally:
        if main_window is not None:
            main_window.controller = None
            main_window.close()
            main_window.deleteLater()
        calib.deleteLater()
        app.processEvents()
