# tests/gui/test_calibration_close_event.py
"""
Regression test: closing the CalibrationWindow with the title-bar X (or
Cmd-W) must de-energize the Keithley and disconnect it, exactly like
JVAnalyzerWindow.closeEvent does.

Bug history: CalibrationWindow had no closeEvent override, so X during a
running sweep fell through to the default QMainWindow.closeEvent, which
closed the window and let main.py hit sys.exit(0) with the SourceMeter
still sourcing (no :OUTP OFF / :ABOR / disconnect ever sent).
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtGui, QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


class _FakeKeithley:
    """Records SCPI writes so the test can assert de-energization."""

    def __init__(self):
        self.writes = []

    def write(self, cmd):
        self.writes.append(cmd)


class _FakeManager:
    """Minimal PyMeasure Manager stand-in that reports a running sweep."""

    def __init__(self, running=True):
        self._running = running
        self.aborted = False

    def is_running(self):
        return self._running

    def abort(self):
        self.aborted = True
        self._running = False


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


def test_close_event_deenergizes_keithley_and_disconnects(
    app, fresh_dirs, clean_logging, monkeypatch
):
    from solarjv_analyzer.windows.calibration_window import CalibrationWindow

    # CalibrationWindow.__init__ calls _connect_hardware(), which would open a
    # real VISA connection. Stub it out — the test injects its own instrument.
    monkeypatch.setattr(CalibrationWindow, "_connect_hardware", lambda self: None)

    w = CalibrationWindow("pytest_user")
    try:
        k = _FakeKeithley()
        w.instrument_manager.keithley = k
        w.manager = _FakeManager(running=True)

        w.closeEvent(QtGui.QCloseEvent())

        assert w.manager.aborted is True, "running sweep was not aborted"
        assert ":OUTP OFF" in k.writes, "output was not turned off"
        assert ":ABOR" in k.writes, "sweep was not aborted on the instrument"
        assert w.instrument_manager.keithley is None, "instrument not disconnected"
    finally:
        # Tear down fully: close() (triggers closeEvent again — idempotent),
        # then flush the deferred delete so pymeasure's shared, parentless
        # LogWidget._blink_qtimer isn't left half-destroyed for the next test.
        w.close()
        w.deleteLater()
        app.processEvents()
