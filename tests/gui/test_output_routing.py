"""Each run mode must file its artifacts in its own folder.

Bug history (2026-09-17): a combined JV+SPO run wrote its SPO raw CSV and SPO
report into the **Main** folder next to the JV report. `start_combined_run`
built both SPO paths from `file_params['directory']`, which is the file panel's
path and therefore always the "Main" mode folder. A *standalone* SPO run was
unaffected because `SpoProcedure` resolves the SPO folder itself — but in
combined mode the controller passes an explicit `csv_path`, which bypasses
that entirely.
"""
import logging
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


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
    """Isolate the DirectoryManager singleton — its mode leaks between tests."""
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")   # no publishing in tests
    yield tmp_path
    DirectoryManager._instance = None


class _FakeInstruments:
    """Enough of InstrumentManager to get past the pre-flight checks."""
    keithley = object()
    mux = object()

    def connect_keithley(self, **kwargs):
        pass

    def connect_mux(self, **kwargs):
        pass

    def is_keithley_alive(self):
        return True

    def is_mux_alive(self):
        return True


def _teardown(app, window):
    """Close a window without leaving Qt objects half-destroyed.

    `start_combined_run` leaves combined state set, so clear it through the
    controller's own API rather than dropping the controller reference: the
    controller owns the pymeasure Manager, and letting that be garbage
    collected mid-teardown is how this suite gets a fatal abort rather than a
    test failure.
    """
    controller = getattr(window, "controller", None)
    if controller is not None:
        try:
            controller._reset_combined_state()
        except Exception:
            pass
    window.close()
    window.deleteLater()
    app.processEvents()


def _window(app, monkeypatch):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    window = JVAnalyzerWindow("yaman3397")
    window.instrument_manager = _FakeInstruments()
    # Swallow queued experiments instead of running them: this test is about
    # the paths the controller chooses, not about driving a sweep.
    monkeypatch.setattr(window.controller.manager, "queue",
                        lambda experiment: None)
    window.file_panel.filename_input.setText("Test")
    return window


def test_combined_run_files_jv_and_spo_in_their_own_folders(
    app, fresh_dirs, clean_logging, monkeypatch
):
    window = _window(app, monkeypatch)
    try:
        controller = window.controller
        assert window.combined_tab.get_selected_channels(), "no channels toggled"

        controller.start_combined_run()

        def folder_of(path):
            return os.path.basename(os.path.dirname(path))

        assert folder_of(controller.merged_file_path) == "Main"
        assert folder_of(controller._combined_spo_csv_path) == "SPO"
        assert folder_of(controller._combined_spo_report_path) == "SPO"

        # The singleton's mode must be put back, or every later JV run files
        # itself under SPO.
        assert window.dir_manager.mode == "Main"
    finally:
        _teardown(app, window)


def test_combined_panel_reports_the_spo_channel_and_hold_voltage(
    app, fresh_dirs, clean_logging, monkeypatch
):
    """In combined mode the app picks the channel, so the operator must be told.

    The hold voltage is shown SIGNED: Vmpp is signed and is used directly as
    the hold voltage, so a p-i-n cell legitimately holds negative. Displaying
    a magnitude here would look like the wrong-polarity bug the metric
    contract exists to prevent.
    """
    window = _window(app, monkeypatch)
    try:
        controller = window.controller

        controller._show_combined_spo_setpoint(3, -0.8423)
        assert window.combined_spo_channel.text() == "Ch 3"
        assert window.combined_spo_hold.text() == "-842.3 mV"

        controller._show_combined_spo_setpoint(2, 0.912)
        assert window.combined_spo_hold.text() == "+912.0 mV"

        controller._show_combined_spo_setpoint()
        assert window.combined_spo_channel.text() == "—"
        assert window.combined_spo_hold.text() == "— mV"
    finally:
        _teardown(app, window)


def test_calibration_report_goes_to_the_calibration_folder(
    app, fresh_dirs, clean_logging, monkeypatch
):
    from solarjv_analyzer.windows.calibration_window import CalibrationWindow

    monkeypatch.setattr(CalibrationWindow, "_connect_hardware", lambda self: None)
    window = CalibrationWindow("yaman3397")
    try:
        path = window._get_calibration_file_path()
        assert os.path.basename(os.path.dirname(path)) == "Calibration", path
    finally:
        _teardown(app, window)
