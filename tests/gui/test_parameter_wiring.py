"""Parameters that reach the measurement, and ones that only reach the report.

Each test here exists because the control looked wired and was not (or looked
dead and was not). They pin the distinction rather than the implementation.
"""
import logging
import os
import tempfile

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets                                    # noqa: E402

from solarjv_analyzer.instruments import port_resolver as pr    # noqa: E402
from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def clean_logging():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    root.setLevel(logging.INFO)
    yield
    from solarjv_analyzer.gui.widgets import log_panel as module
    module.reset_for_tests()
    for handler in list(root.handlers):
        if handler not in saved:
            root.removeHandler(handler)
    for handler in saved:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture(autouse=True)
def isolate_resolver(tmp_path, monkeypatch):
    pr.reset_for_tests()
    monkeypatch.setattr(pr, "settings_path",
                        lambda: str(tmp_path / ".solarjv" / "config.json"))
    yield
    pr.reset_for_tests()


@pytest.fixture
def fresh_dirs(tmp_path, monkeypatch):
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")
    yield tmp_path
    DirectoryManager._instance = None


# ------------------------------------------------------- GPIB address
def test_a_typed_address_overrides_what_resolution_found():
    pr.apply(pr.resolve(ports=[], probe=False, settings={},
                        config_mux="COM4", config_keithley="ASRL3::INSTR"),
             remember=False)
    assert pr.active_keithley_resource() == "ASRL3::INSTR"

    pr.set_session_keithley_resource("GPIB0::24::INSTR")

    assert pr.active_keithley_resource() == "GPIB0::24::INSTR"
    assert pr.session_keithley_override() == "GPIB0::24::INSTR"


def test_clearing_the_override_restores_detection():
    pr.apply(pr.resolve(ports=[], probe=False, settings={},
                        config_mux="COM4", config_keithley="ASRL3::INSTR"),
             remember=False)
    pr.set_session_keithley_resource("GPIB0::24::INSTR")

    pr.set_session_keithley_resource("")

    assert pr.active_keithley_resource() == "ASRL3::INSTR"


def test_the_override_is_never_written_to_disk():
    """A backup address typed to get through one afternoon must not become
    the permanent answer on every future launch."""
    pr.set_session_keithley_resource("GPIB0::24::INSTR")
    pr.apply(pr.resolve(ports=[], probe=False, settings={},
                        config_mux="COM4", config_keithley="ASRL3::INSTR"),
             remember=True)

    saved = pr.load_hardware_settings()
    assert saved.get("keithley_resource") != "GPIB0::24::INSTR"
    assert not os.path.exists(pr.settings_path()) or \
        "GPIB0::24::INSTR" not in open(pr.settings_path()).read()


def test_a_refresh_does_not_undo_the_override(monkeypatch):
    monkeypatch.setattr(pr, "_enumerate", lambda: [])
    pr.set_session_keithley_resource("GPIB0::24::INSTR")
    pr.refresh_if_unconfirmed()
    assert pr.active_keithley_resource() == "GPIB0::24::INSTR"


@pytest.mark.parametrize("text, ok", [
    ("ASRL3::INSTR", True), ("GPIB0::24::INSTR", True),
    ("TCPIP0::192.168.0.5::INSTR", True), ("COM3", False),
    ("ASRL3", False), ("nonsense", False), ("", False),
])
def test_obvious_typos_are_rejected(text, ok):
    from solarjv_analyzer.gui.jv_analyzer_window import _looks_like_visa_resource
    assert _looks_like_visa_resource(text) is ok


def test_the_address_cannot_be_changed_mid_run(app, fresh_dirs, monkeypatch):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    win = JVAnalyzerWindow("yaman3397")
    try:
        before = pr.active_keithley_resource()
        win.controller.is_busy = True
        win._on_address_edited("GPIB0::24::INSTR")
        assert pr.active_keithley_resource() == before, \
            "the address was swapped while a measurement was running"
    finally:
        win.controller.is_busy = False
        win.close()
        win.deleteLater()
        app.processEvents()


# ------------------------------------------------------- line frequency
def test_the_line_frequency_is_the_same_on_every_path():
    """A `:SYST:LFR?` query used to run only on the fallback-connect path, so
    the NPLC timing model differed depending on how the instrument arrived."""
    import inspect

    from solarjv_analyzer.procedures import jv_procedure

    source = inspect.getsource(jv_procedure.JVProcedure.startup)
    assert "LFR" not in source, \
        "startup() still changes the line frequency on one path only"
    assert jv_procedure.JVProcedure.line_frequency.default == 50.0


# ------------------------------------------------------- probe spacing
def test_probe_spacing_reaches_the_report_but_no_metric():
    from pymeasure.experiment import Results

    from solarjv_analyzer.gui.app_controller import AppController
    from solarjv_analyzer.procedures.jv_procedure import JVProcedure

    proc = JVProcedure(probe_spacing=2290.0, sample_thickness=500.0)
    path = os.path.join(tempfile.mkdtemp(), "t.csv")
    Results(proc, path)
    with open(path, "a") as handle:
        handle.write("1.2,0.001\n1.19,0.0011\n")

    _, _, parameters = AppController._parse_temp_file(None, path)
    keys = [name for name, _ in parameters]
    assert "4-Probe Spacing" in keys, "spacing is missing from the report"

    # ...and it is deliberately absent from the metric math.
    import inspect

    from solarjv_analyzer.analysis import analysis
    assert "probe_spacing" not in inspect.signature(
        analysis.compute_jv_metrics).parameters


# ------------------------------------------------------- debug logging
def test_debug_records_are_visible_when_debug_is_switched_on(app, fresh_dirs):
    """The Analysis tab's debug switch raises the root logger to DEBUG. The
    Log page used to filter at INFO, so the extra trace was hidden."""
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    win = JVAnalyzerWindow("yaman3397")
    try:
        win.analysis_settings_tab.enable_validation.setChecked(True)   # debug on
        logging.getLogger("solarjv_analyzer.test.trace").debug("sweep point 42")

        assert "sweep point 42" in win.log_panel.view.toPlainText()
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_the_debug_switch_moves_the_root_logger(app, fresh_dirs):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    win = JVAnalyzerWindow("yaman3397")
    try:
        win.analysis_settings_tab.enable_validation.setChecked(True)
        assert logging.root.level == logging.DEBUG
        win.analysis_settings_tab.enable_validation.setChecked(False)
        assert logging.root.level == logging.INFO
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()
