# tests/gui/test_shared_bottom_section.py
"""
Regression test: the bottom section (Experiment Queue browser + Channel
Analysis panel) must be a SINGLE shared instance, reparented between the
combined (JV+SPO) and Advanced views.

Bug history: _create_bottom_section() was called twice — once inside
_create_combined_display() and once for the Advanced view. The second call
overwrote self.browser_widget / self.analysis_panel, so the PyMeasure
Manager and every controller update path bound to the Advanced instances,
and combined-mode runs showed no live progress and no analysis results
(the combined view held orphaned, never-updated copies).
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture
def clean_logging():
    """JVAnalyzerWindow attaches Qt-emitter log handlers to the root logger;
    once the window is deleted those handlers raise on every later log call
    (dead C++ Emitter), breaking unrelated tests. Snapshot and restore."""
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
    # RESULTS_ROOT is imported from config lazily inside the module.
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    yield
    DirectoryManager._instance = None


def test_bottom_section_is_single_shared_instance(app, fresh_dirs, clean_logging):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    from pymeasure.display.widgets import BrowserWidget
    from solarjv_analyzer.gui.widgets.analysis_panel import AnalysisPanel

    w = JVAnalyzerWindow("pytest_user")
    try:
        # Exactly one instance of each — a second instance means one view
        # holds orphaned widgets that never receive updates.
        browsers = w.findChildren(BrowserWidget)
        panels = w.findChildren(AnalysisPanel)
        assert len(browsers) == 1, "duplicate BrowserWidget instances"
        assert len(panels) == 1, "duplicate AnalysisPanel instances"

        # The bound attributes ARE the visible instances.
        assert browsers[0] is w.browser_widget
        assert panels[0] is w.analysis_panel

        # The experiment Manager is bound to the same single browser —
        # this is what makes live progress rows appear in whichever view
        # is showing.
        assert w.controller.manager.browser is w.browser_widget.browser

        # The shared section physically moves between views on mode switch.
        w._on_mode_button_click(1)  # Advanced
        assert w.vertical_splitter.indexOf(w.bottom_section) != -1, \
            "bottom section did not move into the Advanced splitter"

        w._on_mode_button_click(0)  # back to combined
        assert w.vertical_splitter.indexOf(w.bottom_section) == -1, \
            "bottom section still parented to the Advanced splitter"
        assert w.bottom_section.parent() is not None
    finally:
        w.close()
        w.deleteLater()
