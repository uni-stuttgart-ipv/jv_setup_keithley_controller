"""The live Log page in the JV+SPO graph area.

The operator watches this to see what the run is doing, so the two things that
must not fail are: records raised on a MEASUREMENT thread have to reach the
view, and a warning nobody is looking at has to be noticeable afterwards.
"""
import logging
import threading

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets                             # noqa: E402

from solarjv_analyzer.gui.widgets.log_panel import LogPanel     # noqa: E402
from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def clean_logging():
    """The window installs handlers on the ROOT logger (audit A7). Left
    behind, a deleted one raises in every later test that logs."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    root.setLevel(logging.INFO)          # the window does this; panels need it
    yield
    from solarjv_analyzer.gui.widgets import log_panel as module
    module.reset_for_tests()             # shared handler + history are global
    for handler in list(root.handlers):
        if handler not in saved:
            root.removeHandler(handler)
    for handler in saved:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture
def fresh_dirs(tmp_path, monkeypatch):
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")
    yield tmp_path
    DirectoryManager._instance = None


@pytest.fixture
def panel(app):
    widget = LogPanel()
    yield widget
    widget.detach()
    widget.deleteLater()
    app.processEvents()


def _destroy(app, widget):
    """Tear a window down COMPLETELY before the next one is built.

    `deleteLater()` only posts a DeferredDelete event. If the next
    JVAnalyzerWindow is constructed while that is still queued, building its
    pyqtgraph PlotWidget can segfault — a long-standing fragility in this
    suite that shows up as a fatal crash rather than a test failure. Flushing
    the deferred deletes and collecting makes the teardown actually happen.
    """
    widget.close()
    widget.deleteLater()
    app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture
def window(app, fresh_dirs):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    win = JVAnalyzerWindow("yaman3397")
    yield win
    _destroy(app, win)


# ---------------------------------------------------------------- the panel
def test_a_record_from_a_worker_thread_reaches_the_view(app, panel):
    """Sweeps log from pymeasure's worker, not the GUI thread. Touching a
    widget from there is a crash; the handler has to hop threads via a
    signal."""
    panel.attach_to_root()
    logger = logging.getLogger("solarjv_analyzer.test.worker")
    logger.setLevel(logging.INFO)

    thread = threading.Thread(
        target=lambda: logger.info("swept channel 3"), name="fake-worker")
    thread.start()
    thread.join(5)
    app.processEvents()

    assert "swept channel 3" in panel.view.toPlainText()


def test_warnings_and_errors_are_coloured_apart_from_the_rest(panel):
    panel.attach_to_root()
    logger = logging.getLogger("solarjv_analyzer.test.levels")
    logger.setLevel(logging.INFO)
    logger.info("routine")
    logger.warning("adapter did not answer")
    logger.error("safety abort failed")

    html = panel.view.document().toHtml()
    assert "#b45309" in html, "no warning colour"
    assert "#b91c1c" in html, "no error colour"
    # The level tag keeps the message column aligned whatever the severity.
    assert "WARN" in panel.view.toPlainText()
    assert "ERR" in panel.view.toPlainText()


def test_the_filter_hides_routine_traffic(panel):
    panel.attach_to_root()
    logger = logging.getLogger("solarjv_analyzer.test.filter")
    logger.setLevel(logging.INFO)
    logger.info("routine")
    logger.warning("adapter did not answer")

    panel.level_filter.setCurrentIndex(1)          # Warnings & errors

    text = panel.view.toPlainText()
    assert "adapter did not answer" in text
    assert "routine" not in text

    panel.level_filter.setCurrentIndex(0)          # back to All
    assert "routine" in panel.view.toPlainText(), "filtering discarded records"


def test_detach_stops_delivery(panel):
    panel.attach_to_root()
    panel.detach()
    logging.getLogger("solarjv_analyzer.test.detached").error("after detach")
    assert "after detach" not in panel.view.toPlainText()


# ------------------------------------------------------- wired to the window
def test_the_window_feeds_the_log_page(window):
    logging.getLogger("solarjv_analyzer.test.window").info("hello from the app")
    assert "hello from the app" in window.log_panel.view.toPlainText()


def test_switching_to_log_swaps_the_graph_area(window):
    window.graph_tab_bar.setCurrentIndex(1)
    assert window.graph_stack.currentWidget() is window.log_panel
    assert not window._combined_save_row.isVisible(), (
        "'Save Both Plots' is still offered while the plots are hidden")

    window.graph_tab_bar.setCurrentIndex(0)
    assert window.graph_stack.currentIndex() == 0


def test_a_warning_while_the_plots_are_showing_marks_the_log_tab(window):
    window.graph_tab_bar.setCurrentIndex(0)
    logging.getLogger("solarjv_analyzer.test.attention").warning("compliance clipped")

    assert window.log_panel.has_attention()
    assert window.graph_tab_bar.tabText(1) != "Log", "nothing points at the log"

    window.graph_tab_bar.setCurrentIndex(1)
    assert not window.log_panel.has_attention()
    assert window.graph_tab_bar.tabText(1) == "Log"


def test_logging_after_the_panel_is_destroyed_does_not_raise(app):
    """A worker thread can log while the window is tearing down. Nothing in
    that path may touch a widget whose C++ object is already gone."""
    panel = LogPanel()
    panel.attach_to_root()
    _destroy(app, panel)

    logging.getLogger("solarjv_analyzer.test.late").error("worker logged late")
    app.processEvents()

    # The handler is deliberately process-wide — it is shared by every view
    # and outlives any one panel; PyQt skips delivery to a receiver whose C++
    # object is gone, so the destroyed panel simply stops being updated.
    from solarjv_analyzer.gui.widgets import log_panel as module
    assert module._root_handler in logging.root.handlers
    assert "worker logged late" in [text for _, text in module.history()][-1]


# ------------------------------------------------ the same design everywhere
def test_every_view_has_the_same_plot_log_switch(window):
    """Advanced used to show pymeasure's own Log tab (different chrome, no
    filter) and Advanced→SPO had no log at all."""
    for bar in (window.graph_tab_bar, window.jv_graph_tab_bar,
                window.spo_graph_tab_bar):
        assert [bar.tabText(i) for i in range(bar.count())] == ["Plot", "Log"]
        assert bar.objectName() == "GraphTabBar"

    assert not hasattr(window, "log_widget"), \
        "pymeasure's LogWidget is still around"


def test_all_three_log_pages_show_the_same_record(window):
    logging.getLogger("solarjv_analyzer.test.shared").info("one stream")

    for panel in (window.log_panel, window.jv_log_panel, window.spo_log_panel):
        assert "one stream" in panel.view.toPlainText()


def test_the_advanced_jv_log_replaces_only_the_plot(window):
    window._on_mode_button_click(1)                  # Advanced
    window.jv_graph_tab_bar.setCurrentIndex(1)       # Log

    assert window.jv_graph_stack.currentWidget() is window.jv_log_panel
    assert not window.save_plot_button.isVisible(), \
        "'Save Plot as PNG' is still offered while the plot is hidden"
    assert window.bottom_section.isVisibleTo(window), \
        "the Experiment Queue disappeared with the plot"


def test_the_advanced_spo_view_has_a_log_page(window):
    window._on_mode_button_click(1)
    window.spo_mode_button.setChecked(True)
    window._show_spo_mode()
    window.spo_graph_tab_bar.setCurrentIndex(1)

    assert window.spo_graph_stack.currentWidget() is window.spo_log_panel


def test_one_handler_serves_every_panel(window):
    """A handler per panel would mean three formatting passes per record and
    three handlers left on the root logger each time the relogin loop builds a
    new window."""
    from solarjv_analyzer.gui.widgets import log_panel as module

    qt_handlers = [h for h in logging.root.handlers
                   if isinstance(h, module.QtLogHandler)]
    assert len(qt_handlers) == 1, f"{len(qt_handlers)} Qt log handlers on root"


def test_a_panel_built_later_still_shows_what_happened_before(app):
    from solarjv_analyzer.gui.widgets import log_panel as module

    module.install_root_handler()
    logging.getLogger("solarjv_analyzer.test.before").info("happened earlier")

    panel = LogPanel()
    panel.attach_to_root()
    try:
        assert "happened earlier" in panel.view.toPlainText()
    finally:
        panel.detach()
        _destroy(app, panel)
