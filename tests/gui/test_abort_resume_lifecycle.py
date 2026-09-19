"""The Abort/Resume button's lifecycle, in the main window.

`pymeasure.display.manager.BaseManager.abort()` sets `_start_on_add = False`
and `_is_continuous = False`. The ONLY thing that restores them is
`Manager.resume()`, and the controller calls `resume()` from exactly one place:
the "Resume" button. So any abort the operator does not follow with a Resume
click leaves the manager permanently unable to auto-start a queued experiment —
`Manager.queue()` appends the row and returns without calling `next()`.

Three consequences, one per test below.
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture(autouse=True)
def clean_logging():
    """The window installs pymeasure's LogHandler on the ROOT logger and does
    not remove it (audit A7). Left behind, it raises "wrapped C/C++ object of
    type Emitter has been deleted" in every later test that logs."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved:
            root.removeHandler(handler)
    for handler in saved:
        if handler not in root.handlers:
            root.addHandler(handler)
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


class _FakeWorker:
    def stop(self):
        pass


@pytest.fixture
def window(app, fresh_dirs):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    win = JVAnalyzerWindow("yaman3397")
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def _abort_the_last_sweep(controller):
    """Drive the real lifecycle: running -> user clicks Abort -> abort returns
    with nothing else queued."""
    manager = controller.manager
    manager._worker = _FakeWorker()
    manager._running_experiment = object()     # pymeasure's is_running()
    controller.abort_experiment()              # the user clicks Abort
    manager._worker = None
    manager._running_experiment = None         # pymeasure's _clean_up()
    controller.on_abort_returned()


def test_the_next_run_queues_rows_that_never_start(window):
    controller = window.controller
    _abort_the_last_sweep(controller)

    started = []
    manager = controller.manager
    manager.load = lambda experiment: None     # not what this test is about
    manager.next = lambda: started.append("next")
    manager.queue(object())                    # what queue_experiment() does

    assert started == ["next"], (
        "after an abort, queued experiments are never started: the browser "
        "fills with QUEUED rows and no sweep runs")


def test_the_button_is_rearmed_to_abort_when_the_run_ends(window):
    """The handler is swapped to resume_experiment on abort. When the abort
    returns with an empty queue the label goes back to "Abort" — but the
    handler does not, so the label and the wiring disagree."""
    controller = window.controller
    fired = []
    # Stand-ins, installed before the lifecycle runs so that whatever
    # abort_experiment() connects is one of these and the click is traceable.
    controller.abort_experiment = lambda: fired.append("abort")
    controller.resume_experiment = lambda: fired.append("resume")

    controller.manager._worker = _FakeWorker()
    controller.manager._running_experiment = object()
    type(controller).abort_experiment(controller)      # the user clicks Abort
    controller.manager._worker = None
    controller.manager._running_experiment = None
    controller.on_abort_returned()

    assert window.abort_button.text() == "Abort"
    window.abort_button.setEnabled(True)
    window.abort_button.clicked.emit()

    assert fired == ["abort"], f"the button labelled Abort ran {fired}"


def test_aborting_when_nothing_runs_does_not_kill_the_app(window):
    """`Manager.abort()` raises if nothing is running, and an unhandled
    exception in a PyQt5 slot calls qFatal() — the process dies, no dialog.
    `abort_experiment` must not let that escape."""
    controller = window.controller
    assert not controller.manager.is_running()

    controller.abort_experiment()               # must not raise

    assert window.queue_button.isEnabled(), (
        "the run button is left disabled with nothing running — the window "
        "is stuck with every control greyed out")
