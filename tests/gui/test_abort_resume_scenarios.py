"""The Abort / Resume / Queue lifecycle in the main window, scenario by scenario.

Each test is named for the behaviour the operator expects. A failure here is a
statement about the app, not about the test.
"""
import os
import types

import pandas as pd
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets                                    # noqa: E402
from pymeasure.experiment import Procedure                     # noqa: E402

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


@pytest.fixture
def window(app, fresh_dirs):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    win = JVAnalyzerWindow("yaman3397")
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


class _FakeWorker:
    def stop(self):
        pass


def _exp(name, status=Procedure.QUEUED):
    return types.SimpleNamespace(
        name=name, procedure=types.SimpleNamespace(status=status))


def _running(manager, experiment=None):
    manager._worker = _FakeWorker()
    manager._running_experiment = experiment or object()


def _worker_returned(manager):
    """What pymeasure's _clean_up() does before emitting abort_returned."""
    manager._worker = None
    manager._running_experiment = None


def _watch_next(manager):
    """Record which experiment next() would start, without starting it."""
    picked = []
    manager.next = lambda: picked.append(manager.experiments.next().name)
    return picked


# ---------------------------------------------------------------------------
# 1. Queue several, abort the first, press Resume -> the SECOND one runs.
# ---------------------------------------------------------------------------
def test_resume_continues_with_the_next_queued_sweep(window):
    controller = window.controller
    manager = controller.manager
    one, two, three = _exp("1"), _exp("2"), _exp("3")
    manager.experiments.queue = [one, two, three]

    _running(manager, one)
    controller.abort_experiment()
    one.procedure.status = Procedure.ABORTED      # the worker sets this
    _worker_returned(manager)
    controller.on_abort_returned()

    assert window.abort_button.text() == "Resume"
    assert window.abort_button.isEnabled()

    picked = _watch_next(manager)
    controller.resume_experiment()

    assert picked == ["2"], f"Resume started {picked}, expected the 2nd sweep"


# ---------------------------------------------------------------------------
# 2. A single queued sweep, aborted -> the sweep is discarded and the window
#    returns to idle so the operator can queue a fresh one. An aborted sweep is
#    NEVER re-run: that is what happens with a longer queue too.
# ---------------------------------------------------------------------------
def test_aborting_the_only_sweep_returns_the_window_to_idle(window):
    controller = window.controller
    manager = controller.manager
    only = _exp("1")
    manager.experiments.queue = [only]
    controller.is_busy = True

    _running(manager, only)
    controller.abort_experiment()
    only.procedure.status = Procedure.ABORTED
    _worker_returned(manager)
    controller.on_abort_returned()

    assert window.queue_button.isEnabled()
    assert not window.abort_button.isEnabled()
    assert window.abort_button.text() == "Abort"
    assert controller.is_busy is False
    assert manager._start_on_add is True, (
        "idle in the UI, but the manager will not start anything ever again")
    assert manager._is_continuous is True


# 3. Queue a new run after an abort, or after a resume -> it must run.
# ---------------------------------------------------------------------------
def test_a_run_queued_after_an_abort_starts(window):
    controller = window.controller
    manager = controller.manager
    one = _exp("1")
    manager.experiments.queue = [one]

    _running(manager, one)
    controller.abort_experiment()
    one.procedure.status = Procedure.ABORTED
    _worker_returned(manager)
    controller.on_abort_returned()

    fresh = _exp("2")
    manager.load = lambda experiment: manager.experiments.queue.append(fresh)
    picked = _watch_next(manager)
    manager.queue(fresh)                        # what queue_experiment() does

    assert picked == ["2"], (
        "a run queued after an abort never starts: rows sit in the browser "
        "as QUEUED and no sweep executes")


def test_a_run_queued_after_a_resume_starts(window):
    controller = window.controller
    manager = controller.manager
    one, two = _exp("1"), _exp("2")
    manager.experiments.queue = [one, two]

    _running(manager, one)
    controller.abort_experiment()
    one.procedure.status = Procedure.ABORTED
    _worker_returned(manager)
    controller.on_abort_returned()

    manager.next = lambda: None                 # Resume starts sweep 2
    controller.resume_experiment()
    two.procedure.status = Procedure.FINISHED
    _worker_returned(manager)

    fresh = _exp("3")
    manager.load = lambda experiment: manager.experiments.queue.append(fresh)
    picked = _watch_next(manager)
    manager.queue(fresh)

    assert picked == ["3"], "a run queued after a Resume never starts"


# ---------------------------------------------------------------------------
# 4. Aborting the last sweep must not throw away the sweeps that finished.
# ---------------------------------------------------------------------------
def test_aborting_the_last_sweep_still_writes_the_completed_sweeps(window):
    """Two channels done, abort on the third: `_merge_channel_files` is only
    reachable from `on_finished`, so the finished sweeps are left as `_temp`
    files that nothing ever merges, publishes, or cleans up."""
    controller = window.controller
    manager = controller.manager
    merged = []
    controller.is_single_file_mode = True
    controller._merge_channel_files = lambda: merged.append("merged")

    last = _exp("3")
    manager.experiments.queue = [
        _exp("1", Procedure.FINISHED), _exp("2", Procedure.FINISHED), last]
    _running(manager, last)
    controller.abort_experiment()
    last.procedure.status = Procedure.ABORTED
    _worker_returned(manager)
    controller.on_abort_returned()

    assert merged == ["merged"], (
        "the two completed sweeps were never merged into an output file")


# ---------------------------------------------------------------------------
# 5. Aborting a combined JV+SPO run must not disarm later JV runs.
# ---------------------------------------------------------------------------
def test_aborting_a_combined_run_leaves_the_jv_queue_usable(window):
    controller = window.controller
    manager = controller.manager
    one = _exp("1")
    manager.experiments.queue = [one]
    controller._combined_mode = True
    controller.is_busy = True

    _running(manager, one)
    controller.abort_combined()
    one.procedure.status = Procedure.ABORTED
    _worker_returned(manager)
    controller.on_abort_returned()

    fresh = _exp("2")
    manager.load = lambda experiment: manager.experiments.queue.append(fresh)
    picked = _watch_next(manager)
    manager.queue(fresh)

    assert picked == ["2"], (
        "after a combined-run abort, every later JV run is dead too")


# ---------------------------------------------------------------------------
# 6. Abort the FORWARD sweep, resume, and the reverse sweep's data survives.
# ---------------------------------------------------------------------------
def test_a_channel_with_only_a_reverse_sweep_is_still_written(window, tmp_path):
    """`_process_multi_files` has branches for forward+reverse and for
    forward-only. A channel left holding only a reverse sweep — exactly what
    aborting the forward sweep and resuming produces — matches neither."""
    controller = window.controller
    reverse = tmp_path / "Test_ch1_reverse_temp.csv"
    reverse.write_text("placeholder")
    controller.experiment_files = {"1_reverse": str(reverse)}
    controller.processed_files = set()

    controller._parse_temp_file = lambda path: (
        pd.DataFrame({"Voltage (V)": [0.0, 0.1], "Current (A)": [0.0, 1e-3]}),
        {"Voc (V)": 1.0}, {"user_name": "yaman3397"})
    written = []
    controller._write_formatted_report = (
        lambda path, *args, **kwargs: written.append(os.path.basename(path)))
    controller._format_channel_dataframe = (
        lambda channel, data, analysis: data)

    controller._process_multi_files()

    assert written, "the completed reverse sweep produced no output file"
