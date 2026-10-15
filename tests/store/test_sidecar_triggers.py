"""What wakes the sweeper, and what happens when a wake-up lands mid-sweep.

These cover the two trigger-side changes that replaced the 20 s poll:

  * a `QFileSystemWatcher` over the staging folders, so a report that has just
    been renamed into place is published straight away rather than up to a
    poll interval later;
  * a re-run flag, because the watcher fires once per change and the old code
    dropped any request that arrived while a sweep was already running. With a
    poll that was harmless. With the watcher the dropped request is the one
    carrying the finished report.

Deliberately NOT a full-window test: `test_sidecar_integration` covers the
`attach()` wiring, and building a window here would make these slow and
order-dependent for no extra coverage.
"""
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from solarjv_analyzer.store import sidecar as sidecar_module  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _StubThread(QtCore.QObject):
    """Stands in for `_SweepThread`: records starts, never touches the disk."""

    finished_sweep = QtCore.pyqtSignal(dict)
    starts = 0

    def __init__(self, _parent=None):
        super().__init__()
        self._running = False

    def isRunning(self):
        return self._running

    def start(self):
        self._running = True
        type(self).starts += 1

    def wait(self, _msecs=None):
        self._running = False
        return True


@pytest.fixture
def side(app, monkeypatch, tmp_path):
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(tmp_path / "staging"))
    _StubThread.starts = 0
    monkeypatch.setattr(sidecar_module, "_SweepThread", _StubThread)
    # Parentless on purpose. StoreSidecar makes itself a child of the window
    # it is given, so a throwaway QObject here would be garbage-collected at
    # teardown and take the sidecar's C++ object with it — which surfaces as a
    # hard interpreter crash in a later test, not as a failure here. None of
    # the methods under test touch `window`.
    obj = sidecar_module.StoreSidecar(None)
    yield obj
    obj.shutdown()


# --- the re-run flag ---------------------------------------------------------

def test_a_request_during_a_sweep_is_remembered_not_dropped(side):
    side.sweep_now()                       # first sweep starts
    assert _StubThread.starts == 1
    assert not side._rerun_requested

    side.sweep_now()                       # arrives while the first is running
    assert _StubThread.starts == 1, "a second sweep ran concurrently"
    assert side._rerun_requested, "the request was dropped"


def test_the_remembered_request_fires_when_the_sweep_ends(side):
    side.sweep_now()
    side.sweep_now()                       # queued
    side._thread._running = False          # the sweep finishes
    side._on_swept({})

    assert _StubThread.starts == 2, "the queued request never ran"
    assert not side._rerun_requested


def test_a_quiet_sweep_does_not_re_fire(side):
    """Nothing asked for a re-run, so the chain stops — no busy loop."""
    side.sweep_now()
    side._thread._running = False
    side._on_swept({})

    assert _StubThread.starts == 1


# --- the watcher -------------------------------------------------------------

def test_the_watcher_covers_staging_and_its_subfolders(side, tmp_path):
    staging = tmp_path / "staging"
    (staging / "yaman" / "09-09-2026" / "JV").mkdir(parents=True)
    (staging / "yaman" / "09-09-2026" / "SPO").mkdir(parents=True)

    side._sync_watched_dirs()

    watched = set(side._watcher.directories())
    assert str(staging) in watched
    assert str(staging / "yaman" / "09-09-2026" / "JV") in watched
    assert str(staging / "yaman" / "09-09-2026" / "SPO") in watched


def test_new_folders_are_picked_up_after_a_sweep(side, tmp_path):
    """A run at a new date or mode creates folders nobody registered."""
    staging = tmp_path / "staging"
    (staging / "yaman" / "09-09-2026" / "JV").mkdir(parents=True)
    side._sync_watched_dirs()

    (staging / "yaman" / "10-09-2026" / "Calibration").mkdir(parents=True)
    side._thread = None
    side._on_swept({})                      # the sweep re-syncs the watch set

    watched = set(side._watcher.directories())
    assert str(staging / "yaman" / "10-09-2026" / "Calibration") in watched


def test_folders_that_disappear_are_dropped(side, tmp_path):
    """`_prune_empty_dirs` removes staging folders once they are emptied."""
    staging = tmp_path / "staging"
    gone = staging / "yaman" / "09-09-2026" / "JV"
    gone.mkdir(parents=True)
    side._sync_watched_dirs()
    assert str(gone) in set(side._watcher.directories())

    os.rmdir(str(gone))
    side._sync_watched_dirs()

    assert str(gone) not in set(side._watcher.directories())


def test_missing_staging_is_not_an_error(side):
    side._sync_watched_dirs()               # staging never created; must not raise


def test_a_directory_change_schedules_exactly_one_sweep(side, app, tmp_path):
    """A single report fires several events; they must coalesce into one sweep."""
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    side._sync_watched_dirs()

    side._on_directory_changed(str(staging))
    side._on_directory_changed(str(staging))
    side._on_directory_changed(str(staging))
    assert _StubThread.starts == 0, "a sweep ran before the debounce elapsed"
    assert side._debounce.isActive()

    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while _StubThread.starts == 0 and deadline.elapsed() < 3000:
        app.processEvents()

    assert _StubThread.starts == 1, "the burst did not coalesce into one sweep"


def test_shutdown_stops_watching(side, tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir(parents=True)
    side._sync_watched_dirs()
    assert side._watcher.directories()

    side.shutdown()

    assert side._watcher.directories() == []
    assert not side._debounce.isActive()


# ---------------------------------------------------------------------------
# attach()/detach() bookkeeping
#
# Nothing else stops a sidecar. Its sweep timer and filesystem watcher live
# as long as the process does, so a window that goes away without a detach
# leaves a sweeper running against a staging root nobody is writing to any
# more -- and on a lab PC that sees a login per operator, those accumulate.
# ---------------------------------------------------------------------------

class _FakeWindow(QtCore.QObject):
    """Enough window for attach(): a dir_manager it can repoint."""

    def __init__(self, staging_root):
        super().__init__()
        self._root = staging_root

        class _DM:
            def set_base_root(self_inner, root):
                pass

        self.dir_manager = _DM()


@pytest.fixture
def attachable(tmp_path, monkeypatch):
    """store.attach() enabled, pointed at a throwaway staging root."""
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "1")
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(tmp_path / "staging"))
    (tmp_path / "staging").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    for key in list(sidecar_module._sidecars):
        sidecar_module._retire(key)


def test_detach_stops_the_sweep_timer(app, attachable):
    """The point of detach: no timer, no watcher, nothing left running."""
    window = _FakeWindow(attachable)
    assert sidecar_module.attach(window) is True
    side = sidecar_module._sidecars[id(window)][1]
    assert side._timer.isActive()

    sidecar_module.detach(window)

    assert not side._timer.isActive()
    assert side._watcher.directories() == []
    assert id(window) not in sidecar_module._sidecars


def test_detach_twice_is_harmless(app, attachable):
    window = _FakeWindow(attachable)
    sidecar_module.attach(window)
    sidecar_module.detach(window)
    sidecar_module.detach(window)          # must not raise


def test_a_recycled_window_id_still_gets_its_own_sidecar(app, attachable):
    """CPython reuses id() once an object is freed.

    The registry is keyed by id(window), so a new window landing on a freed
    one's address used to look like "already attached" -- attach() returned
    True and installed nothing, and that operator's reports would sit in
    staging for the rest of the session with no error anywhere.
    """
    first = _FakeWindow(attachable)
    assert sidecar_module.attach(first) is True
    stale = sidecar_module._sidecars[id(first)][1]

    # Forge the collision rather than waiting for the allocator to produce
    # one: move the entry to the second window's id behind attach()'s back.
    second = _FakeWindow(attachable)
    sidecar_module._sidecars[id(second)] = sidecar_module._sidecars.pop(id(first))

    assert sidecar_module.attach(second) is True

    installed = sidecar_module._sidecars[id(second)]
    assert installed[0]() is second, "the registry still points at the dead window"
    assert installed[1] is not stale, "the new window inherited a stale sidecar"
    assert not stale._timer.isActive(), "the stale sweeper was left running"


def test_a_destroyed_window_retires_its_sidecar(app, attachable):
    """Belt and braces for a window that is dropped without a detach()."""
    window = _FakeWindow(attachable)
    assert sidecar_module.attach(window) is True
    key = id(window)
    side = sidecar_module._sidecars[key][1]

    window.deleteLater()
    del window
    # processEvents() does not flush DeferredDelete; those are only delivered
    # when the posting event loop returns. Ask for them explicitly.
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    app.processEvents()

    assert key not in sidecar_module._sidecars, "a destroyed window kept sweeping"
    # Its timer went with it: Qt owns the sidecar's children, so reaching for
    # one now raises rather than reporting a state. That is the proof.
    with pytest.raises(RuntimeError):
        side._timer.isActive()
