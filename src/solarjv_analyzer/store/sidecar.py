r"""Wiring the store to a window — from the outside.

Nothing in the measurement, analysis, instrument or report-writing path
changes. Two things are enough:

1. `DirectoryManager.set_base_root(staging)`. Every output path in the
   application — JV via the file panel, SPO and calibration via the directory
   manager — resolves through `get_base_root()`, and `set_base_root()` is a
   public setter with no other caller. One call redirects every temp file and
   every finished report into staging; the app goes on merging and deleting
   its temps exactly as before, somewhere else.

2. A timer that sweeps staging on a worker thread and publishes what is
   finished.

Plus two pieces of presentation: a label that says where files are really
going, and the file panel's Browse button repurposed to add a second copy
instead of redirecting the app's writes.

Everything here is defensive. Controller and sidecar code in this project bind
to widget attribute *names*, so a rename must degrade to "the store feature is
off" and never to a broken application: every step is guarded and logged.
"""

import logging
import os
import time

from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer import config
from . import paths, sweeper
from .identity import active_user

logger = logging.getLogger(__name__)

_sidecars = {}          # keep references alive: id(window) -> StoreSidecar


def is_enabled() -> bool:
    """Whether publishing should run at all.

    Off when explicitly disabled, and off on a non-Windows machine unless a
    store root has been named — otherwise a development box would try to
    publish to `S:\\Data\\JV` and simply pile files up in staging.
    """
    flag = os.environ.get("SOLARJV_STORE_ENABLED")
    if flag is not None:
        return flag.strip() not in ("0", "false", "False", "no", "")
    if not config.STORE_ENABLED:
        return False
    import sys
    if sys.platform != "win32" and not os.environ.get("SOLARJV_STORE_ROOT"):
        return False
    return True


class _SweepThread(QtCore.QThread):
    """One sweep, off the GUI thread.

    A dead mapped drive can block a bare `os.path.isdir('S:\\')` for tens of
    seconds. The project forbids blocking hardware I/O on the GUI thread; SMB
    deserves the same treatment for the same reason.
    """

    finished_sweep = QtCore.pyqtSignal(dict)

    def run(self):
        try:
            summary = sweeper.sweep()
        except Exception as exc:                      # never kill the thread
            logger.error(f"Store sweep failed: {exc}")
            summary = {"published": 0, "pending": -1, "failed": 1, "errors": [str(exc)]}
        self.finished_sweep.emit(summary)


class StoreSidecar(QtCore.QObject):
    """Publishing attached to one window."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._thread = None
        self._status_label = None
        self._timer = None

    # -- setup ----------------------------------------------------------
    def install(self) -> bool:
        if not self._redirect_output():
            return False
        self._add_status_label()
        self._rewire_browse()
        self._start_timer()
        self.sweep_now()
        return True

    def _redirect_output(self) -> bool:
        manager = getattr(self.window, "dir_manager", None)
        if manager is None or not hasattr(manager, "set_base_root"):
            logger.error("Store: no usable dir_manager on this window; feature off.")
            return False
        staging = paths.staging_root()
        try:
            os.makedirs(staging, exist_ok=True)
            manager.set_base_root(staging)
        except Exception as exc:
            logger.error(f"Store: cannot use staging root {staging}: {exc}")
            return False

        # The file panel caches its directory: JVAnalyzerWindow sets it once in
        # __init__ and again only on a mode switch. set_base_root() alone does
        # not touch it, so without this refresh the FIRST run after startup
        # would still write to the pre-attach location and nothing would be
        # published until the operator happened to toggle a view.
        for refresh in ("_update_save_directory", "_update_display_directory"):
            method = getattr(self.window, refresh, None)
            if callable(method):
                try:
                    method()
                except Exception as exc:
                    logger.debug(f"Store: {refresh}() failed: {exc}")

        logger.info(f"Store: working files staged in {staging}")
        return True

    def _add_status_label(self) -> None:
        panel = getattr(self.window, "file_panel", None)
        layout = panel.layout() if panel is not None else None
        if layout is None:
            return
        try:
            self._status_label = QtWidgets.QLabel()
            self._status_label.setWordWrap(True)
            self._status_label.setStyleSheet(
                "color: #41484b; font-size: 11px; background: transparent;"
                " padding: 2px 0;"
            )
            if isinstance(layout, QtWidgets.QFormLayout):
                layout.addRow("", self._status_label)
            else:
                layout.addWidget(self._status_label)
        except Exception as exc:
            logger.debug(f"Store: could not add the status label: {exc}")
            self._status_label = None
        self._refresh_status()

    def _rewire_browse(self) -> None:
        """Browse becomes "add a second copy" instead of moving the output.

        Done from here rather than by editing the widget, so the feature stays
        additive. If the button is ever renamed this logs and gives up, and
        the store still works — only the second-copy option disappears.
        """
        panel = getattr(self.window, "file_panel", None)
        button = getattr(panel, "browse_button", None) if panel is not None else None
        if button is None:
            return
        try:
            entry = getattr(panel, "directory_input", None)
            if entry is not None:
                entry.setReadOnly(True)
            try:
                button.clicked.disconnect()
            except TypeError:
                pass                                  # nothing connected
            button.setText("Also save a copy to…")
            button.setToolTip(
                "Every report always goes to the protected store on S:.\n"
                "Choose a folder here to also keep a second copy."
            )
            button.clicked.connect(self._pick_extra_copy)
        except Exception as exc:
            logger.debug(f"Store: could not rewire the Browse button: {exc}")

    def _start_timer(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(int(config.STORE_SWEEP_INTERVAL_MS))
        self._timer.timeout.connect(self.sweep_now)
        self._timer.start()

    # -- actions --------------------------------------------------------
    def _pick_extra_copy(self) -> None:
        current = sweeper.extra_target()
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self.window, "Also save a copy of every report to…",
            current or os.path.expanduser("~"),
        )
        if chosen:
            sweeper.set_extra_target(chosen)
            logger.info(f"Store: second copy enabled -> {chosen}")
        elif current:
            answer = QtWidgets.QMessageBox.question(
                self.window, "Second copy",
                f"Stop keeping a second copy in:\n{current}?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                sweeper.set_extra_target("")
                logger.info("Store: second copy disabled")
        self._refresh_status()

    def sweep_now(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._thread = _SweepThread(self)
        self._thread.finished_sweep.connect(self._on_swept)
        self._thread.start()

    def _on_swept(self, summary: dict) -> None:
        if summary.get("published"):
            logger.info(f"Store: published {summary['published']} file(s)")
        self._refresh_status(summary)

    def _refresh_status(self, summary: dict = None) -> None:
        if self._status_label is None:
            return
        try:
            user = active_user()
        except RuntimeError:
            user = "?"          # shown until the operator signs in
        destination = paths.destination_dir(user, paths.store_date(), "…")
        text = f"Saved to  {destination}"
        second = sweeper.extra_target()
        if second:
            text += f"\nSecond copy in  {second}"
        pending = (summary or {}).get("pending")
        if pending is None:
            pending = sweeper.pending_count()
        if pending and pending > 0:
            text += f"\n⚠ {pending} file(s) awaiting publish — will retry automatically"
        self._status_label.setText(text)

    def shutdown(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(5000)


def outstanding_count(staging: str = None) -> int:
    """Everything still in staging, settled or not.

    `pending_count()` deliberately counts only files that have been quiet for
    `STORE_QUIET_SECONDS`, because that is what the sweeper is willing to
    publish on this tick. For "is it safe to close?" that is the wrong
    question: a file written one second ago is not pending yet but is very
    much outstanding. Passing `quiet_seconds=0` counts it.
    """
    try:
        return len(sweeper.find_finished(staging, quiet_seconds=0))
    except Exception as exc:                          # noqa: BLE001
        logger.debug(f"Could not count staged files: {exc}")
        return 0


def flush_before_exit(parent=None, timeout_s: float = 60.0) -> bool:
    """Publish everything staged before this session ends.

    The store folder is named after the SIGNED-IN OPERATOR, so a file left in
    staging when they log out or close the app would be published later under
    whoever signs in next — their work, someone else's folder, on a share that
    refuses deletions. Rather than documenting that as a limitation, hold the
    exit until staging is empty.

    Waits for files that are still being written, since the sweeper will not
    touch a file until it has been quiet for `STORE_QUIET_SECONDS`; the
    timeout is therefore comfortably longer than that.

    Returns:
        bool: True when it is safe to proceed — either everything published,
            or the operator explicitly chose to leave files behind.
    """
    if not is_enabled():
        return True
    outstanding = outstanding_count()
    if not outstanding:
        return True

    logger.info(f"Publishing {outstanding} staged file(s) before exit…")
    progress = QtWidgets.QProgressDialog(
        f"Saving {outstanding} file(s) to the store…\n\n"
        "This finishes writing your results to S: so they are filed under "
        "your name.",
        "Leave them for later", 0, 0, parent)
    progress.setWindowTitle("Finishing up")
    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    progress.show()
    QtWidgets.QApplication.processEvents()

    deadline = time.monotonic() + max(1.0, float(timeout_s))
    try:
        while time.monotonic() < deadline:
            if progress.wasCanceled():
                return _confirm_leaving_files(parent, outstanding_count())
            try:
                sweeper.sweep()
            except Exception as exc:                  # noqa: BLE001
                logger.warning(f"Publish attempt failed: {exc}")
            remaining = outstanding_count()
            if not remaining:
                logger.info("Store is in sync; safe to exit.")
                return True
            progress.setLabelText(
                f"Saving {remaining} file(s) to the store…\n\n"
                "Waiting for files still being written.")
            QtWidgets.QApplication.processEvents()
            time.sleep(0.5)
    finally:
        progress.close()

    # Timed out — almost always the share being unreachable. Never trap the
    # operator in a dialog they cannot satisfy; tell them what is at stake and
    # let them decide.
    return _confirm_leaving_files(parent, outstanding_count(), timed_out=True)


def _confirm_leaving_files(parent, remaining: int, timed_out: bool = False) -> bool:
    """Ask whether to exit with files still unpublished."""
    if not remaining:
        return True
    reason = ("The store could not be reached in time."
              if timed_out else "Publishing was cancelled.")
    answer = QtWidgets.QMessageBox.warning(
        parent, "Files not yet saved to the store",
        f"{reason}\n\n"
        f"{remaining} file(s) are still waiting to be copied to S:.\n\n"
        "They are safe on this computer and will be published later — but "
        "they will then be filed under whoever is signed in at that time, "
        "not under your name.\n\n"
        "Leave them and continue?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.No,
    )
    proceed = answer == QtWidgets.QMessageBox.Yes
    logger.warning(
        "Exiting with %d unpublished file(s): operator chose %s.",
        remaining, "to continue" if proceed else "to stay")
    return proceed


def attach(window) -> bool:
    """Turn on store publishing for `window`. Safe to call more than once.

    Never raises: a failure here disables the feature and leaves the
    application exactly as it was.
    """
    if not is_enabled():
        logger.info("Store publishing is disabled.")
        return False
    key = id(window)
    if key in _sidecars:
        return True
    try:
        sidecar = StoreSidecar(window)
        if not sidecar.install():
            return False
        _sidecars[key] = sidecar
        logger.info(
            f"Store publishing active: {paths.store_root()} "
            f"(sweep every {config.STORE_SWEEP_INTERVAL_MS / 1000:.0f}s)"
        )
        return True
    except Exception as exc:
        logger.error(f"Store: attach failed, feature off: {exc}")
        return False


def detach(window) -> None:
    sidecar = _sidecars.pop(id(window), None)
    if sidecar is not None:
        sidecar.shutdown()
