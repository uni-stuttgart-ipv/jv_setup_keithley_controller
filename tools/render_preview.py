#!/usr/bin/env python3
"""
render_preview.py — offscreen UI verification.

Renders the main window (combined JV+SPO view and Advanced view) and the
calibration window without hardware or a display, saves PNG previews to
tools/previews/, and runs numeric geometry checks. Used by the `ui-preview`
and `release-check` skills; the PNGs it writes are what
tools/vision_inspector.py then sends to a vision model for a real visual
review.

No hardware, no display needed. Exit code 0 = all checks PASS, 1 = any FAIL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Must be set before PyQt5 is imported, otherwise Qt still tries to find a
# display on a headless box.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import logging  # noqa: E402
import tempfile  # noqa: E402

from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

PREVIEW_DIR = REPO_ROOT / "tools" / "previews"

# Attributes the AppController binds to on the main window. If a rename
# removes one of these, the controller silently loses a live update path
# (CLAUDE.md "controller code binds to widget attribute NAMES").
CONTROLLER_BOUND_ATTRS = [
    "username",
    "instrument_manager",
    "file_panel",
    "params_tab",
    "analysis_settings_tab",
    "analysis_panel",
    "browser_widget",
    "plot_widget",
    "log_panel",
    "bottom_stack",
    "bottom_tab_bar",
    "active_analysis_tab",
    "active_instr_tab",
    "queue_button",
    "abort_button",
    "combined_run_button",
    "combined_abort_button",
    "combined_tab",
    "combined_spo_curve",
    "combined_spo_mean",
    "combined_spo_drift",
    "combined_spo_elapsed",
    "spo_widget",
    "spo_start_button",
    "spo_abort_button",
    "update_instrument_lights",
    "update_channel_indicators",
    "update_architecture_badge",
    "_on_spo_vmpp_ready",
]


def _snapshot_logging():
    root = logging.getLogger()
    return list(root.handlers), root.level


def _restore_logging(saved_handlers, saved_level):
    root = logging.getLogger()
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(saved_level)


def _revive_blink_timer():
    """Recreate pymeasure LogWidget's parentless class-level QTimer so a
    second window in the same process can construct (SIP GCs it between
    windows). Mirrors the autouse fixture in tests/conftest.py."""
    try:
        from PyQt5.QtCore import QTimer
        from pymeasure.display.widgets.log_widget import LogWidget
    except Exception:
        return
    LogWidget._blink_qtimer = QTimer()


def _reset_directory_manager(tmp_results):
    """Isolate the DirectoryManager singleton and point RESULTS_ROOT at a
    throwaway dir so previews never touch the real output tree."""
    from solarjv_analyzer.utils.directory_manager import DirectoryManager
    import solarjv_analyzer.config as cfg

    DirectoryManager._instance = None
    cfg.RESULTS_ROOT = str(tmp_results)


def _build_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setStyle("Fusion")

    from solarjv_analyzer.gui.theme import (
        load_design_fonts,
        apply_global_plot_config,
    )
    from solarjv_analyzer.gui.style import DIALOG_STYLESHEET

    load_design_fonts()
    apply_global_plot_config()
    app.setStyleSheet(DIALOG_STYLESHEET)
    return app


def _grab(widget, path):
    """Show, settle, and save a screenshot of a widget."""
    widget.show()
    widget.raise_()
    QTest.qWait(120)
    QtWidgets.QApplication.processEvents()
    pix = widget.grab()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = pix.save(str(path), "PNG")
    return ok


def _run_checks(window, failures):
    """Numeric geometry checks — the same ones the ui-preview skill lists."""
    from pymeasure.display.widgets import BrowserWidget
    from solarjv_analyzer.gui.widgets.analysis_panel import AnalysisPanel

    browsers = window.findChildren(BrowserWidget)
    panels = window.findChildren(AnalysisPanel)

    if len(browsers) != 1:
        failures.append(f"expected 1 BrowserWidget, found {len(browsers)}")
    if len(panels) != 1:
        failures.append(f"expected 1 AnalysisPanel, found {len(panels)}")

    missing = [a for a in CONTROLLER_BOUND_ATTRS if not hasattr(window, a)]
    if missing:
        failures.append(f"controller-bound attributes missing: {missing}")

    # Combined-view plot symmetry: JV and SPO canvases must stay pixel-matched.
    # Compare the ViewBox (the actual data-plotting area), which is the true
    # "canvas" — the widget frame includes the left axis (pinned to 60px on
    # both) and heading chrome that legitimately differ in height.
    jv = window.plot_widget.plot.getViewBox()
    spo = window.combined_spo_plot.getPlotItem().getViewBox()
    jv_r = jv.sceneBoundingRect()
    spo_r = spo.sceneBoundingRect()
    dw = abs(jv_r.width() - spo_r.width())
    dh = abs(jv_r.height() - spo_r.height())
    if dw > 4 or dh > 12:
        failures.append(
            f"combined plot canvases misaligned: "
            f"JV={jv_r.width():.0f}x{jv_r.height():.0f} "
            f"SPO={spo_r.width():.0f}x{spo_r.height():.0f} "
            f"(delta {dw:.0f}px/{dh:.0f}px)"
        )


def main():
    tmp_results = tempfile.mkdtemp(prefix="solarjv_preview_")

    # pymeasure's LogWidget holds a parentless class-level QTimer that SIP
    # can GC between windows; revive it so the second window constructs.
    _revive_blink_timer()

    saved_handlers, saved_level = _snapshot_logging()
    app = _build_app()

    failures: list[str] = []

    # Each window attaches a Qt log handler to the ROOT logger that writes
    # into its own QPlainTextEdit. Once the window is closed the C++ object
    # is deleted but the handler stays attached, so the NEXT window's first
    # log() call raises "wrapped C/C++ object ... deleted". Restore the
    # logger after every window teardown (mirrors tests' clean_logging).
    try:
        # --- Main window (combined view, the default page 0) --------------
        _reset_directory_manager(tmp_results)
        from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

        window = JVAnalyzerWindow("preview_user")
        try:
            _grab(window, PREVIEW_DIR / "main_window.png")
            _run_checks(window, failures)

            # --- Advanced view ------------------------------------------
            window._on_mode_button_click(1)
            _grab(window, PREVIEW_DIR / "main_window_advanced.png")
        finally:
            window.close()
            window.deleteLater()
            app.processEvents()
            _restore_logging(saved_handlers, saved_level)

        # --- Calibration window ------------------------------------------
        _reset_directory_manager(tmp_results)
        from solarjv_analyzer.windows.calibration_window import CalibrationWindow

        # CalibrationWindow.__init__ calls _connect_hardware(), which would
        # open a real VISA connection — stub it out for the preview.
        CalibrationWindow._connect_hardware = lambda self: None

        calib = CalibrationWindow("preview_user")
        try:
            _grab(calib, PREVIEW_DIR / "calibration_window.png")
        finally:
            calib.close()
            calib.deleteLater()
            app.processEvents()
            _restore_logging(saved_handlers, saved_level)
    finally:
        _restore_logging(saved_handlers, saved_level)

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS: all geometry checks passed")
    print(f"previews written to {PREVIEW_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
