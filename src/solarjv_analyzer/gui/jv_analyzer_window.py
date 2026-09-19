"""
Main Window for JV Analyzer Application

Provides the graphical user interface for the JV measurement system,
including parameter input, instrument control, live plotting, and log display.
"""

import logging
import re
import os
import sys
from datetime import datetime

from pathlib import Path
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtWidgets, QtCore, QtGui
from pymeasure.display.widgets import PlotWidget, BrowserWidget

from solarjv_analyzer.gui.widgets.log_panel import LogPanel

from solarjv_analyzer.config import RESULTS_ROOT, DATE_FORMAT
from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.utils.directory_manager import DirectoryManager

from .widgets.parameter_tab import ParameterTab
from .widgets.instrument_tab import InstrumentTab
from .widgets.analysis_settings_tab import AnalysisSettingsTab
from .widgets.file_panel import FilePanel
from .widgets.analysis_panel import AnalysisPanel
from .widgets.combined_tab import CombinedTab
from .app_controller import AppController
from .style import BASE_STYLESHEET

# SPO is a self-contained, optional module: if the spo/ package is removed,
# the JV application must still compile and run (SPO mode is simply hidden).
try:
    from solarjv_analyzer.spo.spo_widget import SpoWidget, SpoParameterTab
    SPO_AVAILABLE = True
except ImportError:
    SpoWidget = None
    SpoParameterTab = None
    SPO_AVAILABLE = False

logger = logging.getLogger(__name__)


class TightStackedWidget(QtWidgets.QStackedWidget):
    """QStackedWidget that sizes to the *current* page, not the maximum page.

    Standard QStackedWidget.sizeHint() returns the largest page's size.
    This subclass delegates to the current widget so the stack tightens
    around whichever page is visible.
    """

    def sizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.minimumSizeHint()
        return super().minimumSizeHint()

    def hasHeightForWidth(self):
        # The base QStackedWidget/QStackedLayout reports hasHeightForWidth()
        # as True and computes heightForWidth() as the *maximum* across all
        # pages (including hidden ones). Since sizeHint()/minimumSizeHint()
        # above are already tightened to the current page, box layouts must
        # not fall back to that stale height-for-width path, or dead space
        # reappears below the current page whenever another page is taller.
        return False


class TightTabWidget(QtWidgets.QTabWidget):
    """QTabWidget that sizes to the *current* page, not the maximum page."""

    def sizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.sizeHint()
        return super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        if w is not None:
            return w.minimumSizeHint()
        return super().minimumSizeHint()

    def hasHeightForWidth(self):
        # See TightStackedWidget.hasHeightForWidth() above: QTabWidget's
        # internal stack otherwise reports the tallest tab's height, which
        # would leave dead space below shorter tabs (e.g. SPO Parameters).
        return False


class ShrinkableLabel(QtWidgets.QLabel):
    """QLabel whose *minimum* width is zero.

    A QLabel without word wrap reports its full text width as its
    ``minimumSizeHint()``, so a row of labels can inflate a panel's minimum
    width and unbalance a splitter (the combined-view SPO heading chips
    otherwise push the horizontal splitter ~34px off 50/50, making the two
    graph canvases uneven). Keep the natural ``sizeHint()`` — so the text
    renders in full when there is room — but report a zero-width minimum so
    the label may shrink (and clip) when space is tight. Pair with a
    horizontal ``QSizePolicy.Maximum`` so the preferred width stays bounded
    by the text.
    """

    def minimumSizeHint(self):
        return QtCore.QSize(0, super().minimumSizeHint().height())


class _HardwareCheckThread(QtCore.QThread):
    """One hardware status check, off the GUI thread.

    Enumerating serial ports is a pure read, but opening one is not: a busy or
    ghost COM port can block for seconds, and the project forbids serial I/O on
    the GUI thread. So the whole check runs here and reports back by signal.

    `may_connect` is False while a measurement is running — the check must then
    only *look*, never open, close or touch an instrument.
    """

    checked = QtCore.pyqtSignal(dict)

    def __init__(self, instrument_manager, may_connect=True, parent=None):
        super().__init__(parent)
        self._manager = instrument_manager
        self._may_connect = may_connect

    def run(self):
        from solarjv_analyzer.instruments import port_status
        from solarjv_analyzer.instruments.port_resolver import (
            active_keithley_resource, active_mux_port,
        )

        # The RESOLVED ports, not config.py — otherwise the lights would watch
        # the wrong port precisely when dynamic resolution has done its job.
        MUX_PORT = active_mux_port()
        GPIB_ADDRESS = active_keithley_resource()

        result = {"mux_error": "", "keithley_error": ""}
        try:
            available = port_status.port_names()
            keithley_port = port_status.visa_to_port(GPIB_ADDRESS)

            mux_present = port_status.is_port_present(MUX_PORT, available)
            keithley_present = port_status.is_port_present(keithley_port, available)

            # Only try to open a port that Windows is actually offering. This
            # is what keeps a missing instrument from being hammered every few
            # seconds, and it costs one registry read rather than a failed
            # open with its timeout.
            if self._may_connect and mux_present and not self._manager.is_mux_alive():
                try:
                    self._manager.connect_mux(simulation=False)
                except Exception as exc:
                    result["mux_error"] = str(exc)

            result.update(
                mux_present=mux_present,
                keithley_present=keithley_present,
                mux_port=MUX_PORT,
                keithley_port=keithley_port or GPIB_ADDRESS,
                mux_description=port_status.describe(MUX_PORT),
                keithley_description=port_status.describe(keithley_port),
                # Connected means BOTH: we hold an open handle, and the adapter
                # is still plugged in. `is_open` alone stays True after the USB
                # device is pulled.
                mux_connected=self._manager.is_mux_alive() and mux_present,
                keithley_connected=self._manager.is_keithley_alive() and keithley_present,
            )
        except Exception as exc:
            result["error"] = str(exc)
        self.checked.emit(result)


_VISA_RESOURCE_RE = re.compile(
    r"^(ASRL\d+|GPIB\d*::\d+(::\d+)?|TCPIP\d*::[^:]+(::.+)?|USB\d*::.+)::INSTR$",
    re.IGNORECASE)


def _looks_like_visa_resource(text: str) -> bool:
    """Catch the obvious typo before it becomes a VISA error at run start."""
    return bool(_VISA_RESOURCE_RE.match(text.strip()))


class JVAnalyzerWindow(QtWidgets.QMainWindow):
    """
    Main application window for J‑V measurement and analysis.

    Provides:
    - Parameter input tabs for experiment configuration
    - Live plot of current vs voltage during measurement
    - Log tab for system messages and debug output
    - Browser for managing multiple experiments
    - Analysis panel for displaying solar cell metrics
    """
    # Signal emitted when the user confirms logout
    logged_out = QtCore.pyqtSignal()

    # -------------------------------------------------------------------------
    # Modern stylesheet
    @staticmethod
    def _app_stylesheet() -> str:
        return BASE_STYLESHEET + """
            /* Dock Widget */
            QDockWidget { border: none; }
            QDockWidget::title {
                font-weight: 600;
                font-size: 14px;
                color: #0f172a;
                padding: 12px;
                background: #ffffff;
                border-bottom: 1px solid #f1f5f9;
            }

            /* Primary Button (Queue / Start) */
            QPushButton#QueueButton {
                background-color: #053a46;
                color: white;
                border: none;
                font-weight: 600;
            }
            QPushButton#QueueButton:hover { background-color: #24515e; }
            QPushButton#QueueButton:pressed { background-color: #047857; }
            QPushButton#QueueButton:disabled {
                background-color: #e2e8f0;
                color: #64748b;
            }

            /* Danger Button (Abort) */
            QPushButton#AbortButton {
                background-color: #ef4444;
                color: white;
                border: none;
                font-weight: 600;
            }
            QPushButton#AbortButton:hover { background-color: #dc2626; }
            QPushButton#AbortButton:pressed { background-color: #b91c1c; }
            QPushButton#AbortButton:disabled {
                background-color: #e2e8f0;
                color: #64748b;
            }

            /* Mode Toggle Buttons (JV Sweep / SPO) — muted blue */
            QPushButton#ModeButton {
                font-weight: 600;
            }
            QPushButton#ModeButton:checked {
                background-color: #24515e;
                color: white;
                border: 1px solid #053a46;
            }
            QPushButton#ModeButton:checked:hover { background-color: #053a46; }
            QPushButton#ModeButton:checked:pressed { background-color: #24515e; }

            /* Hold-voltage source picker (Manual / Quick JV) in Advanced→SPO.
               Both options must read as CHOICES. The previous style painted
               the unselected one pale grey on transparent, which is exactly
               what a disabled button looks like. */
            QPushButton#HoldModeButton {
                font-size: 12px;
                font-weight: 600;
                padding: 5px 16px;
                border-radius: 10px;
                color: #053a46;
                background: #ffffff;
                border: 1px solid #a1cddd;
            }
            QPushButton#HoldModeButton:hover {
                background: #eef6f8;
                border-color: #24515e;
            }
            QPushButton#HoldModeButton:checked {
                background-color: #053a46;
                color: #ffffff;
                border: 1px solid #053a46;
            }
            QPushButton#HoldModeButton:checked:hover { background-color: #24515e; }
            QPushButton#HoldModeButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
                border-color: #e2e8f0;
            }

            /* Save Plot — primary teal (design-system accent) */
            QPushButton#SavePlotButton {
                background-color: #053a46;
                color: white;
                border: none;
                font-weight: 600;
                margin: 8px 0px;
            }
            QPushButton#SavePlotButton:hover { background-color: #24515e; }

            /* ── Bottom Tab Bar – pill-style, centre-aligned ── */
            QTabBar#BottomTabBar {
                alignment: center;
            }
            QTabBar#BottomTabBar::tab {
                padding: 10px 28px;
                min-width: 160px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 8px;
                margin: 4px 6px;
                color: #64748b;
                background: transparent;
                border: 1px solid #e2e8f0;
            }
            QTabBar#BottomTabBar::tab:hover {
                color: #0f172a;
                background: #f1f5f9;
                border-color: #cbd5e1;
            }
            QTabBar#BottomTabBar::tab:selected {
                background-color: #053a46;
                color: white;
                border: 1px solid #24515e;
            }

            /* ── Plot / Log switch above the JV+SPO graphs ── */
            QTabBar#GraphTabBar::tab {
                padding: 5px 22px;
                min-width: 74px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 7px;
                margin: 0px 4px;
                color: #64748b;
                background: transparent;
                border: 1px solid #e2e8f0;
            }
            QTabBar#GraphTabBar::tab:hover {
                color: #0f172a;
                background: #f1f5f9;
                border-color: #cbd5e1;
            }
            QTabBar#GraphTabBar::tab:selected {
                background-color: #053a46;
                color: white;
                border: 1px solid #24515e;
            }

            /* ── Live log page ── */
            QPlainTextEdit#LogView {
                background: #fbfcfc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
                color: #1a1c1d;
                selection-background-color: #cfe3e9;
            }
            QComboBox#LogLevelFilter {
                padding: 3px 8px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                font-size: 12px;
                min-width: 130px;
            }
            QPushButton#LogAction {
                padding: 4px 14px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background: white;
                color: #41484b;
                font-size: 12px;
            }
            QPushButton#LogAction:hover {
                background: #f1f5f9;
                border-color: #cbd5e1;
                color: #0f172a;
            }
            QCheckBox#LogFollow { font-size: 12px; color: #64748b; }
        """


    def __init__(self, username=None, instrument_manager=None):
        """
        Initialize the main window.

        Args:
            username: Name of the user operating the system
            instrument_manager: The process-wide InstrumentManager handed over
                by the calibration window. Passing the manager itself (rather
                than copying its instrument objects into a second manager)
                keeps a single owner for the VISA session — two owners meant
                whichever disconnected first silently invalidated the other's
                handle. A manager is created here only when the window is
                built standalone (tests, previews).
        """
        super().__init__()
        self.username = username
        self.instrument_manager = (
            instrument_manager if instrument_manager is not None else InstrumentManager()
        )

        # Initialize directory manager FIRST (before UI)
        self.dir_manager = DirectoryManager(username=self.username, parent=self, mode="Main")
        self.dir_manager.set_mode("Main") 

        self.setWindowTitle("Custom JV Analyzer")
        self.resize(1200, 720)
        self.setMinimumSize(1000, 700)

        # Apply modern stylesheet globally
        self.setStyleSheet(self._app_stylesheet())

        # Build the user interface (this will also set up the file panel)
        self._layout()

        # Initialize controller
        self.controller = AppController(self)

        # Connect signals and set initial state
        self.connect_signals()
        self._update_save_directory()

        # Filename validation — buttons disabled until user enters a name
        self._filename_valid = False
        self.file_panel.filename_input.textChanged.connect(
            self._on_filename_changed
        )
        self._on_filename_changed("")  # enforce initial state

        # Configure logging to display in Log tab
        self._setup_logging()

        # Debug checkbox connections are wired after all tabs are created
        # (see end of _layout)

        # Connect parameter changes for NPLC preview
        self._connect_nplc_preview_signals()
        self._update_nplc_from_sweep_rate()

        # Initial UI state
        self.browser_widget.show_button.setEnabled(False)
        self.browser_widget.hide_button.setEnabled(False)
        self.browser_widget.clear_button.setEnabled(False)

        logger.info("Application started")

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _layout(self):
        """Construct the user interface layout."""
        # Before any plot exists: pyqtgraph's ViewBox registry still holds
        # every ViewBox from a previously closed window, and building a new
        # one walks the whole list. See forget_dead_viewboxes().
        from solarjv_analyzer.gui.theme import forget_dead_viewboxes
        forget_dead_viewboxes()

        self.main = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main)

        # ---- Core plot widgets (created once, shared by both views) -------
        self.plot_widget = PlotWidget(
            name="Plot",
            columns=JVProcedure.DATA_COLUMNS,
            x_axis="Voltage (V)",
            y_axis="Current (A)"
        )
        self.plot_widget.plot.showGrid(x=True, y=True, alpha=0.3)
        # Live log. One LogPanel per view (see the class docstring for why
        # pymeasure's LogWidget could not be used); they share a single root
        # handler, so all three show the same stream.
        self.log_panel = LogPanel()

        # ---- Sidebar mode selector: button bar + stacked content ----------
        # Using QStackedWidget (not QTabWidget) so each page sizes
        # independently to its content height — no "max page" penalty.

        L1_TAB_STYLE = """
            QPushButton {
                padding: 10px 24px;
                font-size: 13px;
                font-weight: 500;
                color: #94a3b8;
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                min-width: 60px;
            }
            QPushButton:hover {
                color: #64748b;
                background: transparent;
            }
            QPushButton:checked {
                color: #0f172a;
                font-weight: 600;
                background: transparent;
                border-bottom: 2px solid #053a46;
            }
        """

        self._mode_btn_jvspo = QtWidgets.QPushButton("JV + SPO")
        self._mode_btn_jvspo.setCheckable(True)
        self._mode_btn_jvspo.setChecked(True)
        self._mode_btn_jvspo.setCursor(QtCore.Qt.PointingHandCursor)
        self._mode_btn_jvspo.setStyleSheet(L1_TAB_STYLE)
        self._mode_btn_jvspo.clicked.connect(lambda: self._on_mode_button_click(0))

        self._mode_btn_adv = QtWidgets.QPushButton("Advanced")
        self._mode_btn_adv.setCheckable(True)
        self._mode_btn_adv.setCursor(QtCore.Qt.PointingHandCursor)
        self._mode_btn_adv.setStyleSheet(L1_TAB_STYLE)
        self._mode_btn_adv.clicked.connect(lambda: self._on_mode_button_click(1))

        self._mode_btn_group = QtWidgets.QButtonGroup(self)
        self._mode_btn_group.setExclusive(True)
        self._mode_btn_group.addButton(self._mode_btn_jvspo, 0)
        self._mode_btn_group.addButton(self._mode_btn_adv, 1)

        mode_bar = QtWidgets.QHBoxLayout()
        mode_bar.setSpacing(16)
        mode_bar.addStretch()
        mode_bar.addWidget(self._mode_btn_jvspo)
        mode_bar.addWidget(self._mode_btn_adv)
        mode_bar.addStretch()

        # ---- Inner tab styles (L2 + L3) -----------------------------------
        INNER_TAB_STYLE = """
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                padding: 8px 16px;
                font-size: 12px;
                font-weight: 400;
                color: #94a3b8;
                background: transparent;
                border: none;
                margin: 0px 2px;
                min-width: 60px;
            }
            QTabBar::tab:hover { color: #64748b; background: transparent; }
            QTabBar::tab:selected {
                color: #0f172a;
                font-weight: 600;
                background: transparent;
                border: none;
            }
            QPushButton#ModeButton {
                font-weight: 500;
                font-size: 11px;
                padding: 5px 14px;
                border-radius: 10px;
                color: #94a3b8;
                background: transparent;
                border: 1px solid #e2e8f0;
            }
            QPushButton#ModeButton:hover {
                color: #64748b;
                background: #f8fafc;
            }
            QPushButton#ModeButton:checked {
                color: #0f172a;
                font-weight: 600;
                background: #f1f5f9;
                border: 1px solid #cbd5e1;
            }
        """

        # -- Page 0: Combined JV + SPO (default) ---------------------------
        self.combined_tab = CombinedTab()
        self.combined_instr_tab = InstrumentTab()
        self.combined_analysis_tab = AnalysisSettingsTab()
        combined_tab_widget = TightTabWidget()
        combined_tab_widget.addTab(self.combined_tab, "Parameters")
        combined_tab_widget.addTab(self.combined_instr_tab, "Instrument")
        combined_tab_widget.addTab(self.combined_analysis_tab, "Analysis")
        combined_tab_widget.setDocumentMode(True)
        combined_tab_widget.setStyleSheet(INNER_TAB_STYLE)
        combined_tab_widget.tabBar().setExpanding(False)

        # -- Tab 1: Advanced (existing JV / SPO as sub-tabs) ----------------
        # Preserve the EXISTING mode buttons + params_stack + input_tabs in
        # a container widget — zero changes to their internal logic.
        advanced_container = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(advanced_container)
        advanced_layout.setContentsMargins(12, 8, 12, 8)
        advanced_layout.setSpacing(12)

        # Existing mode toggle: JV Sweep vs SPO
        self.jv_mode_button = QtWidgets.QPushButton("JV Sweep")
        self.spo_mode_button = QtWidgets.QPushButton("SPO")
        for btn in (self.jv_mode_button, self.spo_mode_button):
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setObjectName("ModeButton")
        self.jv_mode_button.setChecked(True)

        self.mode_button_group = QtWidgets.QButtonGroup(self)
        self.mode_button_group.setExclusive(True)
        self.mode_button_group.addButton(self.jv_mode_button)
        self.mode_button_group.addButton(self.spo_mode_button)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(self.jv_mode_button)
        mode_row.addWidget(self.spo_mode_button)
        advanced_layout.addLayout(mode_row)

        # Existing parameter tabs with their own Instrument/Analysis instances
        input_tabs = TightTabWidget()
        self.params_tab = ParameterTab()
        self.instr_tab = InstrumentTab()
        self.analysis_settings_tab = AnalysisSettingsTab()

        # SPO fields share the "Parameters" tab slot with JV params.
        # Using a plain QWidget with show/hide (not QStackedWidget) so
        # each child's natural height is preserved — no forced stretching
        # to the largest page.
        self.params_stack = QtWidgets.QWidget()
        params_stack_layout = QtWidgets.QVBoxLayout(self.params_stack)
        params_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.params_tab = ParameterTab()
        params_stack_layout.addWidget(self.params_tab)
        if SPO_AVAILABLE:
            self.spo_param_tab = SpoParameterTab(self)
        else:
            logger.warning("SPO module unavailable; SPO mode disabled.")
            self.spo_param_tab = QtWidgets.QWidget()
        self.spo_param_tab.hide()
        params_stack_layout.addWidget(self.spo_param_tab)

        input_tabs.addTab(self.params_stack, "Parameters")
        input_tabs.addTab(self.instr_tab, "Instrument")
        input_tabs.addTab(self.analysis_settings_tab, "Analysis")
        input_tabs.setDocumentMode(True)
        input_tabs.setStyleSheet(INNER_TAB_STYLE)
        input_tabs.tabBar().setExpanding(False)
        advanced_layout.addWidget(input_tabs)

        # ---- TightStackedWidget: each page sizes to its own content -----------
        self.sidebar_mode_tabs = TightStackedWidget()
        self.sidebar_mode_tabs.addWidget(combined_tab_widget)    # page 0
        self.sidebar_mode_tabs.addWidget(advanced_container)      # page 1

        # ---- Common sidebar elements (shared by both modes) ---------------
        self.file_panel = FilePanel()

        # JV buttons (shown in Advanced tab)
        self.queue_button = QtWidgets.QPushButton("Queue")
        self.queue_button.setObjectName("QueueButton")
        self.abort_button = QtWidgets.QPushButton("Abort")
        self.abort_button.setObjectName("AbortButton")

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(12)
        button_layout.addWidget(self.queue_button)
        button_layout.addWidget(self.abort_button)

        # Advanced-mode SPO buttons
        self.spo_start_button = QtWidgets.QPushButton("Start SPO")
        self.spo_start_button.setObjectName("QueueButton")
        self.spo_start_button.setEnabled(False)
        self.spo_abort_button = QtWidgets.QPushButton("Abort SPO")
        self.spo_abort_button.setObjectName("AbortButton")
        self.spo_abort_button.setEnabled(False)

        spo_button_layout = QtWidgets.QHBoxLayout()
        spo_button_layout.setSpacing(12)
        spo_button_layout.addWidget(self.spo_start_button)
        spo_button_layout.addWidget(self.spo_abort_button)

        # Combined-mode buttons
        self.combined_run_button = QtWidgets.QPushButton("Run JV + SPO")
        self.combined_run_button.setObjectName("QueueButton")
        self.combined_run_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.combined_abort_button = QtWidgets.QPushButton("Abort")
        self.combined_abort_button.setObjectName("AbortButton")
        self.combined_abort_button.setEnabled(False)
        self.combined_abort_button.setCursor(QtCore.Qt.PointingHandCursor)

        combined_button_layout = QtWidgets.QHBoxLayout()
        combined_button_layout.setSpacing(12)
        combined_button_layout.addWidget(self.combined_run_button)
        combined_button_layout.addWidget(self.combined_abort_button)

        # ---- Button stack: swap visibility based on active sidebar tab ----
        self._button_stack = TightStackedWidget()
        combined_btn_wrapper = QtWidgets.QWidget()
        combined_btn_wrapper.setLayout(combined_button_layout)
        self._button_stack.addWidget(combined_btn_wrapper)  # page 0: combined
        advanced_btn_wrapper = QtWidgets.QWidget()
        adv_btn_stack = QtWidgets.QVBoxLayout(advanced_btn_wrapper)
        adv_btn_stack.setContentsMargins(0, 0, 0, 0)
        adv_btn_stack.setSpacing(10)
        adv_btn_stack.addLayout(button_layout)
        adv_btn_stack.addLayout(spo_button_layout)
        self._button_stack.addWidget(advanced_btn_wrapper)  # page 1: advanced

        # ---- Instrument status lights --------------------------------------
        lights_row = self._create_status_lights()

        # ---- Left sidebar -------------------------------------------------
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setMinimumWidth(400)

        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        sidebar_layout.addLayout(mode_bar)
        sidebar_layout.addWidget(self.sidebar_mode_tabs)
        sidebar_layout.setStretchFactor(self.sidebar_mode_tabs, 0)
        sidebar_layout.addWidget(self.file_panel)
        sidebar_layout.addLayout(lights_row)
        sidebar_layout.addWidget(self._button_stack)
        sidebar_layout.addStretch(stretch=1)

        # Start in combined mode (tab 0), hide advanced-specific buttons
        self._button_stack.setCurrentIndex(0)
        self.spo_start_button.hide()
        self.spo_abort_button.hide()

        # Wrap in a scroll area for small screens
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(sidebar_widget)

        sidebar_dock = QtWidgets.QDockWidget("Inputs")
        sidebar_dock.setWidget(scroll_area)
        sidebar_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sidebar_dock)

        # ---- Main display area stack --------------------------------------
        # Page 0: Side-by-side combined view (JV + SPO)
        # Page 1: Legacy view (vertical_splitter + spo_widget overlay)
        #
        # There is exactly ONE bottom section (Experiment Queue browser +
        # Channel Analysis panel). It is SHARED between both views and
        # reparented on view switch — the same pattern used for plot_widget.
        # Creating a second instance per view is a bug: the PyMeasure Manager
        # and the controller bind to self.browser_widget/self.analysis_panel,
        # so whichever view holds the *other* instance shows nothing.
        self.bottom_section = self._create_bottom_section()
        combined_display = self._create_combined_display()
        plot_container = self._create_plot_container()
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.addWidget(plot_container)
        # bottom_section starts inside the combined view (page 0 is shown
        # first); _on_mode_button_click moves it into this splitter when the
        # user switches to the Advanced view.
        self.vertical_splitter.setStretchFactor(0, 2)
        self.vertical_splitter.setStretchFactor(1, 1)

        # Legacy display wrapper (holds vertical_splitter + spo_widget overlay)
        legacy_display = QtWidgets.QWidget()
        legacy_layout = QtWidgets.QVBoxLayout(legacy_display)
        legacy_layout.setContentsMargins(0, 0, 0, 0)
        legacy_layout.addWidget(self.vertical_splitter)

        # SPO view: replaces the JV display area entirely while active.
        if SPO_AVAILABLE:
            self.spo_widget = SpoWidget(self, self.spo_param_tab)
        else:
            self.spo_widget = QtWidgets.QWidget()
            self.spo_mode_button.setEnabled(False)
            self.spo_mode_button.setToolTip("SPO module is not installed.")
        # Wrap the SPO view in the same Plot|Log frame as every other graph
        # area, so Advanced→SPO has a log page too — it previously had none,
        # because this widget replaces the whole JV display when it is shown.
        self.spo_display = QtWidgets.QWidget()
        spo_display_layout = QtWidgets.QVBoxLayout(self.spo_display)
        spo_display_layout.setContentsMargins(0, 0, 0, 0)

        self.spo_log_panel = LogPanel()
        self.spo_graph_tab_bar, spo_tab_row = self._make_plot_log_switch(
            self.spo_log_panel, self._on_spo_graph_tab_changed)
        spo_display_layout.addLayout(spo_tab_row)

        self.spo_graph_stack = QtWidgets.QStackedWidget()
        self.spo_graph_stack.addWidget(self.spo_widget)      # page 0
        self.spo_graph_stack.addWidget(self.spo_log_panel)   # page 1
        spo_display_layout.addWidget(self.spo_graph_stack, stretch=1)

        self.spo_display.hide()
        legacy_layout.addWidget(self.spo_display)

        self._main_display_stack = QtWidgets.QStackedWidget()
        self._main_display_stack.addWidget(combined_display)   # page 0: combined
        self._main_display_stack.addWidget(legacy_display)      # page 1: legacy

        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._main_display_stack)

        # ----- Top‑right user info + logout button -----
        logout_toolbar = QtWidgets.QToolBar("Logout")
        logout_toolbar.setMovable(False)
        logout_toolbar.setContentsMargins(12, 12, 12, 0)   # left, top, right, bottom

        # Spacer that pushes the user info + button to the right edge
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        logout_toolbar.addWidget(spacer)

        # Logged-in username pill, shown to the left of the Logout button
        self.user_label = QtWidgets.QLabel(f"👤 {self.username or 'Guest'}")
        self.user_label.setStyleSheet("""
            QLabel {
                background-color: #f1f5f9;
                color: #334155;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                margin-right: 8px;
            }
        """)
        logout_toolbar.addWidget(self.user_label)

        self.logout_btn = QtWidgets.QPushButton("Logout")
        self.logout_btn.setToolTip("Logout and return to login screen")
        self.logout_btn.setCursor(QtCore.Qt.PointingHandCursor)
        # Updated to match the new flat UI aesthetic
        self.logout_btn.setStyleSheet("""
            QPushButton {
                background-color: #fee2e2;
                color: #ef4444;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #fca5a5;
            }
        """)
        self.logout_btn.clicked.connect(self._confirm_logout)

        logout_toolbar.addWidget(self.logout_btn)
        self.addToolBar(QtCore.Qt.TopToolBarArea, logout_toolbar)

        self.update_instrument_lights()

        # Wire debug checkbox for both analysis tabs (created above)
        self.combined_analysis_tab.connect_debug_signal(self.toggle_debug_logging)
        self.analysis_settings_tab.connect_debug_signal(self.toggle_debug_logging)

    @property
    def active_instr_tab(self):
        """Return the Instrument tab for the currently active sidebar tab."""
        if self.sidebar_mode_tabs.currentIndex() == 0:
            return self.combined_instr_tab
        return self.instr_tab

    @property
    def active_analysis_tab(self):
        """Return the Analysis tab for the currently active sidebar tab."""
        if self.sidebar_mode_tabs.currentIndex() == 0:
            return self.combined_analysis_tab
        return self.analysis_settings_tab

    def _create_status_lights(self):
        """Create instrument connection status indicators."""
        lights_row = QtWidgets.QHBoxLayout()
        lights_row.setSpacing(8) # Unified spacing

        self.keithley_light = QtWidgets.QLabel("  ")
        self.keithley_light.setFixedSize(12, 12)
        self.keithley_light.setStyleSheet("border-radius:6px; background:#ef4444;")

        self.mux_light = QtWidgets.QLabel("  ")
        self.mux_light.setFixedSize(12, 12)
        self.mux_light.setStyleSheet("border-radius:6px; background:#ef4444;")

        # These are indicators, not controls: green means connected, red means
        # not. Everything behind that — polling, re-opening a port, deciding
        # whether the adapter is still plugged in — is the monitor's business
        # and stays out of the operator's way. The detail goes to the session
        # log, not to the screen.

        lights_row.addStretch(1)
        lights_row.addWidget(self.keithley_light)
        lights_row.addWidget(QtWidgets.QLabel("Keithley"))
        lights_row.addSpacing(20)
        lights_row.addWidget(self.mux_light)
        lights_row.addWidget(QtWidgets.QLabel("MUX"))
        lights_row.addStretch(1)

        return lights_row

    def _make_plot_log_switch(self, log_panel, on_change):
        """The centred `Plot | Log` pill bar used above every graph area.

        Every view gets the SAME control in the same place — the JV+SPO
        graphs, the Advanced JV plot and the Advanced SPO plot — so switching
        view never changes where the log lives or what it looks like.
        """
        bar = QtWidgets.QTabBar()
        bar.setObjectName("GraphTabBar")
        bar.setExpanding(False)
        bar.setDrawBase(False)
        bar.setCursor(QtCore.Qt.PointingHandCursor)
        bar.addTab("Plot")
        bar.addTab("Log")
        bar.currentChanged.connect(on_change)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.addStretch(1)
        row.addWidget(bar)
        row.addStretch(1)

        log_panel.attention_changed.connect(
            lambda active, b=bar: self._mark_log_tab(b, active))
        return bar, row

    @staticmethod
    def _mark_log_tab(bar, active):
        """A warning or error arrived while the log was hidden — say so on the
        tab instead of making the operator go looking."""
        bar.setTabText(1, "Log  •" if active else "Log")
        from solarjv_analyzer.gui.theme import tokens
        bar.setTabTextColor(
            1, QtGui.QColor(tokens.LOG_ERROR) if active else QtGui.QColor())

    def _create_plot_container(self):
        """The Advanced (JV) plot area: the same Plot|Log switch as the
        JV+SPO view, plus the Save Plot button.

        Note: plot_widget is NOT added here — it is reparented on view switch
        by _on_mode_button_click() to avoid Qt parent conflicts.
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 8, 12, 12)

        self.jv_log_panel = LogPanel()
        self.jv_graph_tab_bar, tab_row = self._make_plot_log_switch(
            self.jv_log_panel, self._on_jv_graph_tab_changed)
        layout.addLayout(tab_row)

        # plot_widget is inserted at index 0 of this page by
        # _on_mode_button_click(); the page exists so the log can take its
        # place without disturbing the Save button or the queue below.
        self.jv_plot_page = QtWidgets.QWidget()
        self._jv_plot_layout = QtWidgets.QVBoxLayout(self.jv_plot_page)
        self._jv_plot_layout.setContentsMargins(0, 0, 0, 0)

        self.jv_graph_stack = QtWidgets.QStackedWidget()
        self.jv_graph_stack.addWidget(self.jv_plot_page)     # page 0
        self.jv_graph_stack.addWidget(self.jv_log_panel)     # page 1
        layout.addWidget(self.jv_graph_stack, stretch=1)

        self.save_plot_button = QtWidgets.QPushButton("Save Plot as PNG")
        self.save_plot_button.setObjectName("SavePlotButton")
        layout.addWidget(self.save_plot_button)

        return container

    def _log_panels(self):
        """Every live log page: JV+SPO, Advanced/JV, Advanced/SPO."""
        return [panel for panel in (getattr(self, name, None) for name in
                                    ("log_panel", "jv_log_panel", "spo_log_panel"))
                if panel is not None]

    def _on_spo_graph_tab_changed(self, index: int):
        self.spo_graph_stack.setCurrentIndex(index)
        if index == 1:
            self.spo_log_panel.mark_seen()

    def _on_jv_graph_tab_changed(self, index: int):
        self.jv_graph_stack.setCurrentIndex(index)
        self.save_plot_button.setVisible(index == 0)
        if index == 1:
            self.jv_log_panel.mark_seen()

    def _create_combined_display(self):
        """Build the side-by-side JV + SPO graph area for the combined tab.

        Both graph panels have identical chrome (heading + bare PlotWidget)
        so they render at exactly the same size. A single "Save Both Plots"
        button is centred below the pair.
        """
        from .widgets.analysis_panel import AnalysisPanel

        container = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Side-by-side graphs (equal-sized panels) --------------------
        h_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Both panels are built with IDENTICAL chrome so the two plot
        # canvases render at exactly the same size: a fixed-height heading
        # row on top, canvas below (stretch=1), nothing else. The SPO live
        # metrics live IN the heading row (right-aligned chips), and the
        # pymeasure plot widget's internal chrome (axis selector row +
        # coordinates label) is hidden while in the combined view — see
        # _set_plot_widget_chrome_visible(). Both left axes are pinned to
        # the same fixed width so the canvases also align horizontally
        # ("Current (A)" and "Power (mW)" tick labels differ in width).
        HEADING_H = 30

        def _heading(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(
                "font-weight: 600; font-size: 13px; color: #1a1c1d;"
                " background: transparent;"
            )
            return lbl

        # -- Left: JV I-V Curve -------------------------------------------
        jv_panel = QtWidgets.QWidget()
        self._combined_jv_layout = QtWidgets.QVBoxLayout(jv_panel)
        self._combined_jv_layout.setContentsMargins(12, 8, 6, 8)
        self._combined_jv_layout.setSpacing(4)

        jv_heading_row = QtWidgets.QWidget()
        jv_heading_row.setFixedHeight(HEADING_H)
        jv_heading_layout = QtWidgets.QHBoxLayout(jv_heading_row)
        jv_heading_layout.setContentsMargins(0, 0, 0, 0)
        jv_heading_layout.addWidget(_heading("JV I‑V Curve"))
        jv_heading_layout.addStretch()
        self._combined_jv_layout.addWidget(jv_heading_row)
        # plot_widget is inserted at index 1 by _on_mode_button_click()
        # so it lands after the heading. We add it here initially too so it
        # is visible on first paint.
        self._combined_jv_layout.addWidget(self.plot_widget, stretch=1)
        self._set_plot_widget_chrome_visible(False)

        h_splitter.addWidget(jv_panel)

        # -- Right: SPO Power vs Time -------------------------------------
        spo_panel = QtWidgets.QWidget()
        spo_layout = QtWidgets.QVBoxLayout(spo_panel)
        spo_layout.setContentsMargins(6, 8, 12, 8)
        spo_layout.setSpacing(4)

        spo_heading_row = QtWidgets.QWidget()
        spo_heading_row.setFixedHeight(HEADING_H)
        spo_heading_layout = QtWidgets.QHBoxLayout(spo_heading_row)
        spo_heading_layout.setContentsMargins(0, 0, 0, 0)
        spo_heading_layout.setSpacing(10)
        spo_heading_layout.addWidget(_heading("SPO Power vs Time"))
        spo_heading_layout.addStretch()

        # Live metrics as compact chips inside the heading row (keeps both
        # panels' vertical structure identical).
        #
        # Channel and Hold come first: they say what this SPO phase is doing —
        # which channel the JV phase picked as best, and the voltage it is
        # being held at — which the operator cannot otherwise tell in combined
        # mode, since the app chooses the channel rather than the user. The
        # hold voltage is shown signed, because Vmpp is signed and a p-i-n
        # cell legitimately holds negative.
        self.combined_spo_channel = ShrinkableLabel("—")
        self.combined_spo_hold = ShrinkableLabel("— mV")
        self.combined_spo_mean = ShrinkableLabel("— mW")
        self.combined_spo_drift = ShrinkableLabel("— %")
        self.combined_spo_elapsed = ShrinkableLabel("— s")
        for caption, lbl in (("Channel", self.combined_spo_channel),
                             ("Hold", self.combined_spo_hold),
                             ("Mean", self.combined_spo_mean),
                             ("Drift", self.combined_spo_drift),
                             ("Elapsed", self.combined_spo_elapsed)):
            cap = ShrinkableLabel(f"{caption}:")
            cap.setStyleSheet(
                "font-size: 11px; color: #71787b; background: transparent;"
            )
            lbl.setStyleSheet(
                "font-size: 11px; font-weight: 600; color: #41484b;"
                " background: transparent;"
            )
            # The chips must not dictate the panel's minimum width. A plain
            # QLabel reports its full text width as minimumSizeHint(), which
            # inflates the SPO panel's minimum and pushes the splitter off
            # 50/50. ShrinkableLabel (above) reports a zero-width minimum, and
            # the horizontal "Maximum" policy caps the preferred width at the
            # text width — so the chips render in full when there is room and
            # shrink gracefully (rather than clipping) when the splitter needs
            # the space. ("Ignored" is NOT usable: its sizeHint becomes zero,
            # so the trailing addStretch() collapses the chips to zero width
            # and hides the text entirely.)
            for chip in (cap, lbl):
                chip.setSizePolicy(
                    QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred
                )
            spo_heading_layout.addWidget(cap)
            spo_heading_layout.addWidget(lbl)
        spo_layout.addWidget(spo_heading_row)

        self.combined_spo_plot = pg.PlotWidget()
        self.combined_spo_plot.setBackground('#ffffff')
        self.combined_spo_plot.setLabel('bottom', 'Time', units='s')
        self.combined_spo_plot.setLabel('left', 'Power', units='mW')
        self.combined_spo_curve = self.combined_spo_plot.plot(
            [], [], pen=pg.mkPen(color='#053a46', width=2)
        )
        spo_layout.addWidget(self.combined_spo_plot, stretch=1)
        h_splitter.addWidget(spo_panel)

        # Identical axis geometry on both canvases.
        from solarjv_analyzer.gui.theme import style_plot
        style_plot(self.plot_widget.plot)
        style_plot(self.combined_spo_plot.getPlotItem())

        h_splitter.setSizes([500, 500])  # force equal initial widths
        h_splitter.setStretchFactor(0, 1)
        h_splitter.setStretchFactor(1, 1)  # keep growth symmetric on resize

        # ---- Plot | Log switch, centred above the graphs -----------------
        # The graphs and the log occupy the same space rather than sharing it:
        # both want the full width, and an operator is reading one or the
        # other, never both.
        self.graph_tab_bar, tab_row = self._make_plot_log_switch(
            self.log_panel, self._on_graph_tab_changed)
        outer.addLayout(tab_row)

        self.graph_stack = QtWidgets.QStackedWidget()
        self.graph_stack.addWidget(h_splitter)        # page 0: the two plots
        self.graph_stack.addWidget(self.log_panel)    # page 1: live log
        outer.addWidget(self.graph_stack, stretch=2)

        # ---- Save button (centred, just above the results section) --------
        self.combined_save_button = QtWidgets.QPushButton("Save Both Plots as PNG")
        self.combined_save_button.setObjectName("SavePlotButton")
        self.combined_save_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.combined_save_button.setMaximumWidth(300)

        save_row = QtWidgets.QHBoxLayout()
        save_row.setContentsMargins(0, 8, 0, 4)
        save_row.addStretch()
        save_row.addWidget(self.combined_save_button)
        save_row.addStretch()
        self._combined_save_row = QtWidgets.QWidget()
        self._combined_save_row.setLayout(save_row)
        outer.addWidget(self._combined_save_row)

        # ---- Bottom section (Experiment Queue / Channel Analysis) ---------
        # Embed the SHARED bottom section (created once in _init_ui before
        # this method runs). Do NOT create a second instance here — see the
        # comment at the _create_bottom_section() call site.
        self._combined_outer_layout = outer
        outer.addWidget(self.bottom_section, stretch=1)
        return container

    def _on_address_edited(self, resource: str):
        """Apply a typed Keithley address for this session.

        Refused while anything is running: the address decides which VISA
        session `connect_keithley()` opens, and swapping it mid-sweep would
        leave the running worker talking to a handle the manager no longer
        owns.
        """
        from solarjv_analyzer.instruments import port_resolver

        busy = bool(getattr(self.controller, "is_busy", False)
                    or getattr(self.controller, "spo_running", False))
        if busy:
            QtWidgets.QMessageBox.information(
                self, "Measurement running",
                "The instrument address cannot be changed while a measurement "
                "is running. Stop the run and try again.")
            self._sync_address_tabs(port_resolver.active_keithley_resource())
            return

        resource = (resource or "").strip()
        if resource and not _looks_like_visa_resource(resource):
            QtWidgets.QMessageBox.warning(
                self, "Not a VISA resource",
                f"'{resource}' is not a VISA resource address.\n\n"
                "Expected something like ASRL3::INSTR, GPIB0::24::INSTR or "
                "USB0::0x05E6::0x2400::...::INSTR.")
            self._sync_address_tabs(port_resolver.active_keithley_resource())
            return

        applied = port_resolver.set_session_keithley_resource(resource)
        # Drop the current session so the next connect uses the new address.
        try:
            self.instrument_manager.disconnect_keithley()
            self.instrument_manager.connect_keithley(simulation=False)
            logger.info(f"Reconnected the Keithley at {applied}")
        except Exception as exc:
            logger.warning(f"Could not connect at {applied}: {exc}")
        self._sync_address_tabs(applied)
        self.update_instrument_lights()

    def _sync_address_tabs(self, resource: str):
        """Keep both Instrument tabs showing the address actually in force."""
        for tab in (self.instr_tab, self.combined_instr_tab):
            tab.set_address(resource)

    def _on_graph_tab_changed(self, index: int):
        """Swap the graph area between the plots (0) and the live log (1)."""
        self.graph_stack.setCurrentIndex(index)
        # "Save Both Plots as PNG" belongs to the plots, not the log.
        self._combined_save_row.setVisible(index == 0)
        if index == 1:
            self.log_panel.mark_seen()

    def _create_bottom_section(self):
        """Build the bottom dashboard: centred pill tab bar, persistent action
        bar, and a QStackedWidget that swaps between the experiment-queue
        browser and the channel-analysis matrix."""
        display_parameters = [
            "active_channel",
            "compliance_current",
            "delay_between_points",
            "device_area",
            "incident_power",
            "lateral_factor",
            "pre_sweep_delay",
            "sense_mode",
            "start_voltage",
            "step_size",
            "stop_voltage",
            "user_name"
        ]

        self.browser_widget = BrowserWidget(
            JVProcedure,
            display_parameters,
            JVProcedure.DATA_COLUMNS
        )

        # Hide BrowserWidget's built-in action buttons — the persistent
        # action bar handles Show / Hide / Clear / Open for both views.
        self.browser_widget.show_button.hide()
        self.browser_widget.hide_button.hide()
        self.browser_widget.clear_button.hide()
        self.browser_widget.open_button.hide()
        # Strip internal margins so the browser view sits flush against
        # the action bar — matching the zero-margin analysis panel.
        if self.browser_widget.layout():
            self.browser_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.analysis_panel = AnalysisPanel(self)

        # ---- Pill Tab Bar (centred via stylesheet) ---------------------------
        self.bottom_tab_bar = QtWidgets.QTabBar()
        self.bottom_tab_bar.setObjectName("BottomTabBar")
        self.bottom_tab_bar.setExpanding(False)
        self.bottom_tab_bar.setDrawBase(False)
        self.bottom_tab_bar.addTab("Experiment Queue")
        self.bottom_tab_bar.addTab("Channel Analysis")

        # ---- Persistent Action Bar -------------------------------------------
        from solarjv_analyzer.gui.style import SPACING_SM, SPACING_MD

        self.action_show_all = QtWidgets.QPushButton("Show all")
        self.action_hide_all = QtWidgets.QPushButton("Hide all")
        self.action_clear_all = QtWidgets.QPushButton("Clear all")
        self.action_open = QtWidgets.QPushButton("Open")

        for btn in (self.action_show_all, self.action_hide_all,
                     self.action_clear_all, self.action_open):
            btn.setCursor(QtCore.Qt.PointingHandCursor)

        # ---- Channel colour indicators (populated dynamically) ----------------
        self._channel_indicators = []  # list[QtWidgets.QLabel]
        self._channel_indicators_layout = QtWidgets.QHBoxLayout()
        self._channel_indicators_layout.setSpacing(SPACING_SM)
        self._channel_indicators_container = QtWidgets.QWidget()
        self._channel_indicators_container.setLayout(
            self._channel_indicators_layout
        )

        # ---- 3-section action bar for absolute centering ---------------------
        # Left section  (stretch 1) — action buttons, content pinned left
        left_section = QtWidgets.QHBoxLayout()
        left_section.setSpacing(SPACING_MD)
        left_section.addWidget(self.action_show_all)
        left_section.addWidget(self.action_hide_all)
        left_section.addWidget(self.action_clear_all)
        left_section.addStretch()

        # Centre section (stretch 0) — channel badges + architecture badge
        centre_section = QtWidgets.QHBoxLayout()
        centre_section.setSpacing(SPACING_SM)
        centre_section.addWidget(self._channel_indicators_container)

        self._architecture_badge = QtWidgets.QLabel("n-i-p")
        self._architecture_badge.setStyleSheet(
            "background-color: rgba(255,255,255,0.08);"
            " color: #94a3b8; border: 1px solid rgba(148,163,184,0.2);"
            " border-radius: 4px; padding: 2px 8px;"
            " font-size: 11px; font-weight: 600;"
        )
        self._architecture_badge.hide()
        centre_section.addWidget(self._architecture_badge)

        # Right section  (stretch 1) — Open button, content pinned right
        right_section = QtWidgets.QHBoxLayout()
        right_section.setSpacing(SPACING_MD)
        right_section.addStretch()
        right_section.addWidget(self.action_open)

        action_bar = QtWidgets.QHBoxLayout()
        action_bar.setContentsMargins(0, SPACING_SM, 0, SPACING_SM)
        action_bar.addLayout(left_section, stretch=1)
        action_bar.addLayout(centre_section, stretch=0)
        action_bar.addLayout(right_section, stretch=1)

        # ---- Content Stack (tab bar drives this) -----------------------------
        self.bottom_stack = QtWidgets.QStackedWidget()
        self.bottom_stack.addWidget(self.browser_widget)   # page 0
        self.bottom_stack.addWidget(self.analysis_panel)    # page 1

        self.bottom_tab_bar.currentChanged.connect(
            self.bottom_stack.setCurrentIndex
        )

        # ---- Wrap ------------------------------------------------------------
        from solarjv_analyzer.gui.style import SPACING_MD

        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(SPACING_MD, 0, SPACING_MD, SPACING_MD)
        layout.setSpacing(0)
        layout.addWidget(self.bottom_tab_bar)
        layout.addLayout(action_bar)
        layout.addWidget(self.bottom_stack, 1)

        return wrapper

    def _create_file_panel(self):
        """Create the file output panel with directory from manager."""
        self.file_panel = FilePanel()

        self.dir_manager.set_mode("Main")
        self.dir_manager.set_username(self.username)

        # Set the directory to the full Main path
        main_dir = self.dir_manager.get_current_directory(create=True)
        self.file_panel.set_directory(main_dir)
        return self.file_panel

    # -------------------------------------------------------------------------
    # Logout
    # -------------------------------------------------------------------------

    def _confirm_logout(self):
        """Ask the user to confirm logout, then emit logged_out if accepted."""
        # Guard running SPO
        if getattr(self.controller, 'spo_running', False):
            if not self._confirm_abort_spo(
                "An SPO measurement is currently running.\n\n"
                "Logging out will abort it and save the partial data. Continue?"
            ):
                return

        # Guard running JV queue or Quick JV
        jv_busy = getattr(self.controller, 'is_busy', False) and not getattr(
            self.controller, 'spo_running', False
        )
        quick_jv_running = (
            hasattr(self, 'spo_param_tab')
            and getattr(self.spo_param_tab, '_quick_jv_running', False)
        )
        if jv_busy or quick_jv_running:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Measurement In Progress",
                "A measurement is currently running.\n\n"
                "Logging out will abort it and may result in partial or "
                "corrupted data. Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            # Abort the running measurement before logout
            if jv_busy:
                self.controller.manager.abort()
            if quick_jv_running and hasattr(self.spo_param_tab, '_quick_jv_worker'):
                worker = self.spo_param_tab._quick_jv_worker
                if worker is not None:
                    worker.abort()

        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Logout",
            "Are you sure you want to log out?\n\n"
            "This will disconnect instruments and return to the login screen.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.logged_out.emit()

    def closeEvent(self, event):
        """Handle window close (X button): safely abort any running
        measurement, turn off Keithley output, and disconnect instruments
        before the process exits."""
        logger.info("Window close requested — performing safety shutdown.")

        # Stop feeding the log page first. A worker thread that logs during the
        # shutdown below would otherwise emit into a half-destroyed widget
        # ("wrapped C/C++ object ... has been deleted").
        for panel in self._log_panels():
            panel.detach()

        # Let the next window (relogin) build its plots against a clean
        # ViewBox registry rather than one full of this window's corpses.
        try:
            from solarjv_analyzer.gui.theme import forget_dead_viewboxes
            forget_dead_viewboxes()
        except Exception:
            pass

        # 1. Abort any running SPO — and JOIN the worker thread before
        # touching the instruments below. Without wait() the thread can still
        # be mid-:READ? when the VISA session is closed underneath it
        # ("QThread: Destroyed while thread is still running" crash).
        if getattr(self.controller, 'spo_running', False):
            logger.info("Aborting SPO run before close...")
            self.controller.abort_spo()
            spo_worker = getattr(self.controller, '_spo_worker', None)
            if spo_worker is not None:
                spo_worker.wait(15000)

        # 2. Abort any running JV queue
        if getattr(self.controller, 'is_busy', False):
            logger.info("Aborting JV queue before close...")
            try:
                self.controller.manager.abort()
            except Exception:
                pass

        # 3. Abort any running Quick JV — join it too (same teardown race).
        if (hasattr(self, 'spo_param_tab')
                and getattr(self.spo_param_tab, '_quick_jv_running', False)):
            logger.info("Aborting Quick JV before close...")
            worker = getattr(self.spo_param_tab, '_quick_jv_worker', None)
            if worker is not None:
                worker.abort()
                worker.wait(10000)

        # 3b. Stop the hardware monitor and join its worker — a QThread
        #     destroyed while still running takes the process with it.
        hardware_timer = getattr(self, '_hardware_timer', None)
        if hardware_timer is not None:
            hardware_timer.stop()
        hardware_thread = getattr(self, '_hardware_thread', None)
        if hardware_thread is not None and hardware_thread.isRunning():
            hardware_thread.wait(5000)

        # 4. Turn off Keithley output and disconnect instruments
        try:
            if self.instrument_manager.is_keithley_alive():
                self.instrument_manager.keithley.write(":OUTP OFF")
                self.instrument_manager.keithley.write(":ABOR")
        except Exception:
            pass
        try:
            self.instrument_manager.disconnect_keithley()
            self.instrument_manager.disconnect_mux()
        except Exception:
            pass

        # 5. End the session
        try:
            from solarjv_analyzer.auth.session import SessionManager
            SessionManager().end_session()
        except Exception:
            pass

        event.accept()

    # -------------------------------------------------------------------------
    # Sidebar Tab Switching (JV + SPO / Advanced)
    # -------------------------------------------------------------------------

    def _set_plot_widget_chrome_visible(self, visible: bool):
        """Show/hide the pymeasure PlotWidget's internal chrome (X/Y axis
        selector row, coordinates label, layout margins). Hidden in the
        combined view so the JV canvas has exactly the same chrome as the
        bare SPO canvas — this is what keeps the two side-by-side graphs
        the same size. Everything is restored for the Advanced view.

        The pymeasure PlotWidget has THREE layers of chrome that all must be
        collapsed for the canvas to truly match the bare SPO PlotWidget:
          1. the axis-selector row (columns_x/columns_y) — hiding its widgets
             leaves the row's 6px top/bottom margins behind (12px gap);
          2. the PlotFrame's QFrame border (StyledPanel/Sunken, 1px each
             side) — a 2px inset that no margin reset removes;
          3. the outer + frame layout margins.
        All three are captured once and zeroed in the combined view."""
        pw = self.plot_widget
        for attr in ("columns_x", "columns_y", "columns_x_label", "columns_y_label"):
            w = getattr(pw, attr, None)
            if w is not None:
                w.setVisible(visible)

        plot_frame = getattr(pw, "plot_frame", None)
        coords = getattr(plot_frame, "coordinates", None)
        if coords is not None:
            coords.setVisible(visible)

        # Capture the original geometry once so it can be restored later.
        if not hasattr(self, "_pw_orig_chrome"):
            self._pw_orig_chrome = {}
            for name, w in (("outer", pw), ("frame", plot_frame)):
                lay = w.layout() if w is not None else None
                if lay is not None:
                    self._pw_orig_chrome[name] = (lay.getContentsMargins(),
                                                  lay.spacing())
            # The selector row is the first item of the outer layout.
            outer_lay = pw.layout()
            selector = (outer_lay.itemAt(0).layout()
                        if outer_lay is not None and outer_lay.count()
                        else None)
            if selector is not None:
                self._pw_orig_chrome["selector_margins"] = selector.getContentsMargins()
            if plot_frame is not None:
                self._pw_orig_chrome["frame_shape"] = plot_frame.frameShape()
                self._pw_orig_chrome["frame_line_width"] = plot_frame.lineWidth()
                self._pw_orig_chrome["frame_mid_width"] = plot_frame.midLineWidth()

        for name, w in (("outer", pw), ("frame", plot_frame)):
            lay = w.layout() if w is not None else None
            if lay is None or name not in self._pw_orig_chrome:
                continue
            if visible:
                margins, spacing = self._pw_orig_chrome[name]
                lay.setContentsMargins(*margins)
                lay.setSpacing(spacing)
            else:
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(0)

        # Collapse the selector row's vertical margins and the frame border.
        outer_lay = pw.layout()
        selector = (outer_lay.itemAt(0).layout()
                    if outer_lay is not None and outer_lay.count()
                    else None)
        if selector is not None and "selector_margins" in self._pw_orig_chrome:
            if visible:
                selector.setContentsMargins(*self._pw_orig_chrome["selector_margins"])
            else:
                selector.setContentsMargins(0, 0, 0, 0)

        if plot_frame is not None and "frame_shape" in self._pw_orig_chrome:
            if visible:
                plot_frame.setFrameShape(self._pw_orig_chrome["frame_shape"])
                plot_frame.setLineWidth(self._pw_orig_chrome["frame_line_width"])
                plot_frame.setMidLineWidth(self._pw_orig_chrome["frame_mid_width"])
            else:
                plot_frame.setFrameShape(QtWidgets.QFrame.NoFrame)
                plot_frame.setLineWidth(0)
                plot_frame.setMidLineWidth(0)

    def _on_mode_button_click(self, index: int):
        """Called when JV+SPO (0) or Advanced (1) mode button is clicked.

        Swaps the sidebar content stack, main display, reparents the shared
        plot_widget, and toggles button checked states. Forces layout
        recalculation so the sidebar tightens to the new page's content."""
        self.sidebar_mode_tabs.setCurrentIndex(index)
        self._mode_btn_jvspo.setChecked(index == 0)
        self._mode_btn_adv.setChecked(index == 1)

        if index == 0:
            self._button_stack.setCurrentIndex(0)
            self._main_display_stack.setCurrentIndex(0)
            self._combined_jv_layout.insertWidget(1, self.plot_widget)
            self.plot_widget.show()
            self._set_plot_widget_chrome_visible(False)
            # Reparent the shared bottom section (queue browser + analysis
            # panel) into the combined view — same pattern as plot_widget.
            self._combined_outer_layout.addWidget(self.bottom_section, stretch=1)
            self.bottom_section.show()
            self.spo_display.hide()
            self._update_save_directory()
            self._set_file_panel_spo_mode(False)
            self._on_filename_changed(self.file_panel.filename_input.text())
        else:
            self._button_stack.setCurrentIndex(1)
            self._main_display_stack.setCurrentIndex(1)
            self._jv_plot_layout.addWidget(self.plot_widget)
            self.jv_graph_tab_bar.setCurrentIndex(0)
            self.jv_graph_stack.setCurrentIndex(0)
            self.plot_widget.show()
            self._set_plot_widget_chrome_visible(True)
            # Reparent the shared bottom section into the Advanced splitter.
            self.vertical_splitter.addWidget(self.bottom_section)
            self.vertical_splitter.setStretchFactor(
                self.vertical_splitter.indexOf(self.bottom_section), 1
            )
            self.bottom_section.show()
            if self.jv_mode_button.isChecked():
                self._show_jv_mode()
            else:
                self._show_spo_mode()

        # Force the entire sidebar to recalculate heights based on the
        # newly visible page. Without this, Qt keeps the old page's geometry.
        self._relayout_sidebar()

    def _relayout_sidebar(self):
        """Force the entire sidebar layout chain to recalculate heights
        after any inner page switch (mode, JV/SPO, or inner tab change).

        Walks up from every Tight*Widget, invalidating layouts so dead
        space is never left behind when switching to shorter content."""
        from_widgets = [self.sidebar_mode_tabs, self.params_stack,
                        self._button_stack, self._main_display_stack]
        for start in from_widgets:
            w = start
            while w is not None and not isinstance(w, QtWidgets.QScrollArea):
                w.updateGeometry()
                if w.layout():
                    w.layout().invalidate()
                w = w.parent()
        sa = self.findChild(QtWidgets.QScrollArea)
        if sa and sa.widget():
            sa.widget().updateGeometry()
            if sa.widget().layout():
                sa.widget().layout().invalidate()
                sa.widget().layout().activate()

    # -------------------------------------------------------------------------
    # Mode Switching (JV Sweep <-> SPO)
    # -------------------------------------------------------------------------

    def _confirm_abort_spo(self, message: str) -> bool:
        """Ask for confirmation, then abort the running SPO test if accepted."""
        reply = QtWidgets.QMessageBox.question(
            self, "SPO Running", message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return False
        self.controller.abort_spo()
        return True

    def _jv_is_busy(self) -> bool:
        """True if a JV experiment is currently queued or running."""
        return bool(getattr(self.controller, 'is_busy', False)) and not getattr(
            self.controller, 'spo_running', False
        )

    def _on_mode_button_clicked(self, button):
        """Handle JV Sweep / SPO mode toggle button clicks.

        Per design: mode switching is simply BLOCKED (with a warning) while
        either an SPO measurement is running, or a JV measurement is
        queued/running — the switch is cancelled and the previous mode
        button state is restored.
        """
        switching_to_spo = button is self.spo_mode_button

        if getattr(self.controller, 'spo_running', False):
            QtWidgets.QMessageBox.warning(
                self, "SPO Running",
                "An SPO measurement is currently running.\n\n"
                "Please abort it before switching modes."
            )
            self._revert_mode_button(switching_to_spo)
            return

        if self._jv_is_busy():
            QtWidgets.QMessageBox.warning(
                self, "JV Measurement Active",
                "A JV measurement is currently queued or running.\n\n"
                "Please wait for it to finish, or abort it, before switching modes."
            )
            self._revert_mode_button(switching_to_spo)
            return

        if switching_to_spo:
            self._show_spo_mode()
        else:
            self._show_jv_mode()

    def _revert_mode_button(self, was_switching_to_spo: bool):
        """Restore the mode toggle to reflect the mode we're actually still in."""
        if was_switching_to_spo:
            self.jv_mode_button.setChecked(True)
        else:
            self.spo_mode_button.setChecked(True)

    def _show_spo_mode(self):
        """Show the SPO view and hide the JV plot/browser/analysis views."""
        self.vertical_splitter.hide()
        self.spo_display.show()
        self.params_tab.hide()
        self.spo_param_tab.show()
        self.queue_button.hide()
        self.abort_button.hide()
        self.spo_start_button.show()
        self.spo_abort_button.show()
        self._update_spo_save_directory()
        self._set_file_panel_spo_mode(True)
        self.spo_widget.set_mode_spo()
        self.spo_widget.set_filename(self.file_panel.filename_input.text())
        self._relayout_sidebar()

    def _show_jv_mode(self):
        """Show the JV plot/browser/analysis views and hide the SPO view."""
        self.spo_display.hide()
        self.vertical_splitter.show()
        self.spo_param_tab.hide()
        self.params_tab.show()
        self.spo_start_button.hide()
        self.spo_abort_button.hide()
        self.queue_button.show()
        self.abort_button.show()
        self._set_file_panel_spo_mode(False)
        self._update_save_directory()
        self._on_filename_changed(self.file_panel.filename_input.text())
        self._relayout_sidebar()

    def _set_file_panel_spo_mode(self, spo_mode: bool):
        """Adapt the File Panel for SPO mode: only the single-file checkbox
        is JV-specific and gets locked.  The filename prefix stays editable
        in both modes so validation remains active."""
        self.file_panel.single_file_checkbox.setEnabled(not spo_mode)

    def _on_spo_vmpp_ready(self, ready: bool):
        """Enable Start SPO only when filename + Vmpp are ready and SPO is idle."""
        spo_running = getattr(self.controller, 'spo_running', False)
        self.spo_start_button.setEnabled(
            self._filename_valid and ready and not spo_running
        )
        self.spo_start_button.style().unpolish(self.spo_start_button)
        self.spo_start_button.style().polish(self.spo_start_button)

    def _on_filename_changed(self, _text: str) -> None:
        """Enable / disable execution buttons based on filename validity."""
        self._filename_valid = self.file_panel.has_valid_filename()
        self.file_panel.filename_hint.setVisible(not self._filename_valid)

        # Only enable run buttons when idle — a valid filename during a
        # running measurement must not re-enable buttons mid-run.
        busy = (
            getattr(self.controller, 'is_busy', False)
            or getattr(self.controller, 'spo_running', False)
        )
        self.queue_button.setEnabled(self._filename_valid and not busy)
        self.combined_run_button.setEnabled(self._filename_valid and not busy)

        # Re-evaluate SPO button (respects its own Vmpp + idle gating)
        if hasattr(self, 'spo_widget') and hasattr(self.spo_widget, 'has_valid_hold_voltage'):
            self._on_spo_vmpp_ready(self.spo_widget.has_valid_hold_voltage())
        # Force Qt to repaint buttons so :disabled / :enabled CSS takes effect
        for btn in (self.queue_button, self.spo_start_button, self.combined_run_button):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def connect_signals(self):
        """Connect UI signals to controller methods."""
        self.queue_button.clicked.connect(self.controller.queue_experiment)
        self.abort_button.clicked.connect(self.controller.abort_experiment)
        self.save_plot_button.clicked.connect(self.save_plot)

        # Combined-mode buttons
        self.combined_run_button.clicked.connect(self.controller.start_combined_run)
        self.combined_abort_button.clicked.connect(self.controller.abort_combined)

        # Manual VISA-address override (both Instrument tabs share one target).
        for tab in (self.instr_tab, self.combined_instr_tab):
            tab.address_edited.connect(self._on_address_edited)
        self.combined_save_button.clicked.connect(self._save_combined_plots)

        # Mode switching is handled by _on_mode_button_click (wired to L1 buttons)
        self.browser_widget.show_button.clicked.connect(self.show_experiments)
        self.browser_widget.hide_button.clicked.connect(self.hide_experiments)
        self.browser_widget.clear_button.clicked.connect(self.clear_experiments)
        self.browser_widget.open_button.clicked.connect(self.open_experiment)
        self.browser_widget.browser.itemChanged.connect(self.browser_item_changed)
        self.browser_widget.browser.itemSelectionChanged.connect(
            self.controller.on_browser_selection_changed
        )

        # Persistent action bar (visible in both Experiment Queue and
        # Channel Analysis views)
        self.action_show_all.clicked.connect(self.show_experiments)
        self.action_hide_all.clicked.connect(self.hide_experiments)
        self.action_clear_all.clicked.connect(self.clear_experiments)
        self.action_open.clicked.connect(self.open_experiment)

        # SPO mode toggle and Start/Abort buttons
        self.mode_button_group.buttonClicked.connect(self._on_mode_button_clicked)
        if SPO_AVAILABLE:
            self.spo_start_button.clicked.connect(self.controller.start_spo)
            self.spo_abort_button.clicked.connect(self.controller.abort_spo)
            self.spo_widget.vmpp_ready.connect(self._on_spo_vmpp_ready)

    def _connect_nplc_preview_signals(self):
        """Connect parameter signals for NPLC preview calculation."""
        # JV parameter tab
        self.params_tab.sweep_rate.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.sweep_rate_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.start_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.start_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.stop_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.stop_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.step_size.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.step_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)

        # Combined tab (embedded ParameterTab)
        ct = self.combined_tab
        ct.sweep_rate.textChanged.connect(self._update_nplc_from_sweep_rate)
        ct.sweep_rate_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        ct.start_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        ct.start_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        ct.stop_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        ct.stop_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        ct.step_size.textChanged.connect(self._update_nplc_from_sweep_rate)
        ct.step_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------

    def _setup_logging(self):
        """Route logging messages to the LogWidget."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.root.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.root.addHandler(console_handler)

        # Every view's log page. They share one handler on the root logger
        # (log_panel.install_root_handler) and one history, so whichever view
        # the operator is in shows the same stream from the same moment.
        for panel in self._log_panels():
            panel.attach_to_root()
        logger.info("Log routing configured")

    def toggle_debug_logging(self, enabled: bool):
        """Enable or disable debug logging to the Log tab."""
        if enabled:
            logging.root.setLevel(logging.DEBUG)
            logging.getLogger('pyvisa').setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            logging.root.setLevel(logging.INFO)
            logging.getLogger('pyvisa').setLevel(logging.WARNING)
            logger.info("Debug logging disabled")

    # -------------------------------------------------------------------------
    # NPLC Preview
    # -------------------------------------------------------------------------

    def _update_nplc_from_sweep_rate(self):
        """Calculate NPLC from sweep rate and update instrument tab preview.
        Reads from the active JV parameter source (combined tab or advanced tab)."""
        # Determine which parameter source is active
        if self.sidebar_mode_tabs.currentIndex() == 0:
            pt = self.combined_tab  # reads via CombinedTab properties
        else:
            pt = self.params_tab

        try:
            start_v = float(pt.start_voltage.text() or "0")
            if pt.start_unit.currentText() == "mV":
                start_v /= 1000.0

            stop_v = float(pt.stop_voltage.text() or "0")
            if pt.stop_unit.currentText() == "mV":
                stop_v /= 1000.0

            step_v = float(pt.step_size.text() or "0.01")
            if pt.step_unit.currentText() == "mV":
                step_v /= 1000.0
            step_v = abs(step_v)

            sweep_rate = float(pt.sweep_rate.text() or "0.1")
            if pt.sweep_rate_unit.currentText() == "mV/s":
                sweep_rate /= 1000.0

            if sweep_rate > 0 and step_v > 0 and abs(stop_v - start_v) > 0:
                total_points = int(abs(stop_v - start_v) / step_v) + 1
                total_time = abs(stop_v - start_v) / sweep_rate
                time_per_point = total_time / total_points

                nplc = time_per_point * 50
                nplc = max(0.01, min(10.0, nplc))

                # Both the combined view (default) and the advanced view
                # instantiate their OWN InstrumentTab. Update them together,
                # or the combined-view NPLC readout stays on its "1.0"
                # placeholder forever (the user sees "1" every time).
                self.instr_tab.update_nplc(nplc)
                self.combined_instr_tab.update_nplc(nplc)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # File and Directory Management
    # -------------------------------------------------------------------------

    def _update_save_directory(self):
        """Update save directory using directory manager."""
        main_dir = self.dir_manager.get_current_directory(create=True)
        self.file_panel.set_directory(main_dir)

    def _update_spo_save_directory(self):
        """Show the SPO output directory in the file panel while SPO mode
        is active. The directory manager mode is restored to "Main"
        immediately afterward by the shared singleton's own bookkeeping in
        SpoProcedure, so we just need to reflect the right path here."""
        previous_mode = self.dir_manager.mode
        try:
            self.dir_manager.set_mode("SPO")
            spo_dir = self.dir_manager.get_current_directory(create=False)
        finally:
            # ALWAYS restore the process-wide singleton's mode, even if the
            # directory lookup raises (e.g. unreachable network base dir).
            self.dir_manager.set_mode(previous_mode)
        self.file_panel.set_directory(spo_dir)

    # -------------------------------------------------------------------------
    # Instrument Status
    # -------------------------------------------------------------------------

    def update_channel_indicators(self, channels):
        """Rebuild the coloured channel badges in the persistent action bar."""
        from solarjv_analyzer.gui.widgets.analysis_panel import AnalysisPanel

        # Remove old indicators
        for lbl in self._channel_indicators:
            self._channel_indicators_layout.removeWidget(lbl)
            lbl.deleteLater()
        self._channel_indicators.clear()

        # Create fresh badges for each loaded channel
        for ch in sorted(channels):
            hex_colour = AnalysisPanel.CHANNEL_COLORS.get(ch, "#64748b")
            lbl = QtWidgets.QLabel(f" Ch {ch} ")
            lbl.setStyleSheet(
                f"background-color: {hex_colour};"
                f" color: white;"
                f" border-radius: 4px;"
                f" padding: 2px 6px;"
                f" font-size: 11px;"
                f" font-weight: bold;"
            )
            self._channel_indicators_layout.addWidget(lbl)
            self._channel_indicators.append(lbl)

    def update_architecture_badge(self, architecture: str) -> None:
        """Show / update the architecture badge in the action bar."""
        if architecture:
            self._architecture_badge.setText(architecture)
            self._architecture_badge.show()
        else:
            self._architecture_badge.hide()

    def update_instrument_lights(self, status: dict = None):
        """Update status indicator colors based on connection state.

        Reports LIVENESS, not `is not None`: a closed VISA session leaves the
        attribute set, which used to show a green "connected" dot for an
        instrument that could no longer be written to.

        When the monitor supplies a `status`, that is used — it also knows
        whether the USB adapter is still plugged in, which no handle can tell
        us. Without one this falls back to the handle check alone, so a direct
        call (e.g. right after a connect) still repaints sensibly.
        """
        if status is None:
            status = getattr(self, '_hardware_status', None)
        if status:
            k_connected = status.get('keithley_connected',
                                     self.instrument_manager.is_keithley_alive())
            m_connected = status.get('mux_connected',
                                     self.instrument_manager.is_mux_alive())
        else:
            k_connected = self.instrument_manager.is_keithley_alive()
            m_connected = self.instrument_manager.is_mux_alive()

        k_color = '#10b981' if k_connected else '#ef4444' # Updated to modern success/danger hex colors
        m_color = '#10b981' if m_connected else '#ef4444'

        self.keithley_light.setStyleSheet(f"border-radius:6px; background:{k_color};")
        self.mux_light.setStyleSheet(f"border-radius:6px; background:{m_color};")

        # The colour is the message. The tooltip only spells it out in words,
        # for anyone who cannot rely on red-versus-green.
        self.keithley_light.setToolTip(
            "Keithley: " + ("Connected" if k_connected else "Not connected"))
        self.mux_light.setToolTip(
            "MUX: " + ("Connected" if m_connected else "Not connected"))

    # -------------------------------------------------------------------------
    # Live hardware status
    # -------------------------------------------------------------------------
    HARDWARE_POLL_MS = 3000

    def start_hardware_monitor(self):
        """Keep the status lights telling the truth, continuously.

        The lights answer one question — is each instrument connected and
        usable *right now* — and the operator must be able to trust them before
        pressing anything. Previously they reported the application's own
        laziness: the calibration gate is Keithley-only by design and every
        `connect_mux()` call happened at run start, so the MUX light sat red
        with the hardware plugged in and working, then turned green the moment
        a run began. That is precisely backwards — by then it is too late to be
        useful.

        So: poll. Each tick enumerates the serial ports (a pure read), opens
        the MUX if its adapter is present but not yet open, and repaints the
        lights. A disappearing adapter turns the light red on its own, which
        neither `is_open` nor a VISA session handle would ever tell us.
        """
        if getattr(self, '_hardware_timer', None) is not None:
            return
        self._hardware_timer = QtCore.QTimer(self)
        self._hardware_timer.setInterval(self.HARDWARE_POLL_MS)
        self._hardware_timer.timeout.connect(self.check_hardware_status)
        self._hardware_timer.start()
        self.check_hardware_status()

    def check_hardware_status(self):
        """Run one status check on a worker thread."""
        existing = getattr(self, '_hardware_thread', None)
        if existing is not None and existing.isRunning():
            return
        self._hardware_thread = _HardwareCheckThread(
            self.instrument_manager, may_connect=not self._measurement_running(), parent=self
        )
        self._hardware_thread.checked.connect(self._on_hardware_checked)
        self._hardware_thread.start()

    def _measurement_running(self) -> bool:
        """True while anything is driving the instruments.

        The monitor must never open, close or probe an instrument mid-sweep —
        it only looks. Erring towards "busy" on an unexpected controller state
        is the safe direction.
        """
        controller = getattr(self, 'controller', None)
        if controller is None:
            return False
        try:
            return bool(
                getattr(controller, 'is_busy', False)
                or getattr(controller, 'spo_running', False)
                or getattr(controller, '_combined_mode', False)
            )
        except Exception:
            return True

    def _on_hardware_checked(self, status: dict):
        """Repaint the lights, and record any change of state in the log.

        The operator gets a colour. Everything needed to diagnose a red dot —
        which port, which adapter, why the open failed — goes to the session
        log instead of onto the screen, and only when the state actually
        changes: at one check every few seconds, logging every tick would bury
        the run's own messages.
        """
        self._hardware_status = status
        state = (bool(status.get('keithley_connected')),
                 bool(status.get('mux_connected')))
        if state != getattr(self, '_last_hardware_state', None):
            self._last_hardware_state = state
            keithley_ok, mux_ok = state
            logger.info(
                "Hardware status — Keithley: %s (%s%s); MUX: %s (%s%s)%s",
                "connected" if keithley_ok else "NOT connected",
                status.get('keithley_port', '?'),
                f", {status['keithley_description']}" if status.get('keithley_description') else "",
                "connected" if mux_ok else "NOT connected",
                status.get('mux_port', '?'),
                f", {status['mux_description']}" if status.get('mux_description') else "",
                f" — {status['mux_error']}" if status.get('mux_error') else "",
            )
        self.update_instrument_lights(status)

    # Kept for the startup call in main.py and for anything that wants an
    # immediate check rather than waiting for the next tick.
    def ensure_mux_connected(self):
        """Start the live status monitor (and check once, straight away)."""
        self.start_hardware_monitor()

    # -------------------------------------------------------------------------
    # Browser and Experiment Management
    # -------------------------------------------------------------------------

    def show_experiments(self):
        """Show all experiment curves in the plot."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Checked)
        self.analysis_panel.show()

    def hide_experiments(self):
        """Hide all experiment curves in the plot."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Unchecked)
        self.analysis_panel.hide()

    def clear_experiments(self):
        """Clear all experiments from the browser and analysis panel."""
        self.controller.clear_experiments()
        self.analysis_panel.clear_all()

    def open_experiment(self):
        """Open saved result files."""
        main_dir = self.dir_manager.get_current_directory(create=False)
        if not main_dir or not os.path.exists(main_dir):
            main_dir = os.path.expanduser("~")

        dialog = QtWidgets.QFileDialog(self, "Open Results File", main_dir)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dialog.setNameFilter("CSV Files (*.csv);;All Files (*)")
        if dialog.exec_():
            files = dialog.selectedFiles()
            self.controller.load_files(files)

    def browser_item_changed(self, item, column):
        """Show or hide curve when browser item checkbox toggled."""
        if column == 0:
            experiment = self.controller.manager.experiments.with_browser_item(item)
            if experiment:
                if item.checkState(0) == QtCore.Qt.Unchecked:
                    for curve in experiment.curve_list:
                        curve.wdg.remove(curve)
                else:
                    for curve in experiment.curve_list:
                        curve.wdg.load(curve)

    # -------------------------------------------------------------------------
    # Plot Export
    # -------------------------------------------------------------------------

    def save_plot(self):
        """Export the current JV plot as a PNG image (legacy view)."""
        try:
            exporter = ImageExporter(self.plot_widget.plot)
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save Plot", "", "PNG Image (*.png)"
            )
            if filename:
                if not filename.lower().endswith(".png"):
                    filename += ".png"
                exporter.export(filename)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Export Error", f"Failed to save image: {str(e)}"
            )

    def _save_combined_plots(self):
        """Export BOTH the JV and SPO plots as separate PNG images."""
        base, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Both Plots (base name)", "", "PNG Image (*.png)"
        )
        if not base:
            return
        if base.lower().endswith(".png"):
            base = base[:-4]

        errors = []
        # JV plot
        try:
            jv_path = f"{base}_JV.png"
            ImageExporter(self.plot_widget.plot).export(jv_path)
        except Exception as e:
            errors.append(f"JV plot: {e}")

        # SPO plot
        try:
            spo_path = f"{base}_SPO.png"
            ImageExporter(self.combined_spo_plot).export(spo_path)
        except Exception as e:
            errors.append(f"SPO plot: {e}")

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Export Error",
                f"Failed to save some plots:\n" + "\n".join(errors)
            )