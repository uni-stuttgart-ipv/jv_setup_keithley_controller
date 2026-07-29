"""
Main Window for JV Analyzer Application

Provides the graphical user interface for the JV measurement system,
including parameter input, instrument control, live plotting, and log display.
"""

import logging
import os
import sys
from datetime import datetime

from pathlib import Path
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtWidgets, QtCore
from pymeasure.display.widgets import PlotWidget, LogWidget, BrowserWidget

from solarjv_analyzer.config import RESULTS_ROOT, DATE_FORMAT
from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.utils.directory_manager import DirectoryManager

from .widgets.parameter_tab import ParameterTab
from .widgets.instrument_tab import InstrumentTab
from .widgets.analysis_settings_tab import AnalysisSettingsTab
from .widgets.file_panel import FilePanel
from .widgets.analysis_panel import AnalysisPanel
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

            /* Primary Button (Queue) */
            QPushButton#QueueButton {
                background-color: #2563eb;
                color: white;
                border: none;
                font-weight: 600;
            }
            QPushButton#QueueButton:hover { background-color: #1d4ed8; }
            QPushButton#QueueButton:pressed { background-color: #1e40af; }

            /* Danger Button (Abort) */
            QPushButton#AbortButton {
                background-color: #ef4444;
                color: white;
                border: none;
                font-weight: 600;
            }
            QPushButton#AbortButton:hover { background-color: #dc2626; }
            QPushButton#AbortButton:pressed { background-color: #b91c1c; }

            /* Mode Toggle Buttons (JV Sweep / SPO) */
            QPushButton#ModeButton {
                font-weight: 600;
            }
            QPushButton#ModeButton:checked {
                background-color: #16a34a;
                color: white;
                border: 1px solid #15803d;
            }
            QPushButton#ModeButton:checked:hover { background-color: #15803d; }
            QPushButton#ModeButton:checked:pressed { background-color: #166534; }

            /* Success Button (Save Plot) */
            QPushButton#SavePlotButton {
                background-color: #10b981;
                color: white;
                border: none;
                font-weight: 600;
                margin: 8px 0px;
            }
            QPushButton#SavePlotButton:hover { background-color: #059669; }
        """


    def __init__(self, username=None):
        """
        Initialize the main window.

        Args:
            username: Name of the user operating the system
        """
        super().__init__()
        self.username = username
        self.instrument_manager = InstrumentManager()

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

        # Configure logging to display in Log tab
        self._setup_logging()

        # Connect debug checkbox
        self.analysis_settings_tab.connect_debug_signal(self.toggle_debug_logging)

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
        self.main = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main)

        # Input tabs
        input_tabs = QtWidgets.QTabWidget()
        self.params_tab = ParameterTab()
        self.instr_tab = InstrumentTab()
        self.analysis_settings_tab = AnalysisSettingsTab()

        # SPO parameter fields occupy the same "Parameters" tab slot as the
        # JV ParameterTab; a QStackedWidget swaps between them by mode so
        # the Instrument/Analysis tabs and the tab bar itself never change.
        self.params_stack = QtWidgets.QStackedWidget()
        self.params_stack.addWidget(self.params_tab)
        if SPO_AVAILABLE:
            self.spo_param_tab = SpoParameterTab(self)
        else:
            logger.warning("SPO module unavailable; SPO mode disabled.")
            self.spo_param_tab = QtWidgets.QWidget()
        self.params_stack.addWidget(self.spo_param_tab)

        input_tabs.addTab(self.params_stack, "Parameters")
        input_tabs.addTab(self.instr_tab, "Instrument")
        input_tabs.addTab(self.analysis_settings_tab, "Analysis")

        # File panel and control buttons
        self.file_panel = FilePanel()
        self.queue_button = QtWidgets.QPushButton("Queue")
        self.queue_button.setObjectName("QueueButton")
        self.abort_button = QtWidgets.QPushButton("Abort")
        self.abort_button.setObjectName("AbortButton")

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(12) # Added spacing between buttons
        button_layout.addWidget(self.queue_button)
        button_layout.addWidget(self.abort_button)

        # SPO Start/Abort buttons (hidden until SPO mode is selected)
        self.spo_start_button = QtWidgets.QPushButton("Start SPO")
        self.spo_start_button.setObjectName("QueueButton")
        self.spo_start_button.setEnabled(False)  # enabled once a Vmax is available
        self.spo_abort_button = QtWidgets.QPushButton("Abort SPO")
        self.spo_abort_button.setObjectName("AbortButton")
        self.spo_abort_button.setEnabled(False)

        spo_button_layout = QtWidgets.QHBoxLayout()
        spo_button_layout.setSpacing(12)
        spo_button_layout.addWidget(self.spo_start_button)
        spo_button_layout.addWidget(self.spo_abort_button)

        # Mode toggle: JV Sweep vs SPO (mutually exclusive)
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

        # Instrument status lights
        lights_row = self._create_status_lights()

        # Left sidebar (now inside a scroll area)
        sidebar_widget = QtWidgets.QWidget()
        sidebar_widget.setMinimumWidth(400) # Increased width slightly for breathing room
        
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        # Added margins and spacing for a cleaner, modern flow
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(20) 
        
        sidebar_layout.addLayout(mode_row)
        sidebar_layout.addWidget(input_tabs)
        sidebar_layout.addWidget(self.file_panel)
        sidebar_layout.addLayout(lights_row)
        sidebar_layout.addLayout(button_layout)
        sidebar_layout.addLayout(spo_button_layout)
        sidebar_layout.addStretch()

        # SPO buttons stay hidden until SPO mode is selected
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

        # Main display area
        plot_container = self._create_plot_container()
        bottom_splitter = self._create_bottom_splitter()
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vertical_splitter.addWidget(plot_container)
        self.vertical_splitter.addWidget(bottom_splitter)
        self.vertical_splitter.setStretchFactor(0, 2)
        self.vertical_splitter.setStretchFactor(1, 1)

        # SPO view: replaces the JV display area entirely while active.
        # SPO is optional/self-contained: if unavailable, fall back to a
        # disabled placeholder so the rest of the JV app still runs.
        if SPO_AVAILABLE:
            self.spo_widget = SpoWidget(self, self.spo_param_tab)
        else:
            self.spo_widget = QtWidgets.QWidget()
            self.spo_mode_button.setEnabled(False)
            self.spo_mode_button.setToolTip("SPO module is not installed.")
        self.spo_widget.hide()

        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.setContentsMargins(0, 0, 0, 0) # Removed hard main margins for edge-to-edge splitters
        main_layout.addWidget(self.vertical_splitter)
        main_layout.addWidget(self.spo_widget)

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

        lights_row.addStretch(1)
        lights_row.addWidget(self.keithley_light)
        lights_row.addWidget(QtWidgets.QLabel("Keithley"))
        lights_row.addSpacing(20)
        lights_row.addWidget(self.mux_light)
        lights_row.addWidget(QtWidgets.QLabel("MUX"))
        lights_row.addStretch(1)

        return lights_row

    def _create_plot_container(self):
        """Create the plot widget container with save button."""
        self.tabs = QtWidgets.QTabWidget()
        self.plot_widget = PlotWidget(
            name="Plot",
            columns=JVProcedure.DATA_COLUMNS,
            x_axis="Voltage (V)",
            y_axis="Current (A)"
        )
        self.plot_widget.plot.showGrid(x=True, y=True, alpha=0.3)

        self.log_widget = LogWidget(name="Log")
        self.tabs.addTab(self.plot_widget, "Plot")
        self.tabs.addTab(self.log_widget, "Log")

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12) # Added proper padding around plot
        layout.addWidget(self.tabs)

        self.save_plot_button = QtWidgets.QPushButton("Save Plot as PNG")
        self.save_plot_button.setObjectName("SavePlotButton")
        layout.addWidget(self.save_plot_button)

        return container

    def _create_bottom_splitter(self):
        """Create the bottom splitter with browser and analysis panel."""
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

        self.analysis_panel = AnalysisPanel(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.browser_widget)
        splitter.addWidget(self.analysis_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        # Add a wrapper to ensure padding around the bottom section
        wrapper = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(wrapper)
        layout.setContentsMargins(12, 0, 12, 12)
        layout.addWidget(splitter)

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
        if getattr(self.controller, 'spo_running', False):
            if not self._confirm_abort_spo(
                "An SPO measurement is currently running.\n\n"
                "Logging out will abort it and save the partial data. Continue?"
            ):
                return

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
        self.spo_widget.show()
        self.params_stack.setCurrentWidget(self.spo_param_tab)
        self.queue_button.hide()
        self.abort_button.hide()
        self.spo_start_button.show()
        self.spo_abort_button.show()
        self._update_spo_save_directory()
        self._set_file_panel_spo_mode(True)
        self.spo_widget.set_mode_spo()

    def _show_jv_mode(self):
        """Show the JV plot/browser/analysis views and hide the SPO view."""
        self.spo_widget.hide()
        self.vertical_splitter.show()
        self.params_stack.setCurrentWidget(self.params_tab)
        self.spo_start_button.hide()
        self.spo_abort_button.hide()
        self.queue_button.show()
        self.abort_button.show()
        self._set_file_panel_spo_mode(False)
        self._update_save_directory()

    def _set_file_panel_spo_mode(self, spo_mode: bool):
        """Adapt the File Panel for SPO mode: SPO writes its own raw CSV /
        report filenames via DirectoryManager, so the filename prefix and
        single-file options (which only apply to JV sweeps) are locked."""
        self.file_panel.filename_input.setEnabled(not spo_mode)
        self.file_panel.single_file_checkbox.setEnabled(not spo_mode)

    def _on_spo_vmax_ready(self, ready: bool):
        """Enable Start SPO only once a Vmax is available and nothing is running."""
        self.spo_start_button.setEnabled(ready and not getattr(self.controller, 'spo_running', False))

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def connect_signals(self):
        """Connect UI signals to controller methods."""
        self.queue_button.clicked.connect(self.controller.queue_experiment)
        self.abort_button.clicked.connect(self.controller.abort_experiment)
        self.save_plot_button.clicked.connect(self.save_plot)

        self.browser_widget.show_button.clicked.connect(self.show_experiments)
        self.browser_widget.hide_button.clicked.connect(self.hide_experiments)
        self.browser_widget.clear_button.clicked.connect(self.clear_experiments)
        self.browser_widget.open_button.clicked.connect(self.open_experiment)
        self.browser_widget.browser.itemChanged.connect(self.browser_item_changed)
        self.browser_widget.browser.itemSelectionChanged.connect(
            self.controller.on_browser_selection_changed
        )

        # SPO mode toggle and Start/Abort buttons
        self.mode_button_group.buttonClicked.connect(self._on_mode_button_clicked)
        if SPO_AVAILABLE:
            self.spo_start_button.clicked.connect(self.controller.start_spo)
            self.spo_abort_button.clicked.connect(self.controller.abort_spo)
            self.spo_widget.vmax_ready.connect(self._on_spo_vmax_ready)

    def _connect_nplc_preview_signals(self):
        """Connect parameter signals for NPLC preview calculation."""
        self.params_tab.sweep_rate.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.sweep_rate_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.start_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.start_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.stop_voltage.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.stop_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.step_size.textChanged.connect(self._update_nplc_from_sweep_rate)
        self.params_tab.step_unit.currentTextChanged.connect(self._update_nplc_from_sweep_rate)

    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------

    def _setup_logging(self):
        """Route logging messages to the LogWidget."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.root.setLevel(logging.INFO)

        if hasattr(self.log_widget, 'handler'):
            logging.root.addHandler(self.log_widget.handler)
            logger.info("Log routing configured")
        else:
            self._create_fallback_handler()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logging.root.addHandler(console_handler)

    def _create_fallback_handler(self):
        """Create a fallback handler that writes directly to LogWidget's text edit."""
        class DirectLogHandler(logging.Handler):
            def __init__(self, log_widget):
                super().__init__()
                self.log_widget = log_widget
                self.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

            def emit(self, record):
                try:
                    msg = self.format(record)
                    for child in self.log_widget.findChildren(QtWidgets.QTextEdit):
                        child.append(msg)
                        break
                except Exception:
                    pass

        handler = DirectLogHandler(self.log_widget)
        logging.root.addHandler(handler)
        logger.info("Fallback log handler created")

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
        """Calculate NPLC from sweep rate and update instrument tab preview."""
        try:
            start_v = float(self.params_tab.start_voltage.text() or "0")
            if self.params_tab.start_unit.currentText() == "mV":
                start_v /= 1000.0

            stop_v = float(self.params_tab.stop_voltage.text() or "0")
            if self.params_tab.stop_unit.currentText() == "mV":
                stop_v /= 1000.0

            step_v = float(self.params_tab.step_size.text() or "0.01")
            if self.params_tab.step_unit.currentText() == "mV":
                step_v /= 1000.0
            step_v = abs(step_v)

            sweep_rate = float(self.params_tab.sweep_rate.text() or "0.1")
            if self.params_tab.sweep_rate_unit.currentText() == "mV/s":
                sweep_rate /= 1000.0

            if sweep_rate > 0 and step_v > 0 and abs(stop_v - start_v) > 0:
                total_points = int(abs(stop_v - start_v) / step_v) + 1
                total_time = abs(stop_v - start_v) / sweep_rate
                time_per_point = total_time / total_points

                nplc = time_per_point * 50
                nplc = max(0.01, min(10.0, nplc))

                self.instr_tab.update_nplc(nplc)
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
        self.dir_manager.set_mode("SPO")
        spo_dir = self.dir_manager.get_current_directory(create=False)
        self.dir_manager.set_mode(previous_mode)
        self.file_panel.set_directory(spo_dir)

    # -------------------------------------------------------------------------
    # Instrument Status
    # -------------------------------------------------------------------------

    def update_instrument_lights(self):
        """Update status indicator colors based on connection state."""
        k_connected = self.instrument_manager.keithley is not None
        m_connected = self.instrument_manager.mux is not None

        k_color = '#10b981' if k_connected else '#ef4444' # Updated to modern success/danger hex colors
        m_color = '#10b981' if m_connected else '#ef4444'

        self.keithley_light.setStyleSheet(f"border-radius:6px; background:{k_color};")
        self.mux_light.setStyleSheet(f"border-radius:6px; background:{m_color};")

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
        """Export the current plot as a PNG image."""
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