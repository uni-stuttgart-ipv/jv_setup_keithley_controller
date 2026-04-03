"""
Main Window for JV Analyzer Application

Provides the graphical user interface for the JV measurement system,
including parameter input, instrument control, live plotting, and log display.
"""

import logging
import os
import sys
from datetime import datetime

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from pymeasure.display.widgets import PlotWidget, LogWidget, BrowserWidget

from solarjv_analyzer.config import RESULTS_ROOT, DATE_FORMAT
from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure

from .widgets.parameter_tab import ParameterTab
from .widgets.instrument_tab import InstrumentTab
from .widgets.analysis_settings_tab import AnalysisSettingsTab
from .widgets.file_panel import FilePanel
from .widgets.analysis_panel import AnalysisPanel
from .app_controller import AppController

logger = logging.getLogger(__name__)


class JVAnalyzerWindow(QtWidgets.QMainWindow):
    """
    Main application window for J-V measurement and analysis.

    Provides:
    - Parameter input tabs for experiment configuration
    - Live plot of current vs voltage during measurement
    - Log tab for system messages and debug output
    - Browser for managing multiple experiments
    - Analysis panel for displaying solar cell metrics
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

        self.setWindowTitle("Custom JV Analyzer")
        self.resize(1200, 720)
        self.setMinimumSize(1000, 700)

        # Build the user interface
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
        input_tabs.addTab(self.params_tab, "Parameters")
        input_tabs.addTab(self.instr_tab, "Instrument")
        input_tabs.addTab(self.analysis_settings_tab, "Analysis")

        # File panel and control buttons
        self.file_panel = FilePanel()
        self.queue_button = QtWidgets.QPushButton("Queue")
        self.abort_button = QtWidgets.QPushButton("Abort")

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.queue_button)
        button_layout.addWidget(self.abort_button)

        # Instrument status lights
        lights_row = self._create_status_lights()

        # Left sidebar
        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(input_tabs)
        sidebar_layout.addWidget(self.file_panel)
        sidebar_layout.addLayout(lights_row)
        sidebar_layout.addLayout(button_layout)
        sidebar_layout.addStretch()

        sidebar_dock = QtWidgets.QDockWidget("Inputs")
        sidebar_dock.setWidget(sidebar_widget)
        sidebar_dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, sidebar_dock)

        # Main display area
        plot_container = self._create_plot_container()
        bottom_splitter = self._create_bottom_splitter()
        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical_splitter.addWidget(plot_container)
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setStretchFactor(0, 2)
        vertical_splitter.setStretchFactor(1, 1)

        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.addWidget(vertical_splitter)

        self.update_instrument_lights()

    def _create_status_lights(self):
        """Create instrument connection status indicators."""
        lights_row = QtWidgets.QHBoxLayout()
        self.keithley_light = QtWidgets.QLabel("  ")
        self.keithley_light.setFixedSize(14, 14)
        self.keithley_light.setStyleSheet("border-radius:7px; background:#c33;")

        self.mux_light = QtWidgets.QLabel("  ")
        self.mux_light.setFixedSize(14, 14)
        self.mux_light.setStyleSheet("border-radius:7px; background:#c33;")

        lights_row.addStretch(1)
        lights_row.addWidget(self.keithley_light)
        lights_row.addWidget(QtWidgets.QLabel("Keithley"))
        lights_row.addSpacing(12)
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)

        self.save_plot_button = QtWidgets.QPushButton("Save Plot as PNG")
        layout.addWidget(self.save_plot_button)

        return container

    def _create_bottom_splitter(self):
        """Create the bottom splitter with browser and analysis panel."""
        # Browser configuration
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

        return splitter

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def connect_signals(self):
        """Connect UI signals to controller methods."""
        self.queue_button.clicked.connect(self.controller.queue_experiment)
        self.abort_button.clicked.connect(self.controller.abort_experiment)
        self.save_plot_button.clicked.connect(self.save_plot)

        self.file_panel.browse_button.clicked.connect(self.open_directory_dialog)
        self.browser_widget.show_button.clicked.connect(self.show_experiments)
        self.browser_widget.hide_button.clicked.connect(self.hide_experiments)
        self.browser_widget.clear_button.clicked.connect(self.clear_experiments)
        self.browser_widget.open_button.clicked.connect(self.open_experiment)
        self.browser_widget.browser.itemChanged.connect(self.browser_item_changed)
        self.browser_widget.browser.itemSelectionChanged.connect(
            self.controller.on_browser_selection_changed
        )

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
        # Remove existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.root.setLevel(logging.INFO)

        # Use LogWidget's built-in handler if available
        if hasattr(self.log_widget, 'handler'):
            logging.root.addHandler(self.log_widget.handler)
            logger.info("Log routing configured")
        else:
            self._create_fallback_handler()

        # Console handler for debugging (WARNING level only)
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
            # Parse start voltage
            start_v = float(self.params_tab.start_voltage.text() or "0")
            if self.params_tab.start_unit.currentText() == "mV":
                start_v /= 1000.0

            # Parse stop voltage
            stop_v = float(self.params_tab.stop_voltage.text() or "0")
            if self.params_tab.stop_unit.currentText() == "mV":
                stop_v /= 1000.0

            # Parse step size
            step_v = float(self.params_tab.step_size.text() or "0.01")
            if self.params_tab.step_unit.currentText() == "mV":
                step_v /= 1000.0
            step_v = abs(step_v)

            # Parse sweep rate
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
            pass  # Silent failure - user input may be incomplete

    # -------------------------------------------------------------------------
    # File and Directory Management
    # -------------------------------------------------------------------------

    def _update_save_directory(self):
        """Set default save directory to today's date folder (only if not set)."""
        today = datetime.now().strftime(DATE_FORMAT)
        reports_folder = os.path.join(RESULTS_ROOT, today)
        os.makedirs(reports_folder, exist_ok=True)

        # Only set if directory is empty or doesn't exist
        current_dir = self.file_panel.get_directory()
        if not current_dir or not os.path.exists(current_dir):
            self.file_panel.set_directory(reports_folder)

    def open_directory_dialog(self):
        print("Dialog opened")
        """Open folder picker for output directory selection."""
        start_dir = self.file_panel.get_directory() or os.getcwd()
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if selected:
            self.file_panel.set_directory(selected)

    # -------------------------------------------------------------------------
    # Instrument Status
    # -------------------------------------------------------------------------

    def update_instrument_lights(self):
        """Update status indicator colors based on connection state."""
        k_connected = self.instrument_manager.keithley is not None
        m_connected = self.instrument_manager.mux is not None

        k_color = '#3c3' if k_connected else '#c33'
        m_color = '#3c3' if m_connected else '#c33'

        self.keithley_light.setStyleSheet(f"border-radius:7px; background:{k_color};")
        self.mux_light.setStyleSheet(f"border-radius:7px; background:{m_color};")

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
        dialog = QtWidgets.QFileDialog(self, "Open Results File")
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
            exporter = pg.exporters.ImageExporter(self.plot_widget.plot)
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