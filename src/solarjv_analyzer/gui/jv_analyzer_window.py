"""
Main window for the JV Analyzer application.

This class constructs the user interface by assembling various specialized
widget components. It creates and owns the AppController, which manages all
application logic and state. The window's role is to display the UI and
forward user interactions to the controller.
"""
from PyQt5 import QtWidgets, QtCore
import os
from datetime import datetime

from pymeasure.display.widgets import PlotWidget, LogWidget, BrowserWidget
from pymeasure.experiment import Results

from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.config import RESULTS_ROOT, DATE_FORMAT

# --- UI component and controller imports ---
from .widgets.parameter_tab import ParameterTab
from .widgets.instrument_tab import InstrumentTab
from .widgets.analysis_settings_tab import AnalysisSettingsTab
from .widgets.file_panel import FilePanel
from .widgets.analysis_panel import AnalysisPanel
from .app_controller import AppController


class JVAnalyzerWindow(QtWidgets.QMainWindow):
    """
    The main application window, acting as the 'View'.

    This class builds the visual layout from child widgets and delegates all
    control logic (e.g., queuing experiments, handling button clicks) to an
    instance of the AppController.
    """
    def __init__(self, username=None):
        super().__init__()
        self.username = username
        self.instrument_manager = InstrumentManager()
        self.setWindowTitle("Custom JV Analyzer")
        self.resize(1000, 720)
        self.setMinimumSize(1000, 700)

        self._layout()

        # The controller now manages all application logic
        self.controller = AppController(self)

        self.connect_signals()
        self.update_save_directory()

        # Set initial UI state
        self.browser_widget.show_button.setEnabled(False)
        self.browser_widget.hide_button.setEnabled(False)
        self.browser_widget.clear_button.setEnabled(False)

    def _layout(self) -> None:
        """Constructs the UI by assembling widgets from the .widgets module."""
        self.main = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main)

        # Assemble the input tabs from their dedicated classes
        input_tabs = QtWidgets.QTabWidget()
        self.params_tab = ParameterTab()
        self.instr_tab = InstrumentTab()
        self.analysis_settings_tab = AnalysisSettingsTab()
        input_tabs.addTab(self.params_tab, "Parameters")
        input_tabs.addTab(self.instr_tab, "Instrument")
        input_tabs.addTab(self.analysis_settings_tab, "Analysis")

        # Instantiate the file panel and control buttons
        self.file_panel = FilePanel()
        self.queue_button = QtWidgets.QPushButton("Queue")
        self.abort_button = QtWidgets.QPushButton("Abort")

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.queue_button)
        button_layout.addWidget(self.abort_button)

        # Connection status lights
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

        # Assemble the entire left-side dock widget
        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(input_tabs)
        sidebar_layout.addWidget(self.file_panel)
        sidebar_layout.addLayout(lights_row)
        sidebar_layout.addLayout(button_layout)
        sidebar_layout.addStretch()

        dock = QtWidgets.QDockWidget("Inputs")
        dock.setWidget(sidebar_widget)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # Assemble the main display area with splitters
        self.tabs = QtWidgets.QTabWidget()
        self.plot_widget = PlotWidget(name="Plot", columns=JVProcedure.DATA_COLUMNS, x_axis="Voltage (V)", y_axis="Current (A)")
        self.log_widget = LogWidget(name="Log")
        self.tabs.addTab(self.plot_widget, "Plot")
        self.tabs.addTab(self.log_widget, "Log")

        # Create a temporary procedure instance to dynamically get parameter names
        procedure_parameters = list(JVProcedure().parameter_objects().keys())
        self.browser_widget = BrowserWidget(JVProcedure, procedure_parameters, JVProcedure.DATA_COLUMNS)
        self.analysis_panel = AnalysisPanel(self)

        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        bottom_splitter.addWidget(self.browser_widget)
        bottom_splitter.addWidget(self.analysis_panel)
        bottom_splitter.setStretchFactor(0, 4)
        bottom_splitter.setStretchFactor(1, 1)

        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vertical_splitter.addWidget(self.tabs)
        vertical_splitter.addWidget(bottom_splitter)
        vertical_splitter.setStretchFactor(0, 2)
        vertical_splitter.setStretchFactor(1, 1)

        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.addWidget(vertical_splitter)

        self.update_instrument_lights()

    def connect_signals(self) -> None:
        """Connects UI element signals to the AppController's methods."""
        # Main control buttons are connected to the controller
        self.queue_button.clicked.connect(self.controller.queue_experiment)
        self.abort_button.clicked.connect(self.controller.abort_experiment)

        # Browser and file dialog actions are handled by the view
        self.file_panel.browse_button.clicked.connect(self.open_directory_dialog)
        self.browser_widget.show_button.clicked.connect(self.show_experiments)
        self.browser_widget.hide_button.clicked.connect(self.hide_experiments)
        self.browser_widget.clear_button.clicked.connect(self.clear_experiments)
        self.browser_widget.open_button.clicked.connect(self.open_experiment)
        self.browser_widget.browser.itemChanged.connect(self.browser_item_changed)

    def update_save_directory(self) -> None:
        """Sets the default save directory in the file panel to today's date."""
        today = datetime.now().strftime(DATE_FORMAT)
        reports_folder = os.path.join(RESULTS_ROOT, today)
        os.makedirs(reports_folder, exist_ok=True)
        self.file_panel.directory_input.setText(reports_folder)

    def open_directory_dialog(self) -> None:
        """Opens a native folder picker to select the output directory."""
        start_dir = self.file_panel.directory_input.text().strip() or os.getcwd()
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", start_dir)
        if selected:
            self.file_panel.directory_input.setText(selected)

    def update_instrument_lights(self) -> None:
        """Updates the status lights based on instrument connections."""
        k_connected = self.instrument_manager.keithley is not None
        m_connected = self.instrument_manager.mux is not None
        self.keithley_light.setStyleSheet(f"border-radius:7px; background:{'#3c3' if k_connected else '#c33'};")
        self.mux_light.setStyleSheet(f"border-radius:7px; background:{'#3c3' if m_connected else '#c33'};")

    def show_experiments(self) -> None:
        """Shows all experiment curves in the plot."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Checked)
        self.analysis_panel.show()

    def hide_experiments(self) -> None:
        """Hides all experiment curves in the plot."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Unchecked)
        self.analysis_panel.hide()

    def clear_experiments(self) -> None:
        """Clears the browser, plot, and analysis panel via the controller."""
        self.controller.clear_experiments()
        self.analysis_panel.clear_all()

    def open_experiment(self) -> None:
        """Opens existing result files into the application."""
        dialog = QtWidgets.QFileDialog(self, "Open Results File")
        if dialog.exec_():
            for filename in dialog.selectedFiles():
                results = Results.load(filename)
                experiment = self.controller._new_experiment(results)
                self.controller.manager.load(experiment)

    def browser_item_changed(self, item, column):
        """Shows or hides a curve when its checkbox is toggled."""
        if column == 0:
            experiment = self.controller.manager.experiments.with_browser_item(item)
            if experiment:
                if item.checkState(0) == QtCore.Qt.Unchecked:
                    for curve in experiment.curve_list:
                        curve.wdg.remove(curve)
                else:
                    for curve in experiment.curve_list:
                        curve.wdg.load(curve)