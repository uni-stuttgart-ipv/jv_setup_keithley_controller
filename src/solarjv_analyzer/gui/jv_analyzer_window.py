"""
Main window for the JV Analyzer application, providing a GUI to configure,
queue, run, and manage JV measurement experiments.
"""

from PyQt5 import QtWidgets, QtCore
from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.gui.login_dialog import LoginDialog
from pymeasure.display.widgets import PlotWidget, LogWidget, BrowserWidget
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
import sys
import os
import logging
from solarjv_analyzer.config import GPIB_ADDRESS, RESULTS_ROOT, DATE_FORMAT, TIMESTAMP_FORMAT
from pymeasure.display.manager import Manager
# --- Additional imports ---
from pymeasure.display.browser import BrowserItem
from pymeasure.display.manager import Experiment
import pyqtgraph as pg
from datetime import datetime

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class JVAnalyzerWindow(QtWidgets.QMainWindow):
    """
    Main application window for configuring and managing JV experiments.
    Provides input widgets, plotting, logging, and experiment browser.
    """
    def __init__(self, username=None):
        super().__init__()
        self.username = username
        # Instantiate InstrumentManager at the top to ensure attribute exists
        self.instrument_manager = InstrumentManager()
        self.setWindowTitle("Custom JV Analyzer")
        self.resize(1000, 800)
        self._layout()
        self.connect_signals()
        # Initialize default save directory (Reports/YYYYMMDD)
        self.update_save_directory()
        # Disable browser controls until experiments are queued
        self.browser_widget.show_button.setEnabled(False)
        self.browser_widget.hide_button.setEnabled(False)
        self.browser_widget.clear_button.setEnabled(False)
        self.manager = Manager(
            [self.plot_widget, self.log_widget],
            self.browser_widget.browser,
            log_level=logging.INFO,
            parent=self
        )
        # Ensure browser knows exactly which data columns to plot
        self.browser_widget.browser.measured_quantities.clear()
        self.browser_widget.browser.measured_quantities.update(JVProcedure.DATA_COLUMNS)
        self.manager.abort_returned.connect(self.abort_returned)
        self.manager.queued.connect(self.queued)
        self.manager.running.connect(self.running)
        self.manager.finished.connect(self.finished)

    def _layout(self) -> None:
        self.main = QtWidgets.QWidget(self)

        # -------- Left Dock: Tabs for Parameters, Instruments, and Analysis --------
        tabs = QtWidgets.QTabWidget()

        # --- Parameters Tab ---
        params_tab = QtWidgets.QWidget()
        params_layout = QtWidgets.QFormLayout(params_tab)
        self.start_voltage = QtWidgets.QLineEdit("0.0")
        self.stop_voltage = QtWidgets.QLineEdit("1.0")
        self.step_size = QtWidgets.QLineEdit("0.1")
        self.compliance_current = QtWidgets.QLineEdit("0.1")
        params_layout.addRow("Start Voltage (V):", self.start_voltage)
        params_layout.addRow("Stop Voltage (V):", self.stop_voltage)
        params_layout.addRow("Step Size (V):", self.step_size)
        params_layout.addRow("Compliance Current (A):", self.compliance_current)
        # Pre-Sweep Delay stays only here
        self.pre_sweep_delay = QtWidgets.QLineEdit("0.0")
        params_layout.addRow("Pre-Sweep Delay (s):", self.pre_sweep_delay)
        self.device_area = QtWidgets.QLineEdit("0.089")
        params_layout.addRow("Device Area (cm²):", self.device_area)
        # Add a horizontal line separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        params_layout.addRow(line)
        # Channel selection
        self.channels = [QtWidgets.QCheckBox(f"Channel {i+1}") for i in range(6)]
        self.select_all_channels = QtWidgets.QCheckBox("Select All Channels")
        self.select_all_channels.toggled.connect(self.on_select_all_channels)
        channel_widget = QtWidgets.QWidget()
        channel_layout = QtWidgets.QVBoxLayout()
        channel_layout.setAlignment(QtCore.Qt.AlignRight)
        for ch in self.channels:
            channel_layout.addWidget(ch)
        channel_widget.setLayout(channel_layout)
        params_layout.addRow("Channels:", channel_widget)
        channel_layout.addWidget(self.select_all_channels)
        tabs.addTab(params_tab, "Parameters")

        # --- Instrument Tab ---
        instr_tab = QtWidgets.QWidget()
        instr_layout = QtWidgets.QFormLayout(instr_tab)
        self.instrument_name = QtWidgets.QComboBox()
        self.instrument_name.addItem("Keithley 2400")
        self.gpib_address = QtWidgets.QLineEdit(GPIB_ADDRESS)
        instr_layout.addRow("Instrument:", self.instrument_name)
        instr_layout.addRow("GPIB Address:", self.gpib_address)
        # Instrument-specific settings: NPLC, Delay Between Points, Measurement Range, Sense Mode
        self.nplc = QtWidgets.QLineEdit("1")
        instr_layout.addRow("NPLC (PLC units):", self.nplc)
        self.delay_between_points = QtWidgets.QLineEdit("0.1")
        instr_layout.addRow("Delay Between Points (s):", self.delay_between_points)
        self.measurement_range = QtWidgets.QComboBox()
        self.measurement_range.addItems(["Auto", "1 A", "100 mA", "10 mA", "1 mA", "100 uA"])
        instr_layout.addRow("Measurement Range:", self.measurement_range)
        self.sense_mode = QtWidgets.QComboBox()
        self.sense_mode.addItems(["2-wire", "4-wire"])
        instr_layout.addRow("Sense Mode:", self.sense_mode)
        tabs.addTab(instr_tab, "Instrument")

        # --- Analysis Tab ---
        analysis_tab = QtWidgets.QWidget()
        analysis_layout = QtWidgets.QFormLayout(analysis_tab)
        self.incident_power = QtWidgets.QLineEdit("100")
        self.contact_threshold = QtWidgets.QLineEdit("0.001")
        analysis_layout.addRow("Incident Power (mW/cm²):", self.incident_power)
        analysis_layout.addRow("Contact Threshold (A):", self.contact_threshold)
        self.lateral_factor = QtWidgets.QLineEdit("1.0")
        analysis_layout.addRow("4-Probe Lateral Factor (unitless):", self.lateral_factor)
        self.probe_spacing = QtWidgets.QLineEdit("2290")
        analysis_layout.addRow("4-Probe Spacing (μm):", self.probe_spacing)
        self.sample_thickness = QtWidgets.QLineEdit("500")
        analysis_layout.addRow("Sample Thickness (μm):", self.sample_thickness)
        tabs.addTab(analysis_tab, "Analysis")

        # -------- Below Tabs: File Input Section --------
        file_section = QtWidgets.QGroupBox("File Output")
        file_layout = QtWidgets.QFormLayout(file_section)
        self.filename_input = QtWidgets.QLineEdit("Output.csv")
        self.directory_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        browse_layout = QtWidgets.QHBoxLayout()
        browse_layout.addWidget(self.directory_input)
        browse_layout.addWidget(self.browse_button)
        file_layout.addRow("Filename Prefix:", self.filename_input)
        file_layout.addRow("Directory:", browse_layout)
        # Option to save all selected channels in one file
        self.single_file_checkbox = QtWidgets.QCheckBox("Save all channels in one file")
        file_layout.addRow(self.single_file_checkbox)

        # -------- Queue + Abort Buttons --------
        self.simulation_checkbox = QtWidgets.QCheckBox("Simulation Mode")
        # Place the simulation checkbox just above the button row
        self.queue_button = QtWidgets.QPushButton("Queue")
        self.abort_button = QtWidgets.QPushButton("Abort")
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.queue_button)
        button_layout.addWidget(self.abort_button)

        # -------- Combine Sidebar Layout --------
        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.addWidget(tabs)
        sidebar_layout.addWidget(file_section)
        sidebar_layout.addWidget(self.simulation_checkbox)
        sidebar_layout.addLayout(button_layout)
        sidebar_layout.addStretch()

        dock = QtWidgets.QDockWidget("Inputs")
        dock.setWidget(sidebar_widget)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # -------- Main Area Tabs --------
        self.tabs = QtWidgets.QTabWidget()
        self.plot_widget = PlotWidget(
            name="Plot",
            columns=JVProcedure.DATA_COLUMNS,
            x_axis=JVProcedure.DATA_COLUMNS[2],
            y_axis=JVProcedure.DATA_COLUMNS[1]
        )
        self.log_widget = LogWidget(name="Log")
        self.tabs.addTab(self.plot_widget, "Plot")
        self.tabs.addTab(self.log_widget, "Log")

        # -------- Results Browser Widget --------
        self.browser_widget = BrowserWidget(
            JVProcedure,
            ["start_voltage", "stop_voltage", "step_size", "compliance_current"],
            JVProcedure.DATA_COLUMNS,
            parent=self
        )

        # -------- Combine Main Layout --------
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main_splitter.addWidget(self.tabs)
        main_splitter.addWidget(self.browser_widget)

        main_layout = QtWidgets.QVBoxLayout(self.main)
        main_layout.addWidget(main_splitter)

        self.setCentralWidget(self.main)

    def on_select_all_channels(self, checked: bool) -> None:
        """Toggle all individual channel checkboxes."""
        for ch in self.channels:
            ch.setChecked(checked)

    def connect_signals(self) -> None:
        self.queue_button.clicked.connect(self.queue_experiment)
        self.abort_button.clicked.connect(self.abort_experiment)
        # hook up browser widget controls
        self.browser_widget.show_button.clicked.connect(self.show_experiments)
        self.browser_widget.hide_button.clicked.connect(self.hide_experiments)
        self.browser_widget.clear_button.clicked.connect(self.clear_experiments)
        self.browser_widget.open_button.clicked.connect(self.open_experiment)
        self.browser_widget.browser.itemChanged.connect(self.browser_item_changed)
        self.select_all_channels.toggled.connect(self.on_select_all_channels)
        # self.mux_select_button and self.handle_select_channel removed

    # Multiplexer channel selection UI and handler removed

    def queue_experiment(self) -> None:
        """
        Gather current input settings, create one or more JVProcedure instances,
        wrap them in Results and Experiment objects, and queue them for execution.
        """
        # Update save directory to today's Reports folder
        self.update_save_directory()
        logger.info("Experiment queued")
        from pymeasure.experiment import Results

        # Retrieve input values
        start = float(self.start_voltage.text())
        stop = float(self.stop_voltage.text())
        step = float(self.step_size.text())
        compliance = float(self.compliance_current.text())

        step_count = int((stop - start) / step) + 1

        filename = self.filename_input.text()
        directory = self.directory_input.text()
        # Generate a timestamp for unique filenames
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

        # Determine which channels are selected
        channels_selected = [
            i for i, ch in enumerate(self.channels, start=1) if ch.isChecked()
        ]
        if not channels_selected:
            return

        # Connect Keithley and multiplexer, using Simulation Mode if checked
        sim_mode = self.simulation_checkbox.isChecked()
        self.instrument_manager.connect_keithley(simulation=sim_mode)
        self.instrument_manager.connect_mux(simulation=sim_mode)

        base, ext = os.path.splitext(filename)
        if self.single_file_checkbox.isChecked():
            # Bundle all channels into one procedure/output file
            proc = JVProcedure(
                user_name=self.username,
                start_voltage=start,
                stop_voltage=stop,
                step_count=step_count,
                compliance_current=compliance,
                gpib_address=self.gpib_address.text(),
                channel1=1 in channels_selected,
                channel2=2 in channels_selected,
                channel3=3 in channels_selected,
                channel4=4 in channels_selected,
                channel5=5 in channels_selected,
                channel6=6 in channels_selected,
            )
            # Assign mux reference to procedure
            proc.mux = self.instrument_manager.mux
            ch_list_str = "_".join(str(ch) for ch in channels_selected)
            combined_filename = f"{base}_{timestamp}_ch{ch_list_str}{ext}"
            full_path = os.path.join(directory, combined_filename)

            results = Results(proc, full_path)
            logging.getLogger().addHandler(self.log_widget.handler)
            self.browser_widget.browser.measured_quantities.update(JVProcedure.DATA_COLUMNS)
            experiment = self.new_experiment(results)
            self.manager.queue(experiment)
        else:
            # Queue one experiment per selected channel
            for ch_num in channels_selected:
                proc = JVProcedure(
                    user_name=self.username,
                    start_voltage=start,
                    stop_voltage=stop,
                    step_count=step_count,
                    compliance_current=compliance,
                    gpib_address=self.gpib_address.text(),
                    channel1=(ch_num == 1),
                    channel2=(ch_num == 2),
                    channel3=(ch_num == 3),
                    channel4=(ch_num == 4),
                    channel5=(ch_num == 5),
                    channel6=(ch_num == 6),
                    active_channel=ch_num,
                    simulation=sim_mode
                )
                proc.mux = self.instrument_manager.mux
                single_filename = f"{base}_{timestamp}_ch{ch_num}{ext}"
                full_path = os.path.join(directory, single_filename)

                results = Results(proc, full_path)
                logging.getLogger().addHandler(self.log_widget.handler)
                self.browser_widget.browser.measured_quantities.update(JVProcedure.DATA_COLUMNS)
                experiment = self.new_experiment(results)
                self.manager.queue(experiment)

    def update_save_directory(self) -> None:
        """
        Ensure the default save directory is 'Reports/DD-MM-YYYY' and create it if missing.
        """
        today = datetime.now().strftime(DATE_FORMAT)
        reports_folder = os.path.join(RESULTS_ROOT, today)
        os.makedirs(reports_folder, exist_ok=True)
        self.directory_input.setText(reports_folder)

    def abort_experiment(self) -> None:
        """Abort the currently running experiment and switch to 'Resume' mode."""
        logger.info("Experiment aborted")
        # Disable and convert to Resume
        self.abort_button.setEnabled(False)
        self.abort_button.setText("Resume")
        self.abort_button.clicked.disconnect()
        self.abort_button.clicked.connect(self.resume_experiment)
        try:
            self.manager.abort()
        except Exception as e:
            logger.warning(f"Failed to abort: {e}")
            # Restore abort if something went wrong
            self.abort_button.setText("Abort")
            self.abort_button.clicked.disconnect()
            self.abort_button.clicked.connect(self.abort_experiment)

    def resume_experiment(self) -> None:
        """Resume the experiment queue if any remain."""
        logger.info("Experiment resumed")
        # Restore Abort button
        self.abort_button.setText("Abort")
        self.abort_button.clicked.disconnect()
        self.abort_button.clicked.connect(self.abort_experiment)
        # If there are queued experiments, resume the manager
        if self.manager.experiments.has_next():
            self.manager.resume()
        else:
            self.abort_button.setEnabled(False)

    def new_experiment(self, results) -> Experiment:
        """
        Create a new Experiment object with associated plot and browser items.
        
        Args:
            results: The Results object containing experiment data.
        
        Returns:
            An Experiment instance ready for management.
        """
        color = pg.intColor(self.browser_widget.browser.topLevelItemCount() % 8)
        curve = self.plot_widget.new_curve(results, color=color)
        browser_item = BrowserItem(results, color)
        return Experiment(results, [curve], browser_item)

    def abort_returned(self) -> None:
        """Handle logic after abort has returned."""
        logger.info("Abort returned")
        # If there are more experiments queued, allow resume; otherwise re-enable clear
        if self.manager.experiments.has_next():
            self.abort_button.setText("Resume")
            self.abort_button.setEnabled(True)
        else:
            self.browser_widget.clear_button.setEnabled(True)

    def queued(self) -> None:
        """Handle logic when an experiment is successfully queued."""
        logger.info("Experiment successfully queued")
        # Enable abort and browser controls when experiments are queued
        self.abort_button.setEnabled(True)
        self.abort_button.setText("Abort")
        self.abort_button.clicked.disconnect()
        self.abort_button.clicked.connect(self.abort_experiment)
        self.browser_widget.show_button.setEnabled(True)
        self.browser_widget.hide_button.setEnabled(True)
        self.browser_widget.clear_button.setEnabled(True)

    def running(self) -> None:
        """Handle logic when an experiment starts running."""
        logger.info("Experiment started")
        # Prevent clearing while an experiment is running
        self.browser_widget.clear_button.setEnabled(False)

    def finished(self) -> None:
        """Handle logic when an experiment finishes."""
        logger.info("Experiment finished")
        # When no more experiments remain, disable abort and re-enable clear
        if not self.manager.experiments.has_next():
            self.abort_button.setEnabled(False)
            self.abort_button.setText("Abort")
            self.browser_widget.clear_button.setEnabled(True)

    def show_experiments(self) -> None:
        """Show all curves in the plot by checking all items."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Checked)

    def hide_experiments(self) -> None:
        """Hide all curves by unchecking all items."""
        root = self.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            root.child(i).setCheckState(0, QtCore.Qt.Unchecked)

    def clear_experiments(self) -> None:
        """Clear all queued and finished experiments."""
        self.manager.clear()

    def open_experiment(self) -> None:
        """Open existing result files into the plot."""
        from pymeasure.display.widgets import ResultsDialog
        from pymeasure.experiment.results import Results as ResultsLoader

        dialog = ResultsDialog(JVProcedure, widget_list=[self.plot_widget, self.log_widget], parent=self)
        if dialog.exec_():
            for filename in map(str, dialog.selectedFiles()):
                if filename:
                    results = ResultsLoader.load(filename)
                    experiment = self.new_experiment(results)
                    self.manager.load(experiment)

    def browser_item_changed(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        """
        Handle changes in the browser item's checked state to show/hide curves.
        
        Args:
            item: The tree widget item that changed.
            column: The column index that was changed.
        """
        if column == 0:
            state = item.checkState(0)
            experiment = self.manager.experiments.with_browser_item(item)
            if experiment is not None:
                if state == QtCore.Qt.Unchecked:
                    for curve in experiment.curve_list:
                        curve.wdg.remove(curve)
                elif state == QtCore.Qt.Checked:
                    for curve in experiment.curve_list:
                        curve.wdg.load(curve)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    login_dialog = LoginDialog()
    if login_dialog.exec_() == QtWidgets.QDialog.Accepted:
        username = login_dialog.username
        window = JVAnalyzerWindow(username)
        window.show()
        sys.exit(app.exec_())
    else:
        sys.exit(0)