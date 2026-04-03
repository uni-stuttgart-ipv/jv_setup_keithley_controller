"""
Calibration Window for System Validation

Provides a mandatory calibration gate before accessing the main application.
Includes hardware connection verification, reference cell measurement,
and pass/fail criteria based on Isc tolerance.
"""

import logging
import os
import tempfile

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.display.widgets import PlotWidget, BrowserWidget
from pymeasure.experiment import Results

from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helper Classes
# -------------------------------------------------------------------------

class BrowserProgressRelay(QtCore.QObject):
    """Relay signals from non-QObject items in a thread-safe manner."""
    progress_signal = QtCore.pyqtSignal(float)


class SignalBrowserItem(BrowserItem):
    """BrowserItem that emits a Qt signal when progress is updated."""

    def __init__(self, results, color, progress_callback=None):
        super().__init__(results, color)
        self.relay = BrowserProgressRelay()
        if progress_callback:
            self.relay.progress_signal.connect(progress_callback)

    def setProgress(self, progress):
        super().setProgress(progress)
        self.relay.progress_signal.emit(progress)


class CalibrationChecklistDialog(QtWidgets.QDialog):
    """Startup checklist dialog ensuring system readiness."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Readiness Check")
        self.resize(450, 420)
        self._setup_ui()

    def _setup_ui(self):
        """Build the checklist dialog UI."""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
            }
            QLabel#Header {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 18px;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 10px;
            }
            QLabel#SubHeader {
                font-size: 13px;
                color: #7f8c8d;
                margin-bottom: 15px;
            }
            QCheckBox {
                font-size: 14px;
                padding: 8px;
                spacing: 10px;
                color: #34495e;
                border: 1px solid #ecf0f1;
                border-radius: 5px;
                background-color: #fcfcfc;
                margin-bottom: 4px;
            }
            QCheckBox:hover {
                background-color: #f7f9fa;
                border-color: #bdc3c7;
            }
            QPushButton {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
                font-weight: 600;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton#ConfirmBtn {
                background-color: #27ae60;
                color: white;
                border: none;
            }
            QPushButton#ConfirmBtn:disabled {
                background-color: #bdc3c7;
                color: #ecf0f1;
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        # Header
        header = QtWidgets.QLabel("Pre-Calibration Safety Check")
        header.setObjectName("Header")
        header.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(header)

        desc = QtWidgets.QLabel("Verify all conditions before energizing the system:")
        desc.setObjectName("SubHeader")
        desc.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(desc)

        # Checklist items
        self.steps = [
            "1. Turn ON: Chiller, Sun Sim, Keithley",
            "2. Wavelabs: Load 'AM1.5G' Recipe",
            "3. Place Si-Reference Cell (RERA)",
            "4. Verify Lamp Height (46.8 cm)"
        ]

        self.checks = []
        for step in self.steps:
            checkbox = QtWidgets.QCheckBox(step)
            checkbox.setCursor(QtCore.Qt.PointingHandCursor)
            checkbox.stateChanged.connect(self._validate_checklist)
            layout.addWidget(checkbox)
            self.checks.append(checkbox)

        layout.addStretch()

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setSpacing(15)

        self.ok_btn = QtWidgets.QPushButton("Confirm Readiness")
        self.ok_btn.setObjectName("ConfirmBtn")
        self.ok_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)

        button_layout.addWidget(self.ok_btn)
        layout.addLayout(button_layout)

    def _validate_checklist(self):
        """Enable confirm button only when all items are checked."""
        all_checked = all(cb.isChecked() for cb in self.checks)
        self.ok_btn.setEnabled(all_checked)


# -------------------------------------------------------------------------
# Main Calibration Window
# -------------------------------------------------------------------------

class CalibrationWindow(QtWidgets.QMainWindow):
    """
    Calibration gate for system validation before main application access.

    Verifies reference cell measurement against target Isc within tolerance.
    Provides skip option for emergency access when calibration is not required.
    """

    calibration_passed = QtCore.pyqtSignal(object)

    # Protocol constants
    DEFAULT_TARGET_ISC = 0.0596      # 59.6 mA
    DEFAULT_TARGET_JSC = 14.9        # 14.9 mA/cm²
    DEFAULT_TOLERANCE = 5.0          # ±5%
    DEFAULT_AREA = 4.0               # 4 cm²
    DEFAULT_START_V = 0.7
    DEFAULT_STOP_V = -0.2
    DEFAULT_STEP_V = -0.01

    def __init__(self, username, parent=None):
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("System Calibration")
        self.resize(1000, 600)

        self.instrument_manager = InstrumentManager()
        self.manager = None
        self.checklist_confirmed = False

        self._setup_ui()
        self._connect_hardware()

        # Show checklist after UI is ready
        QtCore.QTimer.singleShot(200, self.launch_checklist_dialog)

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        """Build the calibration window interface."""
        # Configure plot appearance
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Main window styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f7f9fc;
            }
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QGroupBox {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 12px;
                font-weight: 600;
                color: #2c3e50;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                background-color: transparent;
            }
            QLabel {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 12px;
                color: #505c6e;
            }
            QPushButton {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-weight: 600;
                border-radius: 5px;
                padding: 6px;
            }
            QComboBox, QDoubleSpinBox {
                padding: 4px;
                border: 1px solid #d1d5da;
                border-radius: 4px;
                background: white;
                font-size: 12px;
                color: #24292e;
            }
            QComboBox::drop-down {
                border: 0px;
            }
        """)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left column (scrollable controls)
        left_scroll = self._create_left_panel()
        main_layout.addWidget(left_scroll)

        # Right column (plot)
        plot_container = self._create_plot_panel()
        main_layout.addWidget(plot_container, stretch=1)

    def _create_left_panel(self):
        """Create the scrollable left control panel."""
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setFixedWidth(370)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        left_content = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_content)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 10, 0)

        # Protocol Reference
        left_layout.addWidget(self._create_protocol_group())

        # Configuration
        left_layout.addWidget(self._create_config_group())

        # Hardware Status
        left_layout.addWidget(self._create_hardware_group())

        # Calibration Status
        left_layout.addWidget(self._create_status_group())

        # Actions
        left_layout.addWidget(self._create_actions_group())

        # Proceed Buttons
        left_layout.addWidget(self._create_proceed_buttons())

        left_layout.addStretch()

        scroll_area.setWidget(left_content)
        return scroll_area

    def _create_protocol_group(self):
        """Create the protocol reference instructions group."""
        group = QtWidgets.QGroupBox("Protocol Reference")
        layout = QtWidgets.QVBoxLayout(group)

        instructions = QtWidgets.QLabel(
            """
            <ol style='margin-left: 15px; margin-top:0px; margin-bottom:0px; line-height: 120%;'>
                <li>Turn ON: Chiller, Sun Sim, Keithley</li>
                <li>Wavelabs: Load 'AM1.5G' Recipe</li>
                <li>Place Si-Reference Cell (RERA)</li>
                <li>Verify Lamp Height (46.8 cm)</li>
            </ol>
            """
        )
        layout.addWidget(instructions)
        return group

    def _create_config_group(self):
        """Create the configuration parameters group."""
        group = QtWidgets.QGroupBox("Configuration")
        layout = QtWidgets.QFormLayout(group)
        layout.setVerticalSpacing(6)
        layout.setLabelAlignment(QtCore.Qt.AlignLeft)

        # Target Isc
        self.spin_target_isc = QtWidgets.QDoubleSpinBox()
        self.spin_target_isc.setDecimals(4)
        self.spin_target_isc.setValue(self.DEFAULT_TARGET_ISC)
        self.spin_target_isc.setSuffix(" A")
        layout.addRow("Target Isc:", self.spin_target_isc)

        # Tolerance
        self.spin_tolerance = QtWidgets.QDoubleSpinBox()
        self.spin_tolerance.setValue(self.DEFAULT_TOLERANCE)
        self.spin_tolerance.setSuffix(" %")
        layout.addRow("Tolerance:", self.spin_tolerance)

        # Reference Area
        self.spin_area = QtWidgets.QDoubleSpinBox()
        self.spin_area.setValue(self.DEFAULT_AREA)
        self.spin_area.setSuffix(" cm²")
        layout.addRow("Ref. Area:", self.spin_area)

        # Separator
        layout.addRow(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Sweep parameters
        self.spin_start_v = QtWidgets.QDoubleSpinBox()
        self.spin_start_v.setRange(-10, 10)
        self.spin_start_v.setValue(self.DEFAULT_START_V)
        self.spin_start_v.setSuffix(" V")
        layout.addRow("Start V:", self.spin_start_v)

        self.spin_stop_v = QtWidgets.QDoubleSpinBox()
        self.spin_stop_v.setRange(-10, 10)
        self.spin_stop_v.setValue(self.DEFAULT_STOP_V)
        self.spin_stop_v.setSuffix(" V")
        layout.addRow("Stop V:", self.spin_stop_v)

        self.spin_step_v = QtWidgets.QDoubleSpinBox()
        self.spin_step_v.setRange(-1, 1)
        self.spin_step_v.setDecimals(3)
        self.spin_step_v.setValue(self.DEFAULT_STEP_V)
        self.spin_step_v.setSuffix(" V")
        layout.addRow("Step Size:", self.spin_step_v)

        # Unlock button
        self.unlock_btn = QtWidgets.QPushButton("Unlock Settings")
        self.unlock_btn.setCheckable(True)
        self.unlock_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.unlock_btn.setStyleSheet("""
            QPushButton { color: #0066cc; border: 1px solid #cce5ff; background: #f0f7ff; padding: 4px; font-size: 11px; }
            QPushButton:checked { background: #e6f0ff; border: 1px solid #0052a3; }
        """)
        self.unlock_btn.toggled.connect(self._toggle_inputs)
        layout.addRow(self.unlock_btn)

        self._toggle_inputs(False)
        return group

    def _create_hardware_group(self):
        """Create the hardware connection status group."""
        group = QtWidgets.QGroupBox("Hardware Connection")
        layout = QtWidgets.QHBoxLayout(group)
        layout.setContentsMargins(10, 5, 10, 5)

        self.keithley_light = self._create_light()
        self.mux_light = self._create_light()

        layout.addWidget(self.keithley_light)
        layout.addWidget(QtWidgets.QLabel("Keithley 2400"))
        layout.addStretch()
        layout.addWidget(self.mux_light)
        layout.addWidget(QtWidgets.QLabel("Multiplexer"))

        return group

    def _create_status_group(self):
        """Create the calibration status display group."""
        group = QtWidgets.QGroupBox("Calibration Status")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Results container
        result_container = QtWidgets.QWidget()
        result_container.setStyleSheet("background-color: #f1f3f5; border-radius: 5px; padding: 5px;")

        grid = QtWidgets.QGridLayout(result_container)
        grid.setContentsMargins(5, 5, 5, 5)
        grid.setSpacing(10)

        # Headers
        lbl_title_isc = QtWidgets.QLabel("Isc (Current)")
        lbl_title_jsc = QtWidgets.QLabel("Jsc (Density)")
        for label in [lbl_title_isc, lbl_title_jsc]:
            label.setStyleSheet("color: #6c757d; font-size: 10px; font-weight: bold; text-transform: uppercase;")
            label.setAlignment(QtCore.Qt.AlignCenter)

        # Measured values
        self.lbl_measured_isc = QtWidgets.QLabel("--.-- mA")
        self.lbl_measured_jsc = QtWidgets.QLabel("--.-- mA/cm²")
        for label in [self.lbl_measured_isc, self.lbl_measured_jsc]:
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("font-size: 18px; font-weight: 700; color: #212529;")

        # Target values
        self.lbl_target_isc = QtWidgets.QLabel(f"Target: {self.DEFAULT_TARGET_ISC*1000:.1f} mA")
        self.lbl_target_jsc = QtWidgets.QLabel(f"Target: {self.DEFAULT_TARGET_JSC:.1f} mA/cm²")
        for label in [self.lbl_target_isc, self.lbl_target_jsc]:
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setStyleSheet("color: #495057; font-size: 10px;")

        grid.addWidget(lbl_title_isc, 0, 0)
        grid.addWidget(lbl_title_jsc, 0, 1)
        grid.addWidget(self.lbl_measured_isc, 1, 0)
        grid.addWidget(self.lbl_measured_jsc, 1, 1)
        grid.addWidget(self.lbl_target_isc, 2, 0)
        grid.addWidget(self.lbl_target_jsc, 2, 1)

        # Status text
        self.lbl_status_text = QtWidgets.QLabel("WAITING")
        self.lbl_status_text.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_status_text.setStyleSheet("""
            background: #e9ecef;
            padding: 4px;
            border-radius: 3px;
            font-weight: bold;
            color: #495057;
            font-size: 11px;
            letter-spacing: 0.5px;
        """)

        layout.addWidget(result_container)
        layout.addWidget(self.lbl_status_text)

        return group

    def _create_actions_group(self):
        """Create the calibration actions group."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        # Channel selection
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems([f"Channel {i}" for i in range(1, 7)])
        self.channel_combo.setFixedHeight(28)

        # Run button
        self.run_button = QtWidgets.QPushButton("RUN CALIBRATION")
        self.run_button.setMinimumHeight(38)
        self.run_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("""
            QPushButton { background-color: #95a5a6; color: white; border: none; font-size: 13px; }
            QPushButton:pressed { background-color: #7f8c8d; }
        """)
        self.run_button.clicked.connect(self._on_run_clicked)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #dfe6e9; border-radius: 2px; }
            QProgressBar::chunk { background-color: #0984e3; border-radius: 2px; }
        """)

        layout.addWidget(QtWidgets.QLabel("Select Active Channel:"))
        layout.addWidget(self.channel_combo)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)

        return widget

    def _create_proceed_buttons(self):
        """Create the proceed buttons (Proceed and Skip)."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        # Proceed button
        self.proceed_button = QtWidgets.QPushButton("PROCEED TO APP ➡")
        self.proceed_button.setMinimumHeight(45)
        self.proceed_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.proceed_button.setEnabled(False)
        self.proceed_button.setStyleSheet("""
            QPushButton { background-color: #dfe6e9; color: #b2bec3; border: none; border-radius: 6px; font-size: 14px; }
            QPushButton:enabled { background-color: #00b894; color: white; }
            QPushButton:enabled:hover { background-color: #00a884; }
        """)
        self.proceed_button.clicked.connect(self._on_proceed)

        # Skip button (emergency bypass)
        self.skip_button = QtWidgets.QPushButton("SKIP TO APP (NOT RECOMMENDED)")
        self.skip_button.setMinimumHeight(45)
        self.skip_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.skip_button.setStyleSheet("""
            QPushButton {
                background-color: #fff3cd;
                color: #856404;
                border: 1px solid #ffeeba;
                border-radius: 6px;
                font-size: 13px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #ffeeba;
                border-color: #ffc107;
            }
        """)
        self.skip_button.clicked.connect(self._on_skip)

        layout.addWidget(self.proceed_button)
        layout.addWidget(self.skip_button)

        # Warning label for skip button
        warning_label = QtWidgets.QLabel("⚠️ Skipping calibration may affect measurement accuracy")
        warning_label.setStyleSheet("color: #856404; font-size: 9px; font-style: italic; text-align: center;")
        warning_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(warning_label)

        return widget

    def _create_plot_panel(self):
        """Create the plot panel for displaying I-V curves."""
        container = QtWidgets.QWidget()
        container.setStyleSheet("background: white; border-radius: 6px; border: 1px solid #e1e4e8;")
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(1, 1, 1, 1)

        self.plot_widget = PlotWidget(
            name="Calibration Curve",
            columns=["Voltage (V)", "Current (A)"],
            x_axis="Voltage (V)",
            y_axis="Current (A)"
        )
        self.plot_widget.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.plot.setLabel('left', 'Current', units='A', **{'font-size': '10pt'})
        self.plot_widget.plot.setLabel('bottom', 'Voltage', units='V', **{'font-size': '10pt'})
        self.plot_widget.plot.getAxis('left').setPen(pg.mkPen(color='#505c6e', width=1))
        self.plot_widget.plot.getAxis('bottom').setPen(pg.mkPen(color='#505c6e', width=1))

        layout.addWidget(self.plot_widget)
        return container

    @staticmethod
    def _create_light():
        """Create a status indicator light."""
        label = QtWidgets.QLabel("")
        label.setFixedSize(12, 12)
        label.setStyleSheet("border-radius: 6px; background: #bdc3c7; border: 1px solid white; box-shadow: 0 0 2px #ccc;")
        return label

    # -------------------------------------------------------------------------
    # Manager Setup
    # -------------------------------------------------------------------------

    def _setup_manager(self):
        """Initialize a fresh PyMeasure Manager."""
        if self.manager:
            try:
                if self.manager.is_running():
                    self.manager.abort()
                self.manager.finished.disconnect()
                self.manager.abort_returned.disconnect()
            except Exception:
                pass

        columns = ["Voltage (V)", "Current (A)"]

        dummy_browser = BrowserWidget(JVProcedure, ["active_channel"], columns)
        dummy_browser.hide()
        self._dummy_browser_widget = dummy_browser

        self.manager = Manager(
            [self.plot_widget],
            self._dummy_browser_widget.browser,
            log_level=logging.INFO,
            parent=self
        )
        self.manager.finished.connect(self._on_sweep_finished)
        self.manager.abort_returned.connect(self._on_abort_complete)

    # -------------------------------------------------------------------------
    # Hardware Management
    # -------------------------------------------------------------------------

    def _connect_hardware(self):
        """Attempt to connect to hardware instruments."""
        try:
            self.instrument_manager.connect_keithley(simulation=False)
            self.instrument_manager.connect_mux(simulation=False)
            logger.info("Hardware initialization attempted")
        except Exception as e:
            logger.warning(f"Hardware connection failed: {e}")
            self.instrument_manager.keithley = None
            self.instrument_manager.mux = None

        # Verify Keithley responsiveness
        if self.instrument_manager.keithley:
            try:
                self.instrument_manager.keithley.id
            except Exception:
                logger.error("Keithley detected but unresponsive")
                self.instrument_manager.keithley = None

        # Verify MUX responsiveness
        if self.instrument_manager.mux:
            try:
                if not hasattr(self.instrument_manager.mux, 'adapter') or \
                   not self.instrument_manager.mux.adapter.connection:
                    pass
            except Exception:
                logger.error("MUX unresponsive")
                self.instrument_manager.mux = None

        self._check_readiness()

    def _check_readiness(self):
        """Update UI based on hardware and checklist readiness."""
        keithley_ok = self.instrument_manager.keithley is not None
        mux_ok = self.instrument_manager.mux is not None

        # Update status lights
        k_color = '#2ecc71' if keithley_ok else '#e74c3c'
        m_color = '#2ecc71' if mux_ok else '#e74c3c'
        self.keithley_light.setStyleSheet(f"border-radius:6px; background:{k_color}; border: 1px solid white;")
        self.mux_light.setStyleSheet(f"border-radius:6px; background:{m_color}; border: 1px solid white;")

        is_running = self.manager.is_running() if self.manager else False
        is_abort_state = self.run_button.text() in ["ABORT", "ABORTING..."]

        if is_abort_state:
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: #e74c3c; color: white;")
            return

        hardware_ok = keithley_ok and mux_ok

        if hardware_ok and not is_running and self.checklist_confirmed:
            self.run_button.setEnabled(True)
            if self.run_button.text() == "RESTART":
                self.run_button.setStyleSheet("background-color: #f1c40f; color: black;")
            else:
                self.run_button.setText("RUN CALIBRATION")
                self.run_button.setStyleSheet("background-color: #0984e3; color: white;")
        else:
            self.run_button.setEnabled(False)
            if not hardware_ok:
                self.run_button.setText("Hardware Disconnected")
            elif not self.checklist_confirmed:
                self.run_button.setText("Awaiting Checklist")
            self.run_button.setStyleSheet("background-color: #95a5a6; color: white;")

    # -------------------------------------------------------------------------
    # UI Callbacks
    # -------------------------------------------------------------------------

    def _toggle_inputs(self, checked):
        """Enable or disable configuration inputs."""
        widgets = [
            self.spin_target_isc, self.spin_tolerance, self.spin_area,
            self.spin_start_v, self.spin_stop_v, self.spin_step_v
        ]
        for widget in widgets:
            widget.setReadOnly(not checked)
            widget.setEnabled(checked)
        self.unlock_btn.setText("Lock Settings" if checked else "Unlock Settings")

    def _on_run_clicked(self):
        """Handle run/abort button click."""
        if self.manager and self.manager.is_running():
            self.manager.abort()
            self.run_button.setText("ABORTING...")
            self.run_button.setEnabled(False)
        else:
            self._start_calibration()

    def _on_proceed(self):
        """Handle proceed to main application."""
        if self.manager and self.manager.is_running():
            self.manager.abort()
        self.calibration_passed.emit(self.instrument_manager)
        self.close()

    def _on_skip(self):
        """Handle skip to main application (emergency bypass)."""
        # Show confirmation dialog for skip
        reply = QtWidgets.QMessageBox.warning(
            self,
            "Skip Calibration",
            "WARNING: Skipping calibration may result in inaccurate measurements.\n\n"
            "Only use this option if the application is frozen and you need to recover data.\n\n"
            "Are you sure you want to skip calibration?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            if self.manager and self.manager.is_running():
                self.manager.abort()
            self.calibration_passed.emit(self.instrument_manager)
            self.close()

    # -------------------------------------------------------------------------
    # Calibration Execution
    # -------------------------------------------------------------------------

    def _start_calibration(self):
        """Execute the calibration measurement."""
        self._setup_manager()

        # Reset UI state
        self.proceed_button.setEnabled(False)
        self.lbl_measured_isc.setText("--.-- mA")
        self.lbl_measured_jsc.setText("--.-- mA/cm²")
        self.lbl_status_text.setText("MEASURING...")
        self.lbl_status_text.setStyleSheet("background: #d1ecf1; color: #0c5460; padding: 4px; border-radius: 3px;")

        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #e0e0e0; height: 4px; border-radius: 2px; }
            QProgressBar::chunk { background-color: #f1c40f; border-radius: 2px; }
        """)

        self.plot_widget.plot.clear()

        self.run_button.setText("ABORT")
        self.run_button.setStyleSheet("background-color: #e74c3c; color: white;")
        self.run_button.setEnabled(True)

        try:
            channel_text = self.channel_combo.currentText()
            channel_id = int(channel_text.split()[-1])
            area = self.spin_area.value()
            start_v = self.spin_start_v.value()
            stop_v = self.spin_stop_v.value()
            step_v = self.spin_step_v.value()
        except ValueError:
            return

        # Create measurement procedure
        procedure = JVProcedure(
            instrument=self.instrument_manager.keithley,
            mux=self.instrument_manager.mux,
            active_channel=channel_id,
            start_voltage=start_v,
            stop_voltage=stop_v,
            step_size=step_v,
            device_area=area,
            incident_power=100.0,
            compliance_current=0.1,
            simulation=False,
            channel1=(channel_id == 1), channel2=(channel_id == 2),
            channel3=(channel_id == 3), channel4=(channel_id == 4),
            channel5=(channel_id == 5), channel6=(channel_id == 6),
            user_name=self.username
        )

        # Create temporary file for results
        fd, temp_path = tempfile.mkstemp(prefix="calib_", suffix=".csv")
        os.close(fd)
        os.remove(temp_path)

        results = Results(procedure, temp_path)
        curve = self.plot_widget.new_curve(results, color=pg.mkColor('#0984e3'), width=2)

        browser_item = SignalBrowserItem(
            results,
            pg.intColor(0),
            progress_callback=self._update_progress_bar
        )
        browser_item.setText(0, "Data")

        experiment = Experiment(results, [curve], browser_item)
        self.manager.queue(experiment)

    def _update_progress_bar(self, value):
        """Update the progress bar value."""
        self.progress_bar.setValue(int(value))

    # -------------------------------------------------------------------------
    # Calibration Results
    # -------------------------------------------------------------------------

    def _on_sweep_finished(self, experiment):
        """Handle sweep completion and evaluate calibration."""
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #e0e0e0; height: 4px; border-radius: 2px; }
            QProgressBar::chunk { background-color: #27ae60; border-radius: 2px; }
        """)

        if self.manager.is_running():
            return

        self.run_button.setText("RESTART")
        self._check_readiness()

        try:
            results = experiment.procedure.analysis_results
            channel = int(experiment.procedure.active_channel)

            if not results or channel not in results:
                self._set_fail_state("No data", 0, 0, 0)
                return

            metrics = results[channel]
            measured_isc = metrics.get("Isc", 0.0)

            # Calculate Jsc
            area_val = self.spin_area.value()
            measured_jsc = (measured_isc / area_val) * 1000 if area_val > 0 else 0

            target_isc = self.spin_target_isc.value()
            tolerance_pct = self.spin_tolerance.value()

            diff = abs(measured_isc - target_isc)
            limit = target_isc * (tolerance_pct / 100.0)

            self.lbl_measured_isc.setText(f"{measured_isc * 1000:.2f} mA")
            self.lbl_measured_jsc.setText(f"{measured_jsc:.2f} mA/cm²")

            if diff <= limit:
                self._set_pass_state()
            else:
                self._set_fail_state("Tolerance", measured_isc, target_isc, measured_jsc)

        except Exception as e:
            logger.error(f"Calibration evaluation error: {e}")
            self._set_fail_state("Error", 0, 0, 0)

    def _on_abort_complete(self):
        """Handle abort completion."""
        self.lbl_status_text.setText("ABORTED")
        self.lbl_status_text.setStyleSheet("background: #f8d7da; color: #721c24; padding: 4px; border-radius: 3px;")
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #e0e0e0; height: 4px; border-radius: 2px; }
            QProgressBar::chunk { background-color: #e74c3c; border-radius: 2px; }
        """)
        self.run_button.setText("RESTART")
        self._check_readiness()

    def _set_pass_state(self):
        """Update UI for calibration pass."""
        self.lbl_status_text.setText("PASS")
        self.lbl_status_text.setStyleSheet("background: #d4edda; color: #155724; border: 1px solid #c3e6cb; padding: 4px; border-radius: 3px;")
        self.proceed_button.setEnabled(True)
        self.proceed_button.setText("PROCEED TO APP ➡")
        self.run_button.setText("RE-CALIBRATE")
        self.run_button.setStyleSheet("background-color: #0984e3; color: white;")

    def _set_fail_state(self, reason, measured_val, target_val, measured_jsc):
        """Update UI for calibration failure."""
        self.lbl_status_text.setText("FAIL")
        self.lbl_status_text.setStyleSheet("background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 4px; border-radius: 3px;")
        self.proceed_button.setEnabled(False)

        if target_val != 0:
            pct_diff = ((measured_val - target_val) / target_val) * 100
        else:
            pct_diff = 0.0

        if pct_diff > 0:
            hint = "Current is too HIGH → Move lamp UP."
        else:
            hint = "Current is too LOW → Move lamp DOWN."

        message = (
            f"<b>Measurement out of tolerance!</b><br><br>"
            f"Measured Isc: <b>{measured_val * 1000:.2f} mA</b><br>"
            f"Measured Jsc: <b>{measured_jsc:.2f} mA/cm²</b><br>"
            f"Target Isc: <b>{target_val * 1000:.2f} mA</b><br>"
            f"Diff: <b>{pct_diff:+.1f}%</b><br><br>"
            f"<i>Hint: {hint}</i>"
        )
        QtWidgets.QMessageBox.warning(self, "Calibration Failed", message)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def launch_checklist_dialog(self):
        """Show the checklist dialog at startup."""
        dialog = CalibrationChecklistDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.checklist_confirmed = True
        else:
            self.checklist_confirmed = False
        self._check_readiness()