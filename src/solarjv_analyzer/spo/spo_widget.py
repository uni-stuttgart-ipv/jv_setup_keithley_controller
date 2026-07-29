"""
SPO (Set-Point Operation) Widgets

Two widgets are defined here:
    SpoParameterTab - the SPO configuration fields (Hold Duration, Sampling
        Interval, Pre-conditioning, Channel, Vmax, Quick JV). This is
        swapped into the sidebar's existing "Parameters" tab slot in place
        of the JV ParameterTab while SPO mode is active.
    SpoWidget - the live Power-vs-Time plot and live metrics, which
        replaces the JV plot/browser/analysis display area while SPO mode
        is active.

Both follow the same modern, flat visual style (QGroupBox cards / plain
form layouts, same color palette) already defined by JVAnalyzerWindow's
global stylesheet, which cascades down to these widgets automatically.
"""

import logging
import os
import shutil

from typing import List

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.spo.spo_procedure import SpoWorker
from solarjv_analyzer.gui.widgets.toggle_switch import ToggleSwitch
from solarjv_analyzer.gui.widgets.channel_pinout import build_pinout_label

logger = logging.getLogger(__name__)


class SpoParameterTab(QtWidgets.QWidget):
    """
    SPO configuration fields, designed to occupy the sidebar's
    "Parameters" tab slot in place of the JV ParameterTab while SPO mode
    is active. Mirrors the existing ParameterTab's plain, borderless
    form-layout style (no extra QGroupBox chrome needed since it already
    lives inside a QTabWidget page).
    """

    # Emitted whenever the Vmax field becomes valid/invalid, so the window
    # can enable/disable the "Start SPO" button accordingly.
    vmax_ready = QtCore.pyqtSignal(bool)

    # Fixed, fast single-sweep parameters used only to locate Vmax quickly.
    QUICK_JV_START = 1.0
    QUICK_JV_STOP = -0.1
    QUICK_JV_STEP = 0.02
    QUICK_JV_RATE = 1.0  # V/s

    def __init__(self, main_window, parent=None):
        """
        Args:
            main_window: The JVAnalyzerWindow instance, used to reach the
                shared InstrumentManager and parameter tabs for Quick JV.
        """
        super().__init__(parent)
        self.main_window = main_window
        self._quick_jv_worker = None
        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(10)

        self.hold_duration_input = QtWidgets.QLineEdit("300")
        layout.addRow("Hold Duration (s):", self.hold_duration_input)

        self.sampling_interval_input = QtWidgets.QLineEdit("1.0")
        layout.addRow("Sampling Interval (s):", self.sampling_interval_input)

        self.preconditioning_input = QtWidgets.QLineEdit("5.0")
        layout.addRow("Pre-conditioning (s):", self.preconditioning_input)

        self._create_channel_selector(layout)

        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(separator)

        self.vmax_input = QtWidgets.QLineEdit()
        self.vmax_input.setPlaceholderText("Run Quick JV or enter manually (V)")
        self.vmax_input.textChanged.connect(self._on_vmax_changed)
        layout.addRow("Vmax (V):", self.vmax_input)

        self.quick_jv_button = QtWidgets.QPushButton("Run Quick JV")
        self.quick_jv_button.setObjectName("QueueButton")
        self.quick_jv_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.quick_jv_button.clicked.connect(self._run_quick_jv)
        layout.addRow("", self.quick_jv_button)

        helper_text = QtWidgets.QLabel(
            "Quick JV runs a fast single sweep to auto-fill Vmax."
        )
        helper_text.setStyleSheet("color: gray; font-size: 8pt;")
        helper_text.setWordWrap(True)
        layout.addRow("", helper_text)

    def _create_channel_selector(self, parent_layout):
        """
        Create the SPO "Channel Selection" card using the same reference
        pinout image + toggle-switch style as the main window's
        ParameterTab, but as a single-selection group (SPO holds only one
        channel at a time) — no "Select All" toggle.

        Layout (physical), matching ParameterTab / channel_pinout.png:
            Ch3   Ch4
            Ch2   Ch5
            Ch1   Ch6
        """
        self.channels: List[ToggleSwitch] = []
        self.channel_number_labels: List[QtWidgets.QLabel] = []

        card = QtWidgets.QGroupBox("Channel Selection")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setSpacing(10)

        body_layout = QtWidgets.QHBoxLayout()
        body_layout.setSpacing(16)

        pinout_column = QtWidgets.QVBoxLayout()
        pinout_column.addWidget(build_pinout_label())
        caption = QtWidgets.QLabel("Reference Pinout")
        caption.setAlignment(QtCore.Qt.AlignCenter)
        caption.setStyleSheet("color: #64748b; font-size: 8pt;")
        pinout_column.addWidget(caption)
        body_layout.addLayout(pinout_column)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        body_layout.addWidget(divider)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        mapping = {
            3: (0, 0), 4: (0, 1),
            2: (1, 0), 5: (1, 1),
            1: (2, 0), 6: (2, 1),
        }

        self.channel_button_group = QtWidgets.QButtonGroup(self)
        self.channel_button_group.setExclusive(True)

        for i in range(1, 7):
            number_label = QtWidgets.QLabel(str(i))
            number_label.setFixedSize(28, 28)
            number_label.setAlignment(QtCore.Qt.AlignCenter)
            self.channel_number_labels.append(number_label)

            toggle = ToggleSwitch()
            self.channels.append(toggle)
            self.channel_button_group.addButton(toggle, i)

        for ch_num, (row, col) in mapping.items():
            idx = ch_num - 1
            pair_layout = QtWidgets.QHBoxLayout()
            pair_layout.setSpacing(8)
            pair_layout.addWidget(self.channel_number_labels[idx])
            pair_layout.addWidget(self.channels[idx])
            grid.addLayout(pair_layout, row, col)

        body_layout.addLayout(grid)
        body_layout.addStretch(1)
        card_layout.addLayout(body_layout)

        parent_layout.addRow(card)

        # Default to Channel 1 selected, matching the SPO procedure default.
        self.channels[0].setChecked(True)
        self._update_channel_number_chip(0, True)

        for idx, toggle in enumerate(self.channels):
            toggle.toggled.connect(lambda checked, i=idx: self._update_channel_number_chip(i, checked))

    def _update_channel_number_chip(self, index: int, checked: bool):
        """Style the channel number chip: soft green background when the
        channel is selected, plain/muted when it isn't."""
        label = self.channel_number_labels[index]
        if checked:
            label.setStyleSheet(
                "background-color: #dcfce7; color: #16a34a; font-weight: 600;"
                "border-radius: 14px;"
            )
        else:
            label.setStyleSheet(
                "background-color: transparent; color: #94a3b8; font-weight: 600;"
            )

    def get_selected_channel(self) -> int:
        """Get the currently selected channel number (1-based)."""
        checked_id = self.channel_button_group.checkedId()
        return checked_id if checked_id != -1 else 1

    # -------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """Collect SPO-specific parameters from the configuration fields."""
        return {
            'hold_voltage': float(self.vmax_input.text() or 0.0),
            'hold_duration': float(self.hold_duration_input.text() or 300.0),
            'sampling_interval': float(self.sampling_interval_input.text() or 1.0),
            'preconditioning_time': float(self.preconditioning_input.text() or 5.0),
            'active_channel': self.get_selected_channel(),
        }

    def has_valid_hold_voltage(self) -> bool:
        """True if the Vmax field currently holds a parsable float."""
        try:
            float(self.vmax_input.text())
            return True
        except (TypeError, ValueError):
            return False

    def _on_vmax_changed(self, _text):
        self.vmax_ready.emit(self.has_valid_hold_voltage())

    def set_config_enabled(self, enabled: bool):
        """Enable/disable all configuration inputs (locked while running)."""
        for widget in (
            self.hold_duration_input, self.sampling_interval_input,
            self.preconditioning_input, self.vmax_input, self.quick_jv_button,
            *self.channels,
        ):
            widget.setEnabled(enabled)

    # -------------------------------------------------------------------
    # Quick JV (temporary, in-memory sweep solely to locate Vmax)
    # -------------------------------------------------------------------

    def _run_quick_jv(self):
        instrument_manager = self.main_window.instrument_manager
        try:
            instrument_manager.connect_keithley(simulation=False)
            instrument_manager.connect_mux(simulation=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Connection Failed",
                f"Could not connect to instruments:\n{e}\n\n"
                "Please check that the Keithley is powered on and connected."
            )
            return
        self.main_window.update_instrument_lights()

        channel = self.get_selected_channel()
        device_area = 0.089
        try:
            device_area = float(
                self.main_window.params_tab.get_parameters().get('device_area', device_area)
            )
        except Exception:
            pass

        # In-memory only: no `results` object is attached, so JVProcedure
        # never writes a file. This cannot interfere with the JV analysis
        # panel or browser.
        proc = JVProcedure(
            instrument=instrument_manager.keithley,
            mux=instrument_manager.mux,
            manager=instrument_manager,
            simulation=False,
            active_channel=channel,
            start_voltage=self.QUICK_JV_START,
            stop_voltage=self.QUICK_JV_STOP,
            step_size=self.QUICK_JV_STEP,
            sweep_rate=self.QUICK_JV_RATE,
            single_sweep_mode=True,
            sweep_direction="Forward",
            device_area=device_area,
            incident_power=100.0,
            check_errors_between_points=False,
        )

        self.quick_jv_button.setEnabled(False)
        self.quick_jv_button.setText("Running Quick JV...")

        self._quick_jv_worker = SpoWorker(proc)
        self._quick_jv_worker.run_finished.connect(self._on_quick_jv_finished)
        self._quick_jv_worker.run_failed.connect(self._on_quick_jv_failed)
        self._quick_jv_worker.start()

    def _on_quick_jv_finished(self, proc):
        self._reset_quick_jv_button()
        self._disconnect_quick_jv_instruments()

        try:
            channel = self.get_selected_channel()
            channel_results = proc.analysis_results.get(channel, {})
            metrics = channel_results.get("Forward")
            if metrics:
                vmax_v = metrics.get("Vmax", 0.0) / 1000.0  # stored in mV
                self.vmax_input.setText(f"{vmax_v:.4f}")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Quick JV", "No analysis result was produced."
                )
        except Exception as e:
            logger.error(f"Quick JV post-processing failed: {e}")
            QtWidgets.QMessageBox.warning(self, "Quick JV", f"Failed to read Vmax: {e}")

    def _on_quick_jv_failed(self, message):
        self._reset_quick_jv_button()
        self._disconnect_quick_jv_instruments()
        QtWidgets.QMessageBox.warning(self, "Quick JV Failed", message)

    def _reset_quick_jv_button(self):
        self.quick_jv_button.setEnabled(True)
        self.quick_jv_button.setText("Run Quick JV")

    def _disconnect_quick_jv_instruments(self):
        try:
            self.main_window.instrument_manager.disconnect_keithley()
            self.main_window.instrument_manager.disconnect_mux()
        except Exception:
            pass
        finally:
            self.main_window.update_instrument_lights()


class SpoWidget(QtWidgets.QWidget):
    """
    Live plot and live metrics for an SPO stability test. Replaces the JV
    plot/browser/analysis display area while SPO mode is active. The
    configuration fields themselves live in `SpoParameterTab`, which is
    swapped into the sidebar's "Parameters" tab instead of being duplicated
    here.

    Public methods (called by the main window / app controller):
        set_mode_spo(), start_spo(), abort_spo(), update_plot(),
        update_metrics(), on_spo_finished(), save_report()
    """

    # Re-emitted from param_tab so existing connections (main window) don't
    # need to reach into the parameter tab directly.
    vmax_ready = QtCore.pyqtSignal(bool)

    def __init__(self, main_window, param_tab, parent=None):
        """
        Args:
            main_window: The JVAnalyzerWindow instance.
            param_tab: The SpoParameterTab instance living in the sidebar's
                "Parameters" tab, used to read config and lock it while running.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.param_tab = param_tab
        self.param_tab.vmax_ready.connect(self.vmax_ready.emit)

        self._running = False
        self._times = []
        self._powers = []
        self._report_path = None

        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        layout.addWidget(self._build_plot_card(), stretch=1)
        layout.addWidget(self._build_metrics_card())

    def _build_plot_card(self):
        container = QtWidgets.QGroupBox("Live Power vs Time")
        layout = QtWidgets.QVBoxLayout(container)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Power', units='W')
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen(color='#2563eb', width=2))

        layout.addWidget(self.plot_widget)
        return container

    def _build_metrics_card(self):
        group = QtWidgets.QGroupBox("Live Metrics")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        self.mean_power_label = QtWidgets.QLabel("--")
        self.drift_label = QtWidgets.QLabel("--")
        self.elapsed_label = QtWidgets.QLabel("--")
        self.status_label = QtWidgets.QLabel("Idle")

        grid.addWidget(QtWidgets.QLabel("Mean Power"), 0, 0)
        grid.addWidget(self.mean_power_label, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Drift %"), 0, 2)
        grid.addWidget(self.drift_label, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Elapsed Time"), 1, 0)
        grid.addWidget(self.elapsed_label, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Status"), 1, 2)
        grid.addWidget(self.status_label, 1, 3)

        self.save_report_button = QtWidgets.QPushButton("Save Report")
        self.save_report_button.setObjectName("SavePlotButton")
        self.save_report_button.setEnabled(False)
        self.save_report_button.clicked.connect(self.save_report)
        grid.addWidget(self.save_report_button, 2, 0, 1, 4)

        return group

    # -------------------------------------------------------------------
    # Parameters (delegated to the SpoParameterTab in the sidebar)
    # -------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """Collect SPO-specific parameters from the sidebar's config fields."""
        return self.param_tab.get_parameters()

    def has_valid_hold_voltage(self) -> bool:
        """True if the Vmax field currently holds a parsable float."""
        return self.param_tab.has_valid_hold_voltage()

    # -------------------------------------------------------------------
    # Public API used by the main window / app controller
    # -------------------------------------------------------------------

    def set_mode_spo(self):
        """Called when the window switches into SPO mode."""
        if not self._running:
            self.status_label.setText("Idle")

    def start_spo(self):
        """Reset live views to a running state. Called when an SPO run starts."""
        self._running = True
        self._times = []
        self._powers = []
        self._report_path = None
        self.curve.setData([], [])
        self.mean_power_label.setText("--")
        self.drift_label.setText("--")
        self.elapsed_label.setText("0 s")
        self.status_label.setText("Running")
        self.save_report_button.setEnabled(False)
        self.param_tab.set_config_enabled(False)

    def abort_spo(self):
        """Called immediately after an abort has been requested."""
        self.status_label.setText("Aborting...")

    def update_plot(self, elapsed_s: float, power_w: float):
        """Append one point to the live Power vs Time curve."""
        self._times.append(elapsed_s)
        self._powers.append(power_w)
        self.curve.setData(self._times, self._powers)
        self.elapsed_label.setText(f"{elapsed_s:.1f} s")

    def update_metrics(self, metrics: dict):
        """Refresh the live metrics labels from a partial or final metrics dict."""
        self.mean_power_label.setText(f"{metrics.get('mean_power_mw', 0.0):.3f} mW")
        self.drift_label.setText(f"{metrics.get('drift_percent', 0.0):.2f} %")

    def on_spo_finished(self, metrics: dict, report_path: str = None):
        """Called once the run (completed or aborted) has fully stopped and
        a formatted report has been generated (report_path may be None if
        no data was collected, e.g. aborted during pre-conditioning)."""
        self._running = False
        self._report_path = report_path
        self.update_metrics(metrics)
        self.status_label.setText("Complete" if report_path else "Aborted (no data)")
        self.save_report_button.setEnabled(bool(report_path))
        self.param_tab.set_config_enabled(True)

    def save_report(self):
        """Let the user copy the already-generated formatted report elsewhere."""
        if not self._report_path or not os.path.exists(self._report_path):
            QtWidgets.QMessageBox.warning(self, "No Report", "No SPO report is available yet.")
            return

        default_name = os.path.basename(self._report_path)
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save SPO Report", default_name, "CSV Files (*.csv)"
        )
        if filename:
            try:
                shutil.copyfile(self._report_path, filename)
                QtWidgets.QMessageBox.information(self, "Saved", f"Report saved to:\n{filename}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Save Failed", str(e))
