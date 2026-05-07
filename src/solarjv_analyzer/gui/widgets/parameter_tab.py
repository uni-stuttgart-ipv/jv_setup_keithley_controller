"""
Parameters Tab for JV Measurement Configuration

Provides input controls for sweep parameters including voltage range,
step size, sweep rate, compliance current, device area, and channel selection.
"""

from typing import List

from PyQt5 import QtWidgets, QtCore


class ParameterTab(QtWidgets.QWidget):
    """
    Configuration panel for J-V sweep parameters.

    Handles unit conversion for:
    - Voltage (V, mV)
    - Sweep rate (V/s, mV/s)
    - Current (A, mA, uA)
    - Area (cm², mm², m²)
    """

    def __init__(self, parent=None):
        """Initialize the parameters tab."""
        super().__init__(parent)
        self._layout()
        self._connect_signals()

        # Update time estimate after UI is ready
        QtCore.QTimer.singleShot(100, self._update_time_estimate)

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _layout(self):
        """Build the parameters form layout."""
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(1)

        # Start Voltage
        self.start_voltage = QtWidgets.QLineEdit("1.2")
        self.start_unit = QtWidgets.QComboBox()
        self.start_unit.addItems(["V", "mV"])
        layout.addRow("Start Voltage:", self._row(self.start_voltage, self.start_unit))

        # Stop Voltage
        self.stop_voltage = QtWidgets.QLineEdit("-0.200")
        self.stop_unit = QtWidgets.QComboBox()
        self.stop_unit.addItems(["V", "mV"])
        layout.addRow("Stop Voltage:", self._row(self.stop_voltage, self.stop_unit))

        # Step Size
        self.step_size = QtWidgets.QLineEdit("-10")
        self.step_unit = QtWidgets.QComboBox()
        self.step_unit.addItems(["V", "mV"])
        self.step_unit.setCurrentText("mV")
        layout.addRow("Step Size:", self._row(self.step_size, self.step_unit))

        # Sweep Rate
        self.sweep_rate = QtWidgets.QLineEdit("0.1")
        self.sweep_rate_unit = QtWidgets.QComboBox()
        self.sweep_rate_unit.addItems(["V/s", "mV/s"])
        layout.addRow("Sweep Rate:", self._row(self.sweep_rate, self.sweep_rate_unit))

        helper_text = QtWidgets.QLabel("(Determines measurement speed and NPLC)")
        helper_text.setStyleSheet("color: gray; font-size: 8pt;")
        layout.addRow("", helper_text)

        # Compliance Current
        self.compliance_current = QtWidgets.QLineEdit("180")
        self.comp_unit = QtWidgets.QComboBox()
        self.comp_unit.addItems(["A", "mA", "uA"])
        self.comp_unit.setCurrentText("mA")
        layout.addRow("Compliance Current:", self._row(self.compliance_current, self.comp_unit))

        # Device Area
        self.device_area = QtWidgets.QLineEdit("0.089")
        self.area_unit = QtWidgets.QComboBox()
        self.area_unit.addItems(["cm²", "mm²", "m²"])
        layout.addRow("Device Area:", self._row(self.device_area, self.area_unit))

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(separator)

        # Channel Selection
        self._create_channel_selector(layout)

        # Estimated Time Display
        self.estimated_time_label = QtWidgets.QLabel("Estimated sweep time: --")
        self.estimated_time_label.setStyleSheet("color: blue; font-weight: bold;")
        layout.addRow("", self.estimated_time_label)

        # Connect signals for real-time time estimation
        self.start_voltage.textChanged.connect(self._update_time_estimate)
        self.stop_voltage.textChanged.connect(self._update_time_estimate)
        self.sweep_rate.textChanged.connect(self._update_time_estimate)
        self.start_unit.currentTextChanged.connect(self._update_time_estimate)
        self.stop_unit.currentTextChanged.connect(self._update_time_estimate)
        self.sweep_rate_unit.currentTextChanged.connect(self._update_time_estimate)

    def _create_channel_selector(self, parent_layout):
        """
        Create the channel selection checkboxes.

        Args:
            parent_layout: QFormLayout to add the channel selector to
        """
        self.channels: List[QtWidgets.QCheckBox] = [
            QtWidgets.QCheckBox(f"Channel {i+1}") for i in range(6)
        ]
        self.select_all_channels = QtWidgets.QCheckBox("Select All Channels")

        channel_widget = QtWidgets.QWidget()
        channel_layout = QtWidgets.QVBoxLayout(channel_widget)
        channel_layout.setAlignment(QtCore.Qt.AlignRight)
        channel_layout.setSpacing(2)
        channel_layout.setContentsMargins(0, 0, 0, 0)

        for channel in self.channels:
            channel_layout.addWidget(channel)
        channel_layout.addWidget(self.select_all_channels)

        parent_layout.addRow("Channels:", channel_widget)

    @staticmethod
    def _row(input_field, combo_box) -> QtWidgets.QWidget:
        """
        Create a horizontal layout with input field and unit selector.

        Args:
            input_field: QLineEdit or similar input widget
            combo_box: QComboBox for unit selection

        Returns:
            QWidget containing the combined layout
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(input_field)
        layout.addWidget(combo_box)
        return container

    # -------------------------------------------------------------------------
    # Time Estimation
    # -------------------------------------------------------------------------

    def _update_time_estimate(self):
        """Calculate and display estimated sweep time based on current parameters."""
        try:
            # Parse start voltage
            start_v = float(self.start_voltage.text() or "0")
            if self.start_unit.currentText() == "mV":
                start_v /= 1000.0

            # Parse stop voltage
            stop_v = float(self.stop_voltage.text() or "0")
            if self.stop_unit.currentText() == "mV":
                stop_v /= 1000.0

            # Parse sweep rate
            rate = float(self.sweep_rate.text() or "0.1")
            if self.sweep_rate_unit.currentText() == "mV/s":
                rate /= 1000.0

            if rate > 0:
                voltage_range = abs(stop_v - start_v)
                time_seconds = voltage_range / rate

                if time_seconds < 60:
                    self.estimated_time_label.setText(
                        f"Estimated sweep time: {time_seconds:.1f} seconds"
                    )
                else:
                    minutes = time_seconds / 60
                    self.estimated_time_label.setText(
                        f"Estimated sweep time: {minutes:.1f} minutes ({time_seconds:.0f} seconds)"
                    )
            else:
                self.estimated_time_label.setText("Estimated sweep time: --")
        except Exception:
            self.estimated_time_label.setText("Estimated sweep time: --")

    # -------------------------------------------------------------------------
    # Parameter Retrieval
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """
        Retrieve sweep parameters converted to standard units.

        Standard units:
        - Voltage: V
        - Sweep rate: V/s
        - Current: A
        - Area: cm²

        Returns:
            Dictionary of parameter names and values
        """
        # Start Voltage -> V
        start_v = float(self.start_voltage.text())
        if self.start_unit.currentText() == "mV":
            start_v *= 1e-3

        # Stop Voltage -> V
        stop_v = float(self.stop_voltage.text())
        if self.stop_unit.currentText() == "mV":
            stop_v *= 1e-3

        # Step Size -> V
        step_v = float(self.step_size.text())
        if self.step_unit.currentText() == "mV":
            step_v *= 1e-3

        # Sweep Rate -> V/s
        sweep_rate = float(self.sweep_rate.text())
        if self.sweep_rate_unit.currentText() == "mV/s":
            sweep_rate /= 1000.0

        # Compliance Current -> A
        compliance = float(self.compliance_current.text())
        comp_unit = self.comp_unit.currentText()
        if comp_unit == "mA":
            compliance *= 1e-3
        elif comp_unit == "uA":
            compliance *= 1e-6

        # Device Area -> cm²
        area = float(self.device_area.text())
        area_unit = self.area_unit.currentText()
        if area_unit == "mm²":
            area /= 100.0
        elif area_unit == "m²":
            area *= 10000.0

        return {
            'start_voltage': start_v,
            'stop_voltage': stop_v,
            'step_size': step_v,
            'sweep_rate': sweep_rate,
            'compliance_current': compliance,
            'device_area': area,
        }

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.select_all_channels.toggled.connect(self.on_select_all_channels)

    def on_select_all_channels(self, checked: bool):
        """
        Select or deselect all channel checkboxes.

        Args:
            checked: True to select all, False to deselect all
        """
        for channel in self.channels:
            channel.setChecked(checked)

    # -------------------------------------------------------------------------
    # Channel Selection
    # -------------------------------------------------------------------------

    def get_selected_channels(self) -> List[int]:
        """
        Get the list of selected channel numbers.

        Returns:
            List of channel numbers (1-based) that are checked
        """
        return [i for i, channel in enumerate(self.channels, start=1) if channel.isChecked()]