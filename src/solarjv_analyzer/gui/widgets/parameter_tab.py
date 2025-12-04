from PyQt5 import QtWidgets, QtCore
from typing import List

class ParameterTab(QtWidgets.QWidget):
    """
    A widget for the 'Parameters' tab with comprehensive Unit Conversion.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout()
        self._connect_signals()

    def _layout(self) -> None:
        """Initializes and arranges the widgets."""
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(1) 

        # 1. Start Voltage
        self.start_voltage = QtWidgets.QLineEdit("1.2")
        self.start_unit = QtWidgets.QComboBox()
        self.start_unit.addItems(["V", "mV"])
        layout.addRow("Start Voltage:", self._row(self.start_voltage, self.start_unit))

        # 2. Stop Voltage
        self.stop_voltage = QtWidgets.QLineEdit("-0.200")
        self.stop_unit = QtWidgets.QComboBox()
        self.stop_unit.addItems(["V", "mV"])
        layout.addRow("Stop Voltage:", self._row(self.stop_voltage, self.stop_unit))

        # 3. Step Size (Default changed to -10 mV)
        self.step_size = QtWidgets.QLineEdit("-10") 
        self.step_unit = QtWidgets.QComboBox()
        self.step_unit.addItems(["V", "mV"])
        self.step_unit.setCurrentText("mV") # Set default unit
        layout.addRow("Step Size:", self._row(self.step_size, self.step_unit))

        # 4. Compliance Current (Default changed to 180 mA)
        self.compliance_current = QtWidgets.QLineEdit("180")
        self.comp_unit = QtWidgets.QComboBox()
        self.comp_unit.addItems(["A", "mA", "uA"])
        self.comp_unit.setCurrentText("mA") # Set default unit
        layout.addRow("Compliance Current:", self._row(self.compliance_current, self.comp_unit))

        # 5. Pre-Sweep Delay
        self.pre_sweep_delay = QtWidgets.QLineEdit("0.0")
        self.pre_delay_unit = QtWidgets.QComboBox()
        self.pre_delay_unit.addItems(["s", "ms"])
        layout.addRow("Pre-Sweep Delay:", self._row(self.pre_sweep_delay, self.pre_delay_unit))
        
        # 6. Delay Between Points
        self.delay_between_points = QtWidgets.QLineEdit("0.1") 
        self.point_delay_unit = QtWidgets.QComboBox()
        self.point_delay_unit.addItems(["s", "ms"])
        layout.addRow("Dwell Time:", self._row(self.delay_between_points, self.point_delay_unit))

        # 7. Device Area
        self.device_area = QtWidgets.QLineEdit("0.089")
        self.area_unit = QtWidgets.QComboBox()
        self.area_unit.addItems(["cm²", "mm²", "m²"])
        layout.addRow("Device Area:", self._row(self.device_area, self.area_unit))

        # Separator line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(line)

        # Channel selection checkboxes
        self.channels: List[QtWidgets.QCheckBox] = [
            QtWidgets.QCheckBox(f"Channel {i+1}") for i in range(6)
        ]
        self.select_all_channels = QtWidgets.QCheckBox("Select All Channels")
        
        channel_widget = QtWidgets.QWidget()
        channel_layout = QtWidgets.QVBoxLayout(channel_widget)
        channel_layout.setAlignment(QtCore.Qt.AlignRight)
        channel_layout.setSpacing(2) 
        channel_layout.setContentsMargins(0, 0, 0, 0)
        
        for ch in self.channels:
            channel_layout.addWidget(ch)
        channel_layout.addWidget(self.select_all_channels)
        
        layout.addRow("Channels:", channel_widget)

    def _row(self, input_field, combo_box) -> QtWidgets.QWidget:
        """Helper to create a layout with [Input] [Unit Dropdown]"""
        container = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(container)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(2)
        l.addWidget(input_field)
        l.addWidget(combo_box)
        return container

    def get_parameters(self) -> dict:
        """
        Returns parameters converted to STANDARD UNITS used by the backend.
        Standard Units: 
        - Voltage: V
        - Current: A
        - Time: s
        - Area: cm²
        """
        # 1. Voltage Conversions -> Target: V
        start_v = float(self.start_voltage.text())
        if self.start_unit.currentText() == "mV": start_v *= 1e-3

        stop_v = float(self.stop_voltage.text())
        if self.stop_unit.currentText() == "mV": stop_v *= 1e-3

        step_v = float(self.step_size.text())
        if self.step_unit.currentText() == "mV": step_v *= 1e-3

        # 2. Current Conversion -> Target: A
        comp_i = float(self.compliance_current.text())
        c_unit = self.comp_unit.currentText()
        if c_unit == "mA": comp_i *= 1e-3
        elif c_unit == "uA": comp_i *= 1e-6

        # 3. Time Conversions -> Target: s
        pre_delay = float(self.pre_sweep_delay.text())
        if self.pre_delay_unit.currentText() == "ms": pre_delay *= 1e-3

        point_delay = float(self.delay_between_points.text())
        if self.point_delay_unit.currentText() == "ms": point_delay *= 1e-3

        # 4. Area Conversion -> Target: cm²
        area_val = float(self.device_area.text())
        a_unit = self.area_unit.currentText()
        if a_unit == "mm²": area_val /= 100.0
        elif a_unit == "m²": area_val *= 10000.0

        return {
            'start_voltage': start_v,
            'stop_voltage': stop_v,
            'step_size': step_v,
            'compliance_current': comp_i,
            'pre_sweep_delay': pre_delay,
            'delay_between_points': point_delay,
            'device_area': area_val,
        }

    def _connect_signals(self) -> None:
        """Connects the 'Select All' checkbox to its handler."""
        self.select_all_channels.toggled.connect(self.on_select_all_channels)

    def on_select_all_channels(self, checked: bool) -> None:
        """Toggles all individual channel checkboxes."""
        for ch in self.channels:
            ch.setChecked(checked)
            
    def get_selected_channels(self) -> List[int]:
        """Returns a list of the selected channel numbers."""
        return [i for i, ch in enumerate(self.channels, start=1) if ch.isChecked()]