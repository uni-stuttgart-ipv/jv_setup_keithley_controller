from PyQt5 import QtWidgets, QtCore
from typing import List

class ParameterTab(QtWidgets.QWidget):
    """
    A widget for the 'Parameters' tab in the main JV analyzer window.

    This class encapsulates all input fields related to the sweep parameters
    (e.g., voltage range, step size) and the selection of measurement channels.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout()
        self._connect_signals()

    def _layout(self) -> None:
        """Initializes and arranges the widgets in a form layout."""
        layout = QtWidgets.QFormLayout(self)
        
        self.start_voltage = QtWidgets.QLineEdit("1.2")
        self.stop_voltage = QtWidgets.QLineEdit("-0.200")
        self.step_size = QtWidgets.QLineEdit("-0.010")
        self.compliance_current = QtWidgets.QLineEdit("0.18")
        self.pre_sweep_delay = QtWidgets.QLineEdit("0.0")
        self.device_area = QtWidgets.QLineEdit("0.089")

        layout.addRow("Start Voltage (V):", self.start_voltage)
        layout.addRow("Stop Voltage (V):", self.stop_voltage)
        layout.addRow("Step Size (V):", self.step_size)
        layout.addRow("Compliance Current (A):", self.compliance_current)
        layout.addRow("Pre-Sweep Delay (s):", self.pre_sweep_delay)
        layout.addRow("Device Area (cm²):", self.device_area)

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
        for ch in self.channels:
            channel_layout.addWidget(ch)
        channel_layout.addWidget(self.select_all_channels)
        
        layout.addRow("Channels:", channel_widget)

    def get_parameters(self) -> dict:
        """
        Returns the current values from the input fields as a dictionary.
        """
        return {
            'start_voltage': float(self.start_voltage.text()),
            'stop_voltage': float(self.stop_voltage.text()),
            'step_size': float(self.step_size.text()),
            'compliance_current': float(self.compliance_current.text()),
            'pre_sweep_delay': float(self.pre_sweep_delay.text()),
            'device_area': float(self.device_area.text()),
        }

    def _connect_signals(self) -> None:
        """Connects the 'Select All' checkbox to its handler."""
        self.select_all_channels.toggled.connect(self.on_select_all_channels)

    def on_select_all_channels(self, checked: bool) -> None:
        """Toggles all individual channel checkboxes.

        Args:
            checked: The new state of the 'Select All' checkbox.
        """
        for ch in self.channels:
            ch.setChecked(checked)
            
    def get_selected_channels(self) -> List[int]:
        """Returns a list of the selected channel numbers.

        Returns:
            A list of integers (e.g., [1, 3, 4]) representing the checked channels.
        """
        return [i for i, ch in enumerate(self.channels, start=1) if ch.isChecked()]