from PyQt5 import QtWidgets
from solarjv_analyzer.config import GPIB_ADDRESS

class InstrumentTab(QtWidgets.QWidget):
    """
    A widget for the 'Instrument' tab in the main JV analyzer window.

    This class contains all settings specific to the source-measure unit (SMU),
    such as its address, measurement speed (NPLC), and sensor configuration.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout()

    def _layout(self) -> None:
        """Initializes and arranges the widgets in a form layout."""
        layout = QtWidgets.QFormLayout(self)
        
        self.instrument_name = QtWidgets.QComboBox()
        self.instrument_name.addItem("Keithley 2400")
        
        self.gpib_address = QtWidgets.QLineEdit(GPIB_ADDRESS)
        self.nplc = QtWidgets.QLineEdit("1")
        
        self.measurement_range = QtWidgets.QComboBox()
        self.measurement_range.addItems(["Auto", "1 A", "100 mA", "10 mA", "1 mA", "100 uA"])
        
        self.sense_mode = QtWidgets.QComboBox()
        self.sense_mode.addItems(["2-wire", "4-wire"])
        
        layout.addRow("Instrument:", self.instrument_name)
        layout.addRow("GPIB Address:", self.gpib_address)
        layout.addRow("NPLC (PLC units):", self.nplc)
        layout.addRow("Measurement Range:", self.measurement_range)
        layout.addRow("Sense Mode:", self.sense_mode)

    def get_parameters(self) -> dict:
        """
        Returns the current values from the input fields as a dictionary.
        """
        return {
            'gpib_address': self.gpib_address.text(),
            'nplc': float(self.nplc.text()),
            'measurement_range': self.measurement_range.currentText(),
            'sense_mode': self.sense_mode.currentText(),
        }