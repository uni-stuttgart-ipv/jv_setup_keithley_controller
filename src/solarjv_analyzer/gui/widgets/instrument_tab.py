"""
Instrument Configuration Tab for JV Analyzer

Provides controls for Keithley 2400 instrument settings including:
- GPIB/Serial address
- Auto-calculated NPLC display
- Measurement range selection
- Sense mode (2-wire / 4-wire)
"""

from PyQt5 import QtWidgets
from solarjv_analyzer.config import GPIB_ADDRESS


class InstrumentTab(QtWidgets.QWidget):
    """
    Configuration panel for Keithley 2400 instrument settings.

    NPLC is automatically calculated from the sweep rate and displayed
    as read-only. All other parameters are user-configurable.
    """

    def __init__(self, parent=None):
        """Initialize the instrument configuration tab."""
        super().__init__(parent)
        self._calculated_nplc = 1.0
        self._layout()

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _layout(self):
        """Build the instrument settings form layout."""
        layout = QtWidgets.QFormLayout(self)

        # Instrument selection
        self.instrument_name = QtWidgets.QComboBox()
        self.instrument_name.addItem("Keithley 2400")
        layout.addRow("Instrument:", self.instrument_name)

        # Communication address
        self.gpib_address = QtWidgets.QLineEdit(GPIB_ADDRESS)
        layout.addRow("GPIB Address:", self.gpib_address)

        # NPLC display (read-only, auto-calculated)
        self.nplc_display = QtWidgets.QLineEdit("1.0")
        self.nplc_display.setReadOnly(True)
        self.nplc_display.setStyleSheet("background-color: #f0f0f0; color: #333;")

        nplc_helper = QtWidgets.QLabel("(Auto-calculated from sweep rate)")
        nplc_helper.setStyleSheet("color: gray; font-size: 8pt;")

        nplc_widget = QtWidgets.QWidget()
        nplc_layout = QtWidgets.QVBoxLayout(nplc_widget)
        nplc_layout.setContentsMargins(0, 0, 0, 0)
        nplc_layout.setSpacing(2)
        nplc_layout.addWidget(self.nplc_display)
        nplc_layout.addWidget(nplc_helper)

        layout.addRow("NPLC (calculated):", nplc_widget)

        # Measurement range
        self.measurement_range = QtWidgets.QComboBox()
        self.measurement_range.addItems(["Auto", "1 A", "100 mA", "10 mA", "1 mA", "100 uA"])
        layout.addRow("Measurement Range:", self.measurement_range)

        # Sense mode (2-wire vs 4-wire)
        self.sense_mode = QtWidgets.QComboBox()
        self.sense_mode.addItems(["2-wire", "4-wire"])
        layout.addRow("Sense Mode:", self.sense_mode)

    # -------------------------------------------------------------------------
    # NPLC Management
    # -------------------------------------------------------------------------

    def update_nplc(self, calculated_nplc: float):
        """
        Update the displayed NPLC value.

        Args:
            calculated_nplc: NPLC value calculated from sweep rate
        """
        self._calculated_nplc = calculated_nplc
        self.nplc_display.setText(f"{calculated_nplc:.3f}")

    # -------------------------------------------------------------------------
    # Parameter Retrieval
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """
        Get current instrument configuration.

        Returns:
            Dictionary containing:
            - gpib_address: Communication address
            - nplc: Auto-calculated NPLC value
            - measurement_range: Selected current range
            - sense_mode: Selected sense mode (2-wire/4-wire)
        """
        return {
            'gpib_address': self.gpib_address.text(),
            'nplc': self._calculated_nplc,
            'measurement_range': self.measurement_range.currentText(),
            'sense_mode': self.sense_mode.currentText(),
        }