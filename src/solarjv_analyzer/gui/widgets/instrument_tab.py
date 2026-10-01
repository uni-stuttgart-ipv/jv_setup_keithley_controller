"""
Instrument Configuration Tab for JV Analyzer

Provides controls for Keithley 2400 instrument settings including:
- GPIB/Serial address
- Auto-calculated NPLC display
- Measurement range selection
- Sense mode (2-wire / 4-wire)
"""

from PyQt5 import QtCore, QtWidgets
from solarjv_analyzer.instruments.port_resolver import active_keithley_resource


class InstrumentTab(QtWidgets.QWidget):
    """
    Configuration panel for Keithley 2400 instrument settings.

    NPLC is automatically calculated from the sweep rate and displayed
    as read-only. All other parameters are user-configurable.
    """

    #: Emitted when the operator types a new VISA resource and commits it.
    #: The window applies it — this widget cannot know whether a sweep is
    #: running, and swapping the address mid-measurement must not happen.
    address_edited = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the instrument configuration tab."""
        super().__init__(parent)
        self._calculated_nplc = 1.0
        self._applied_address = active_keithley_resource()
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
        # The RESOLVED address, not config.py's: ports are decided at startup
        # and config.py ships inside the packaged app, so showing its value
        # here would display an address the app is not actually using.
        # `active_keithley_resource()` falls back to config when nothing was
        # resolved, so this is never empty.
        self.gpib_address = QtWidgets.QLineEdit(active_keithley_resource())
        self.gpib_address.setToolTip(
            "Manual backup. Type a VISA resource (e.g. ASRL3::INSTR or\n"
            "GPIB0::24::INSTR) and press Enter to use it instead of the\n"
            "address found at startup. Applies to this session only — the\n"
            "next launch goes back to auto-detection."
        )
        self.gpib_address.editingFinished.connect(self._on_address_edited)
        layout.addRow("GPIB Address:", self.gpib_address)

        # NPLC display (read-only, auto-calculated). Styled by the theme's
        # QLineEdit[readOnly="true"] rule — no inline stylesheet, which would
        # drop the monospace font and box-model and dim the value to a faint
        # gray-on-gray that reads as broken. The "(Auto-calculated from sweep
        # rate)" note lives in a tooltip rather than a helper QLabel stacked
        # below the field: the helper label added ~13px of vertical space that
        # broke the form's otherwise-uniform row spacing (the NPLC row sat
        # further from "Measurement Range" than every other adjacent pair).
        self.nplc_display = QtWidgets.QLineEdit("1.0")
        self.nplc_display.setReadOnly(True)
        self.nplc_display.setToolTip("Auto-calculated from sweep rate")

        layout.addRow("NPLC (calculated):", self.nplc_display)

        # Measurement range
        self.measurement_range = QtWidgets.QComboBox()
        self.measurement_range.addItems(["Auto", "1 A", "100 mA", "10 mA", "1 mA", "100 uA"])
        layout.addRow("Measurement Range:", self.measurement_range)

        # Sense mode (2-wire vs 4-wire)
        self.sense_mode = QtWidgets.QComboBox()
        self.sense_mode.addItems(["2-wire", "4-wire"])
        self.sense_mode.setCurrentText("4-wire")  # Kelvin remote sense: correct for solar-cell J-V
        layout.addRow("Sense Mode:", self.sense_mode)

    # -------------------------------------------------------------------------
    # Address override
    # -------------------------------------------------------------------------

    def _on_address_edited(self):
        """Announce a typed address. The window decides whether to apply it —
        it is the only thing that knows whether a measurement is running."""
        text = self.gpib_address.text().strip()
        if text != self._applied_address:
            self.address_edited.emit(text)

    def set_address(self, resource: str):
        """Show `resource` without re-emitting (used to apply or revert)."""
        self._applied_address = (resource or "").strip()
        blocked = self.gpib_address.blockSignals(True)
        self.gpib_address.setText(self._applied_address)
        self.gpib_address.blockSignals(blocked)

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