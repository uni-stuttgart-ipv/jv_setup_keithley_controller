"""A value box with a unit selector beside it.

The J-V Parameters tab pairs a QLineEdit with a QComboBox and converts to a
base unit by hand in `get_parameters()`. That works, but the conversion
tables live at the point of use, so every new field re-types them and a
typo is a wrong measurement rather than a crash. This widget keeps the
tables in one place and hands callers a number already in the base unit.

One deliberate difference from the J-V tab: changing the unit here CONVERTS
the displayed number, so the physical quantity stays what the operator
meant. On the J-V tab, switching Compliance from mA to A leaves "180" on
screen and silently means 180 A. Preserving the value is the safer default
for a field that sets a current limit through someone's cell.
"""

import logging

from PyQt5 import QtCore, QtWidgets

logger = logging.getLogger(__name__)

# name -> (base unit, {unit: factor to the base unit})
UNIT_FAMILIES = {
    "voltage":    ("V",      {"V": 1.0, "mV": 1e-3}),
    "current":    ("A",      {"A": 1.0, "mA": 1e-3, "µA": 1e-6}),
    "time_s":     ("s",      {"s": 1.0, "min": 60.0}),
    "time_min":   ("min",    {"min": 1.0, "s": 1.0 / 60.0}),
    "area":       ("cm²", {"cm²": 1.0, "mm²": 0.01, "m²": 10000.0}),
    "irradiance": ("mW/cm²", {"mW/cm²": 1.0, "W/m²": 0.1}),
    "rate":       ("V/s",    {"V/s": 1.0, "mV/s": 1e-3}),
}


class UnitField(QtWidgets.QWidget):
    """A QLineEdit plus a unit QComboBox, laid out like the J-V tab's rows."""

    value_changed = QtCore.pyqtSignal()

    def __init__(self, family: str, value: float, display_unit: str = None,
                 decimals: int = 6, parent=None):
        super().__init__(parent)
        if family not in UNIT_FAMILIES:
            raise ValueError(f"unknown unit family {family!r}")
        self.family = family
        self.base_unit, self._factors = UNIT_FAMILIES[family]
        self._decimals = decimals
        self._converting = False

        self.edit = QtWidgets.QLineEdit()
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(list(self._factors))
        if display_unit:
            self.unit.setCurrentText(display_unit)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.edit, stretch=1)
        layout.addWidget(self.unit)

        self.set_value(value)
        # Record the starting unit NOW. Relying on showEvent meant a field
        # that had never been painted could not tell which unit it was
        # converting FROM, so the first unit change silently did nothing --
        # "10 mV" became "10 V".
        self._remember_unit()
        self.unit.currentTextChanged.connect(self._on_unit_changed)
        self.edit.textChanged.connect(lambda _t: self.value_changed.emit())

    # -- reading and writing ---------------------------------------------
    def value(self, default: float = 0.0) -> float:
        """The number in the BASE unit, whatever the box is showing."""
        try:
            shown = float(self.edit.text())
        except (TypeError, ValueError):
            return default
        return shown * self._factors[self.unit.currentText()]

    def set_value(self, base_value: float) -> None:
        """Show `base_value` (given in the base unit) in the current unit."""
        try:
            shown = float(base_value) / self._factors[self.unit.currentText()]
        except (TypeError, ValueError, ZeroDivisionError):
            return
        self.edit.setText(self._format(shown))

    def text(self) -> str:
        return self.edit.text()

    def is_valid(self) -> bool:
        try:
            float(self.edit.text())
            return True
        except (TypeError, ValueError):
            return False

    def setText(self, text) -> None:                      # noqa: N802
        """Set the displayed number, in the CURRENT unit.

        A passthrough so this drops into code (and tests) that used to hold a
        plain QLineEdit here.
        """
        self.edit.setText(str(text))

    def setEnabled(self, enabled: bool) -> None:          # noqa: N802
        # Disabling the parent is enough -- Qt greys the children with it --
        # and it keeps isEnabled() honest. An earlier version forced the
        # parent to stay enabled "so the row stays laid out", which is not a
        # thing disabling affects, and it made the widget report itself as
        # enabled while its inputs were locked.
        super().setEnabled(enabled)

    # -- internals --------------------------------------------------------
    def _on_unit_changed(self, _unit_text):
        """Keep the physical quantity; restate it in the new unit."""
        if self._converting:
            return
        try:
            shown = float(self.edit.text())
        except (TypeError, ValueError):
            self.value_changed.emit()
            return
        previous = self._previous_factor()
        base = shown * previous
        self._converting = True
        try:
            self.edit.setText(
                self._format(base / self._factors[self.unit.currentText()]))
        finally:
            self._converting = False
            self._remember_unit()
        self.value_changed.emit()

    def _previous_factor(self) -> float:
        return self._factors.get(getattr(self, "_last_unit", None),
                                 self._factors[self.unit.currentText()])

    def _remember_unit(self):
        self._last_unit = self.unit.currentText()

    def _format(self, number: float) -> str:
        """Enough significant digits to survive a round trip.

        Fixed decimals lost the value on awkward factors: 5 s shown in
        minutes rounded to 0.083333, and switching back gave 4.99998 s. %g
        keeps the precision and still drops trailing zeros, so clean factors
        (1000, 60) stay clean.
        """
        text = f"{number:.10g}"
        return text if text not in ("", "-") else "0"

    def showEvent(self, event):                            # noqa: N802
        self._remember_unit()
        super().showEvent(event)
