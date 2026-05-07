"""
Analysis Settings Tab for JV Measurement

Provides configuration options for analysis parameters including incident power,
contact threshold, 4-probe settings, and debug options.
"""

from PyQt5 import QtWidgets, QtCore


class AnalysisSettingsTab(QtWidgets.QWidget):
    """
    Configuration panel for analysis parameters.

    Handles unit conversion for:
    - Incident power (mW/cm², W/m², W/cm²)
    - Contact threshold (A, mA, uA)
    - Probe spacing (μm, mm, cm)
    - Sample thickness (nm, μm, mm)
    - 4-probe lateral factor (unitless)
    """

    def __init__(self, parent=None):
        """Initialize the analysis settings tab."""
        super().__init__(parent)
        self._layout()

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _layout(self):
        """Build the settings form layout."""
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(1)

        # Incident Power
        self.incident_power = QtWidgets.QLineEdit("100")
        self.power_unit = QtWidgets.QComboBox()
        self.power_unit.addItems(["mW/cm²", "W/m²", "W/cm²"])
        layout.addRow("Incident Power:", self._row(self.incident_power, self.power_unit))

        # Contact Threshold
        self.contact_threshold = QtWidgets.QLineEdit("0.001")
        self.threshold_unit = QtWidgets.QComboBox()
        self.threshold_unit.addItems(["A", "mA", "uA"])
        layout.addRow("Contact Threshold:", self._row(self.contact_threshold, self.threshold_unit))

        # Lateral Factor (4-probe)
        self.lateral_factor = QtWidgets.QLineEdit("1.0")
        layout.addRow("4-Probe Lateral Factor:", self.lateral_factor)

        # Probe Spacing
        self.probe_spacing = QtWidgets.QLineEdit("2290")
        self.spacing_unit = QtWidgets.QComboBox()
        self.spacing_unit.addItems(["μm", "mm", "cm"])
        layout.addRow("4-Probe Spacing:", self._row(self.probe_spacing, self.spacing_unit))

        # Sample Thickness
        self.sample_thickness = QtWidgets.QLineEdit("500")
        self.thickness_unit = QtWidgets.QComboBox()
        self.thickness_unit.addItems(["nm", "μm", "mm"])
        self.thickness_unit.setCurrentText("μm")
        layout.addRow("Sample Thickness:", self._row(self.sample_thickness, self.thickness_unit))

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addRow(separator)

        # Debug Options
        self.enable_validation = QtWidgets.QCheckBox("Enable detailed validation (debug only)")
        self.enable_validation.setChecked(False)
        layout.addRow("Debug Options:", self.enable_validation)

        helper_text = QtWidgets.QLabel("(Adds voltage sequence validation to logs)")
        helper_text.setStyleSheet("color: gray; font-size: 8pt;")
        layout.addRow("", helper_text)

    @staticmethod
    def _row(input_field, combo_box) -> QtWidgets.QWidget:
        """
        Create a horizontal layout containing an input field and unit selector.

        Args:
            input_field: QLineEdit or similar input widget
            combo_box: QComboBox for unit selection

        Returns:
            QWidget containing the combined layout
        """
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(input_field)
        layout.addWidget(combo_box)
        return container

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def connect_debug_signal(self, callback):
        """
        Connect the validation checkbox to a callback function.

        Args:
            callback: Function to call when checkbox toggled
        """
        self.enable_validation.toggled.connect(callback)

    # -------------------------------------------------------------------------
    # Parameter Retrieval
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """
        Retrieve analysis parameters converted to standard units.

        Standard units:
        - Incident power: mW/cm²
        - Contact threshold: A
        - Probe spacing: μm
        - Sample thickness: μm
        - Lateral factor: unitless

        Returns:
            Dictionary of parameter names and values
        """
        # Incident Power Conversion -> mW/cm²
        power_val = float(self.incident_power.text())
        power_unit = self.power_unit.currentText()

        if power_unit == "W/m²":
            power_val *= 0.1      # 1000 mW / 10000 cm² = 0.1
        elif power_unit == "W/cm²":
            power_val *= 1000.0   # 1 W = 1000 mW

        # Contact Threshold Conversion -> A
        threshold_val = float(self.contact_threshold.text())
        threshold_unit = self.threshold_unit.currentText()

        if threshold_unit == "mA":
            threshold_val *= 1e-3
        elif threshold_unit == "uA":
            threshold_val *= 1e-6

        # Probe Spacing Conversion -> μm
        spacing_val = float(self.probe_spacing.text())
        spacing_unit = self.spacing_unit.currentText()

        if spacing_unit == "mm":
            spacing_val *= 1000.0
        elif spacing_unit == "cm":
            spacing_val *= 10000.0

        # Sample Thickness Conversion -> μm
        thickness_val = float(self.sample_thickness.text())
        thickness_unit = self.thickness_unit.currentText()

        if thickness_unit == "nm":
            thickness_val /= 1000.0
        elif thickness_unit == "mm":
            thickness_val *= 1000.0

        return {
            'incident_power': power_val,
            'contact_threshold': threshold_val,
            'lateral_factor': float(self.lateral_factor.text()),
            'probe_spacing': spacing_val,
            'sample_thickness': thickness_val,
            'enable_validation': self.enable_validation.isChecked(),
        }