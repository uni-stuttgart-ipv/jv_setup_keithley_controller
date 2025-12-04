from PyQt5 import QtWidgets, QtCore

class AnalysisSettingsTab(QtWidgets.QWidget):
    """
    A widget for the 'Analysis' tab with Unit Conversion logic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout()

    def _layout(self) -> None:
        """Initializes and arranges the widgets."""
        layout = QtWidgets.QFormLayout(self)
        layout.setVerticalSpacing(1)
        # --- Incident Power ---
        self.incident_power = QtWidgets.QLineEdit("100")
        self.power_unit = QtWidgets.QComboBox()
        self.power_unit.addItems(["mW/cm²", "W/m²", "W/cm²"])
        layout.addRow("Incident Power:", self._row(self.incident_power, self.power_unit))

        # --- Contact Threshold (Amps are standard, but we can add mA) ---
        self.contact_threshold = QtWidgets.QLineEdit("0.001")
        self.threshold_unit = QtWidgets.QComboBox()
        self.threshold_unit.addItems(["A", "mA", "uA"])
        layout.addRow("Contact Threshold:", self._row(self.contact_threshold, self.threshold_unit))

        # --- Lateral Factor (Unitless) ---
        self.lateral_factor = QtWidgets.QLineEdit("1.0")
        layout.addRow("4-Probe Lateral Factor:", self.lateral_factor)

        # --- Probe Spacing ---
        self.probe_spacing = QtWidgets.QLineEdit("2290")
        self.spacing_unit = QtWidgets.QComboBox()
        self.spacing_unit.addItems(["μm", "mm", "cm"])
        layout.addRow("4-Probe Spacing:", self._row(self.probe_spacing, self.spacing_unit))

        # --- Sample Thickness ---
        self.sample_thickness = QtWidgets.QLineEdit("500")
        self.thickness_unit = QtWidgets.QComboBox()
        self.thickness_unit.addItems(["nm", "μm", "mm"])
        self.thickness_unit.setCurrentText("μm") # Default
        layout.addRow("Sample Thickness:", self._row(self.sample_thickness, self.thickness_unit))

    def _row(self, input_field, combo_box) -> QtWidgets.QWidget:
        """Helper to create a layout with [Input] [Unit Dropdown]"""
        container = QtWidgets.QWidget()
        l = QtWidgets.QHBoxLayout(container)
        l.setContentsMargins(0, 0, 0, 0)
        l.addWidget(input_field)
        l.addWidget(combo_box)
        return container

    def get_parameters(self) -> dict:
        """
        Returns parameters converted to STANDARD UNITS used by analysis.py.
        Standard Units: 
        - Power: mW/cm²
        - Current: A
        - Length: μm
        """
        # 1. Power Conversion -> Target: mW/cm²
        p_val = float(self.incident_power.text())
        p_unit = self.power_unit.currentText()
        if p_unit == "W/m²":
            p_val *= 0.1      # 1000 mW / 10000 cm2 = 0.1
        elif p_unit == "W/cm²":
            p_val *= 1000.0   # 1 W = 1000 mW

        # 2. Threshold Conversion -> Target: A
        t_val = float(self.contact_threshold.text())
        t_unit = self.threshold_unit.currentText()
        if t_unit == "mA":
            t_val *= 1e-3
        elif t_unit == "uA":
            t_val *= 1e-6

        # 3. Spacing Conversion -> Target: μm
        s_val = float(self.probe_spacing.text())
        s_unit = self.spacing_unit.currentText()
        if s_unit == "mm":
            s_val *= 1000.0
        elif s_unit == "cm":
            s_val *= 10000.0

        # 4. Thickness Conversion -> Target: μm
        th_val = float(self.sample_thickness.text())
        th_unit = self.thickness_unit.currentText()
        if th_unit == "nm":
            th_val /= 1000.0
        elif th_unit == "mm":
            th_val *= 1000.0

        return {
            'incident_power': p_val,
            'contact_threshold': t_val,
            'lateral_factor': float(self.lateral_factor.text()),
            'probe_spacing': s_val,
            'sample_thickness': th_val,
        }