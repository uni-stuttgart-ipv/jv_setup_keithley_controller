from PyQt5 import QtWidgets

class AnalysisSettingsTab(QtWidgets.QWidget):
    """
    A widget for the 'Analysis' tab in the main JV analyzer window.

    This class holds input fields for parameters that are used in the
    post-measurement calculation of solar cell performance metrics.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout()

    def _layout(self) -> None:
        """Initializes and arranges the widgets in a form layout."""
        layout = QtWidgets.QFormLayout(self)
        
        self.incident_power = QtWidgets.QLineEdit("100")
        self.contact_threshold = QtWidgets.QLineEdit("0.001")
        self.lateral_factor = QtWidgets.QLineEdit("1.0")
        self.probe_spacing = QtWidgets.QLineEdit("2290")
        self.sample_thickness = QtWidgets.QLineEdit("500")

        layout.addRow("Incident Power (mW/cm²):", self.incident_power)
        layout.addRow("Contact Threshold (A):", self.contact_threshold)
        layout.addRow("4-Probe Lateral Factor (unitless):", self.lateral_factor)
        layout.addRow("4-Probe Spacing (μm):", self.probe_spacing)
        layout.addRow("Sample Thickness (μm):", self.sample_thickness)

    def get_parameters(self) -> dict:
        """
        Returns the current values from the input fields as a dictionary.
        """
        return {
            'incident_power': float(self.incident_power.text()),
            'contact_threshold': float(self.contact_threshold.text()),
            'lateral_factor': float(self.lateral_factor.text()),
            'probe_spacing': float(self.probe_spacing.text()),
            'sample_thickness': float(self.sample_thickness.text()),
        }