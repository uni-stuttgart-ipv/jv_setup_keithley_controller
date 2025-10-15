from PyQt5 import QtWidgets

class FilePanel(QtWidgets.QGroupBox):
    """
    A widget group for managing all file output settings.

    This class provides controls for specifying the output filename and
    directory, as well as options for simulation mode and how data from
    multiple channels should be saved.
    """
    def __init__(self, parent=None):
        super().__init__("File Output", parent)
        self._layout()

    def _layout(self) -> None:
        """Initializes and arranges the widgets in a form layout."""
        layout = QtWidgets.QFormLayout(self)
        
        self.filename_input = QtWidgets.QLineEdit("Output.csv")
        self.directory_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")

        # Create a horizontal layout for the directory input and browse button
        browse_layout = QtWidgets.QHBoxLayout()
        browse_layout.setContentsMargins(0, 0, 0, 0)
        browse_layout.addWidget(self.directory_input)
        browse_layout.addWidget(self.browse_button)

        layout.addRow("Filename Prefix:", self.filename_input)
        layout.addRow("Directory:", browse_layout)
        
        self.single_file_checkbox = QtWidgets.QCheckBox("Save all channels in one file")
        layout.addRow(self.single_file_checkbox)
        
        self.simulation_checkbox = QtWidgets.QCheckBox("Simulation Mode")
        layout.addRow(self.simulation_checkbox)

    def get_parameters(self) -> dict:
        """
        Returns the current values from the input fields as a dictionary.
        """
        return {
            'filename': self.filename_input.text(),
            'directory': self.directory_input.text(),
            'single_file': self.single_file_checkbox.isChecked(),
            'simulation': self.simulation_checkbox.isChecked(),
        }