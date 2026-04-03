"""
File Output Panel for JV Analyzer

Provides controls for configuring output file location and name,
including directory selection, filename input, and output mode options.
"""

import os
import subprocess
import sys

from PyQt5 import QtWidgets, QtCore


class FilePanel(QtWidgets.QGroupBox):
    """
    Widget group for managing file output settings.

    Controls:
    - Filename prefix input
    - Directory selection with Browse and Open buttons
    - Single file mode toggle (all channels in one file)
    - Simulation mode toggle
    """

    def __init__(self, parent=None):
        """Initialize the file panel."""
        super().__init__("File Output", parent)
        self._layout()
        self._connect_signals()

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _layout(self):
        """Build the file output form layout."""
        layout = QtWidgets.QFormLayout(self)

        # Filename input
        self.filename_input = QtWidgets.QLineEdit("Output.csv")
        layout.addRow("Filename Prefix:", self.filename_input)

        # Directory selection with Browse and Open buttons
        self.directory_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.open_button = QtWidgets.QPushButton("Open Folder")

        # Horizontal layout for directory controls
        dir_layout = QtWidgets.QHBoxLayout()
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(5)
        dir_layout.addWidget(self.directory_input, stretch=1)
        dir_layout.addWidget(self.browse_button)
        dir_layout.addWidget(self.open_button)

        layout.addRow("Directory:", dir_layout)

        # Output mode options
        self.single_file_checkbox = QtWidgets.QCheckBox("Save all channels in one file")
        layout.addRow(self.single_file_checkbox)

        self.simulation_checkbox = QtWidgets.QCheckBox("Simulation Mode")
        layout.addRow(self.simulation_checkbox)

    def _connect_signals(self):
        """Connect UI signals to their handlers."""
        self.browse_button.clicked.connect(self._on_browse_clicked)
        self.open_button.clicked.connect(self._on_open_clicked)

    # -------------------------------------------------------------------------
    # Directory Management
    # -------------------------------------------------------------------------

    def set_directory(self, directory: str):
        """
        Set the output directory.

        Args:
            directory: Path to the output directory
        """
        self.directory_input.setText(directory)

    def get_directory(self) -> str:
        """
        Get the current output directory.

        Returns:
            Current directory path
        """
        return self.directory_input.text()

    def _on_browse_clicked(self):
        """Open folder dialog to select output directory."""
        start_dir = self.directory_input.text().strip() or os.getcwd()
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if selected:
            self.directory_input.setText(selected)

    def _on_open_clicked(self):
        """Open the selected directory in file explorer."""
        directory = self.directory_input.text().strip()

        if not directory:
            QtWidgets.QMessageBox.warning(
                self, "No Directory", "Please select a directory first."
            )
            return

        if not os.path.exists(directory):
            QtWidgets.QMessageBox.warning(
                self, "Directory Not Found",
                f"The directory does not exist:\n{directory}"
            )
            return

        # Open directory in system file explorer
        if sys.platform == "win32":
            os.startfile(directory)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", directory])
        else:  # Linux
            subprocess.run(["xdg-open", directory])

    # -------------------------------------------------------------------------
    # Parameter Retrieval
    # -------------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """
        Get current file output settings.

        Returns:
            Dictionary with filename, directory, and mode flags
        """
        return {
            'filename': self.filename_input.text(),
            'directory': self.directory_input.text(),
            'single_file': self.single_file_checkbox.isChecked(),
            'simulation': self.simulation_checkbox.isChecked(),
        }