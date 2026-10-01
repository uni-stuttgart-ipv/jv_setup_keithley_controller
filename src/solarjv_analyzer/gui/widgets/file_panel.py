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

        # Filename input — starts empty; user must enter a name before running
        self.filename_input = QtWidgets.QLineEdit("")
        self.filename_input.setPlaceholderText("Enter experiment filename…")
        layout.addRow("Filename Prefix:", self.filename_input)

        # Validation hint (hidden once a valid filename is entered)
        self.filename_hint = QtWidgets.QLabel(
            "Please enter an experiment filename before running."
        )
        self.filename_hint.setStyleSheet(
            "color: #ef4444; font-size: 11px; font-style: italic;"
            " background: transparent; padding: 2px 0;"
        )
        layout.addRow("", self.filename_hint)

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
        # Start where the reports are, not in the staging path shown in the
        # field (and never in os.getcwd(), which for an installed app is
        # wherever the shortcut happened to point).
        start_dir = ""
        getter = self.dialog_start_dir_getter
        if callable(getter):
            try:
                start_dir = getter() or ""
            except Exception:
                start_dir = ""
        if not start_dir:
            start_dir = self.directory_input.text().strip() or os.path.expanduser("~")
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start_dir
        )
        if selected:
            self.directory_input.setText(selected)

    # Set by the window to `DirectoryManager.store_destination`. The field
    # above shows the STAGING path — where files are written before the
    # publisher moves them — so opening it sent the operator to a folder that
    # empties itself. This asks where the finished reports actually are.
    store_destination_getter = None

    # Set by the window to `DirectoryManager.dialog_start_dir`, so Browse
    # opens where the reports are rather than in the staging folder.
    dialog_start_dir_getter = None

    def _on_open_clicked(self):
        """Open the folder the finished reports are in."""
        from solarjv_analyzer.utils.directory_manager import open_folder

        getter = self.store_destination_getter
        if callable(getter):
            try:
                destination = getter()
            except Exception:
                destination = ""
            if destination:
                open_folder(destination, self)
                return

        directory = self.directory_input.text().strip()
        if not directory:
            QtWidgets.QMessageBox.warning(
                self, "No Directory", "Please select a directory first."
            )
            return
        open_folder(directory, self)

    # -------------------------------------------------------------------------
    # Parameter Retrieval
    # -------------------------------------------------------------------------

    def has_valid_filename(self) -> bool:
        """Return True if the filename prefix is non-empty after trimming."""
        return bool(self.filename_input.text().strip())

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
        }