"""
Directory Manager for Output File Storage

Manages user preferences for output directory location across the application.
Structure: Base/Username/Date/Calibration/ and Base/Username/Date/Main/
"""

import json
import os
import subprocess
import sys
from datetime import datetime

from PyQt5 import QtWidgets, QtCore


class DirectoryManager:
    """
    Manages output directory selection and persistence.

    Provides:
    - Directory selection widget for UI integration
    - Save/load directory preference to config file
    - Open folder in file explorer
    - User-based folder structure: base_dir/username/date/{Calibration|Main}/
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure consistent directory across windows."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, username=None, parent=None, mode="Main"):
        """
        Initialize the directory manager.

        Args:
            username: Logged-in username
            parent: Parent widget
            mode: 'Calibration' or 'Main' - determines which subfolder to use
        """
        if hasattr(self, '_initialized'):
            return
        self._initialized = True

        self.parent = parent
        self.username = username
        self.mode = mode  # 'Calibration' or 'Main'
        self.directory_input = None
        self.browse_button = None
        self.open_button = None
        self.hint_label = None
        self._base_root = None

    def set_mode(self, mode):
        """Set the mode ('Calibration' or 'Main')."""
        self.mode = mode
        self._update_display_directory()

    def set_username(self, username):
        """Set the current logged-in username."""
        self.username = username
        self._update_display_directory()

    def set_base_root(self, base_root):
        """Set the base root directory for all user data."""
        self._base_root = base_root
        self._update_display_directory()

    def get_base_root(self):
        """Get the base root directory."""
        if self._base_root:
            return self._base_root
        from solarjv_analyzer.config import RESULTS_ROOT
        return RESULTS_ROOT

    def _get_user_dir(self, create=False):
        """Get user's base directory (base/username)."""
        if not self.username:
            return ''
        base = self.get_base_root()
        user_dir = os.path.join(base, self.username)
        if create and not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def _get_dated_dir(self, create=False):
        """
        Get dated directory for current mode.

        Args:
            create: If True, create directories if they don't exist

        Returns:
            str: Path to base/username/date/mode/
        """
        user_dir = self._get_user_dir(create)
        if not user_dir:
            return ''

        date_str = datetime.now().strftime("%d-%m-%Y")
        dated_dir = os.path.join(user_dir, date_str, self.mode)

        if create and not os.path.exists(dated_dir):
            os.makedirs(dated_dir, exist_ok=True)

        return dated_dir

    def get_current_directory(self, create=False):
        """Get the current directory for the active mode."""
        return self._get_dated_dir(create)

    def get_calibration_dir(self, create=False):
        """Get directory for calibration data."""
        return self._get_dated_dir(create) if self.mode == "Calibration" else None

    def get_main_dir(self, create=False):
        """Get directory for main measurement data."""
        return self._get_dated_dir(create) if self.mode == "Main" else None

    def get_timestamp_filename(self, prefix="measurement", extension=".csv"):
        """Generate ISO timestamp filename."""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"{prefix}_{timestamp}{extension}"

    def get_file_path(self, prefix="measurement", create=True):
        """Get full path for a file in the current mode directory."""
        current_dir = self.get_current_directory(create)
        filename = self.get_timestamp_filename(prefix)
        return os.path.join(current_dir, filename)

    @staticmethod
    def _get_config_path():
        """Get path to user config file."""
        config_dir = os.path.join(os.path.expanduser("~"), ".solarjv")
        os.makedirs(config_dir, exist_ok=True)
        return os.path.join(config_dir, "config.json")

    def save_preference(self, base_directory):
        """Save base directory preference to config file."""
        if not base_directory:
            return
        config_path = self._get_config_path()
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        config['base_directory'] = base_directory
        with open(config_path, 'w') as f:
            json.dump(config, f)

    def load_preference(self):
        """Load base directory preference from config file."""
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get('base_directory', '')
        return ''

    def get_user_selected_base(self):
        """Get the user-selected base directory (from config)."""
        return self.load_preference()

    def _get_display_directory(self):
        """
        Get the directory to display in the input field.
        This is the full path including username and mode.
        """
        base = self.get_user_selected_base()
        if not base:
            base = self.get_base_root()

        if not self.username:
            return base

        date_str = datetime.now().strftime("%d-%m-%Y")
        return os.path.join(base, self.username, date_str, self.mode)

    def _update_display_directory(self):
        """Update the directory input field with the full path."""
        if self.directory_input:
            display_dir = self._get_display_directory()
            self.directory_input.setText(display_dir)
            self._update_hint()

    def create_directory_widget(self, title="Output Directory"):
        """
        Create a directory selection widget.

        Args:
            title: Group box title

        Returns:
            QWidget: Group box containing directory input and buttons
        """
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(group)

        # Input row
        input_layout = QtWidgets.QHBoxLayout()
        self.directory_input = QtWidgets.QLineEdit()
        self.directory_input.setPlaceholderText("Select base output directory...")
        self.directory_input.setReadOnly(True)
        self.directory_input.setStyleSheet("background-color: #f5f5f5;")

        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.browse_button.clicked.connect(self._on_browse)

        self.open_button = QtWidgets.QPushButton("Open Folder")
        self.open_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_button.clicked.connect(self._on_open)

        input_layout.addWidget(self.directory_input, stretch=1)
        input_layout.addWidget(self.browse_button)
        input_layout.addWidget(self.open_button)
        layout.addLayout(input_layout)

        # Hint label
        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setStyleSheet("color: gray; font-size: 9pt; margin-top: 4px;")
        layout.addWidget(self.hint_label)

        # Load saved preference and update display
        self._update_display_directory()

        return group

    def _update_hint(self):
        """Update the hint label showing where files will be saved."""
        if hasattr(self, 'hint_label') and self.hint_label and self.username:
            base = self.get_user_selected_base()
            if not base:
                base = self.get_base_root()
            hint = f"Files will be saved in: {base}/{self.username}/[Date]/{self.mode}/"
            self.hint_label.setText(hint)

    def get_display_directory(self):
        """Get the currently displayed directory (full path with mode)."""
        return self.directory_input.text() if self.directory_input else ''

    def get_base_directory(self):
        """Get the base directory from user preference."""
        return self.load_preference() or self.get_base_root()

    def set_base_directory(self, base_directory):
        """Set the base directory and update display."""
        if base_directory:
            self.save_preference(base_directory)
            self._update_display_directory()

    def _on_browse(self):
        """Open folder dialog to select base output directory."""
        current_base = self.get_user_selected_base() or self.get_base_root()
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent, "Select Base Output Directory", current_base
        )
        if selected:
            self.save_preference(selected)
            self._update_display_directory()

    def _on_open(self):
        """Open the current full directory in file explorer."""
        directory = self.get_current_directory(create=False)
        if not directory or not os.path.exists(directory):
            # Try to create it
            directory = self.get_current_directory(create=True)

        if not directory or not os.path.exists(directory):
            QtWidgets.QMessageBox.warning(
                self.parent, "Directory Not Found",
                f"Cannot open directory. Please select a base directory first."
            )
            return

        if sys.platform == "win32":
            os.startfile(directory)
        elif sys.platform == "darwin":
            subprocess.run(["open", directory])
        else:
            subprocess.run(["xdg-open", directory])
