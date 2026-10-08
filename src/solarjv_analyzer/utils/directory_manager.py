"""
Directory Manager for Output File Storage

Manages user preferences for output directory location across the application.
Structure: Base/Username/Date/{Calibration,JV,SPO}/
"""

import contextlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

from PyQt5 import QtWidgets, QtCore

logger = logging.getLogger(__name__)


MODE_FOLDERS = ("Calibration", "JV", "SPO")


def ensure_day_folders(username: str = None) -> list:
    """Create today's Calibration / JV / SPO folders once, at startup.

    Called when the application opens so the day's folders exist before anyone
    goes looking for them: Open Folder then always lands somewhere real, and
    the operator can browse to the right place from Explorer without having to
    run a measurement first to bring the folder into being.

    Idempotent by construction — `exist_ok=True` means an existing folder is
    left completely untouched, so nothing is overwritten and a second launch
    on the same day does nothing at all.

    Creates them on the store (the folders people actually browse) when
    publishing is enabled, and locally otherwise. Never raises: a disconnected
    share must not stop the application from starting.

    Returns:
        list: the folders that were created or already present.
    """
    created = []
    try:
        from solarjv_analyzer import store
        if store.is_enabled():
            user, date_str = store.active_user(), store.store_date()
            for mode in MODE_FOLDERS:
                path = store.destination_dir(user, date_str, mode)
                try:
                    os.makedirs(path, exist_ok=True)
                    created.append(path)
                except OSError as exc:
                    logger.warning(f"Could not create {path}: {exc}")
            if created:
                logger.info(
                    f"Store folders ready for {date_str}: "
                    f"{', '.join(MODE_FOLDERS)}")
            return created
    except Exception as exc:
        logger.debug(f"Store folder preparation skipped: {exc}")

    if not username:
        return created
    try:
        from solarjv_analyzer.config import RESULTS_ROOT
        date_str = datetime.now().strftime("%d-%m-%Y")
        for mode in MODE_FOLDERS:
            path = os.path.join(RESULTS_ROOT, username, date_str, mode)
            try:
                os.makedirs(path, exist_ok=True)
                created.append(path)
            except OSError as exc:
                logger.warning(f"Could not create {path}: {exc}")
    except Exception as exc:
        logger.debug(f"Local folder preparation skipped: {exc}")
    return created


class DirectoryManager:
    """
    Manages output directory selection and persistence.

    Provides:
    - Directory selection widget for UI integration
    - Save/load directory preference to config file
    - Open folder in file explorer
    - User-based folder structure: base_dir/username/date/{Calibration|JV|SPO}/
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure consistent directory across windows."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, username=None, parent=None, mode="JV"):
        """
        Initialize the directory manager.

        Args:
            username: Logged-in username
            parent: Parent widget
            mode: 'Calibration', 'JV' or 'SPO' - which subfolder to use
        """
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self.directory_input = None
            self.browse_button = None
            self.open_button = None
            self.hint_label = None
            self._base_root = None
            self._widget_mode = None

        # Applied on EVERY construction, not just the first. This is a
        # singleton, so the second window to ask for one gets the first
        # window's object back — and used to get the first window's `mode`
        # with it, silently ignoring the argument it just passed. That bit
        # on the relogin loop: the main window leaves the shared mode at
        # "JV", logout tears the windows down but NOT the class attribute
        # holding the instance, so the next login's calibration window
        # constructed itself with mode="Calibration" and was handed a JV
        # manager. Its reports and its Open Folder button went to JV.
        self.parent = parent
        if username:
            self.username = username
        elif not hasattr(self, 'username'):
            self.username = None
        self.mode = mode  # 'Calibration', 'JV' or 'SPO'

    @contextlib.contextmanager
    def scoped_mode(self, mode):
        """Resolve paths as `mode` for the duration of the block.

        `self.mode` is shared process-wide, so any window that needs a folder
        other than whatever the last caller left behind must say so explicitly
        rather than trust the current value. Restores the previous mode even
        if the body raises.
        """
        previous = self.mode
        self.mode = mode or previous
        try:
            yield self
        finally:
            self.mode = previous

    def _widget_scope(self):
        """Scope for the directory widget's own buttons and labels.

        The widget belongs to one window and therefore to one folder; it must
        keep showing — and opening — that folder no matter which mode some
        other window has since switched the shared manager into.
        """
        return self.scoped_mode(self._widget_mode or self.mode)

    def set_mode(self, mode):
        """Set the mode ('Calibration', 'JV' or 'SPO')."""
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
        return self._get_dated_dir(create) if self.mode == "JV" else None

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
        destination = self.store_destination()
        if destination:
            return destination

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
            with self._widget_scope():
                display_dir = self._get_display_directory()
                self.directory_input.setText(display_dir)
                self._update_hint()

    def create_directory_widget(self, title="Output Directory", mode=None):
        """
        Create a directory selection widget.

        Args:
            title: Group box title
            mode: pin the widget to this folder ('Calibration', 'JV', 'SPO').
                Defaults to the manager's current mode. Pass it whenever the
                owning window has a fixed folder, so the display and the Open
                Folder button stay on it.

        Returns:
            QWidget: Group box containing directory input and buttons
        """
        self._widget_mode = mode or self.mode
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
            destination = self.store_destination()
            if destination:
                hint = f"Reports are saved to: {destination}"
            else:
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

    def store_destination(self) -> str:
        """Where finished reports actually end up, or "" when publishing is off.

        The local directory is STAGING: `store.attach()` repoints the base root
        at it, and the publisher then copies each finished report to the
        protected store and removes the local copy. So the path the operator
        needs — the one that still has their file in it an hour later — is the
        store path, not the working directory. Opening staging sent people to
        a folder that empties itself.

        Resolved fresh each call: the date rolls over at midnight, and a
        session left open overnight must not keep pointing at yesterday.
        """
        try:
            from solarjv_analyzer import store
            if not store.is_enabled():
                return ""
            return store.destination_dir(
                store.active_user(), store.store_date(), self.mode)
        except Exception:
            # Never let the store break plain local operation.
            return ""

    def dialog_start_dir(self, mode: str = None) -> str:
        """Where a file dialog should open: the store, else the local folder.

        Every open/save/browse dialog in the application starts here, so the
        operator always lands where the reports actually are rather than in
        the staging folder they pass through — or, worse, in whatever
        directory Qt happened to remember.

        Falls back through local -> home so a dialog always opens somewhere
        real, even with the share disconnected.
        """
        previous = self.mode
        try:
            if mode:
                self.mode = mode
            for candidate in (self.store_destination(),
                              self.get_current_directory(create=False),
                              self.get_base_root()):
                if candidate and os.path.isdir(candidate):
                    return candidate
        except Exception as exc:                      # noqa: BLE001
            logger.debug(f"Could not resolve a dialog start directory: {exc}")
        finally:
            self.mode = previous
        return os.path.expanduser("~")

    def _on_browse(self):
        """Open folder dialog to select base output directory."""
        with self._widget_scope():
            start = self.dialog_start_dir()
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self.parent, "Select Base Output Directory", start
        )
        if selected:
            self.save_preference(selected)
            self._update_display_directory()

    def _on_open(self):
        """Open the folder the finished reports are in."""
        with self._widget_scope():
            self._open_current_folder()

    def _open_current_folder(self):
        destination = self.store_destination()
        if destination:
            self._open_path(destination)
            return

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

        self._open_path(directory)

    def _open_path(self, path: str):
        """Instance shim for the module-level :func:`open_folder`."""
        open_folder(path, self.parent)


def open_folder(path: str, parent=None):
    """Open `path`, creating it or falling back to its nearest existing parent.

    The dated store folder does not exist until the day's first report is
    published, and the share allows creation — so make it rather than
    refusing. If even that fails (share offline) walk up to something that
    does exist, which is more use to the operator than an error box.

    `parent` is the Qt widget to parent the warning dialog to.
    """
    if not path:
        return
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            pass

    target = path
    while target and not os.path.isdir(target):
        # NB: not `parent` — that name is the Qt widget argument.
        above = os.path.dirname(target)
        if above == target:
            target = ""
            break
        target = above

    if not target:
        QtWidgets.QMessageBox.warning(
            parent, "Folder Unavailable",
            f"Cannot open {path}.\n\nThe share may be disconnected. "
            "Reports are still saved and will be published when it returns."
        )
        return

    try:
        if sys.platform == "win32":
            os.startfile(target)
        elif sys.platform == "darwin":
            subprocess.run(["open", target])
        else:
            subprocess.run(["xdg-open", target])
    except Exception as exc:
        logger.warning(f"Could not open {target}: {exc}")
