"""
Session management for the JV Analyzer.

Handles:
- Storing the currently logged‑in user (in‑memory).
- Starting / stopping a per‑session log file that captures all
  application log output (calibration + main window).
"""

import logging
import os
from datetime import datetime
from typing import Optional

from .database import _get_app_data_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
class SessionManager:
    """
    Holds the current session state.

    Attributes:
        current_user (Optional[str]): The username of the logged‑in user,
                                      or None if not logged in.
        _file_handler (Optional[logging.FileHandler]): The handler that writes
                                                       logs to the session file.
    """

    current_user: Optional[str] = None
    _file_handler: Optional[logging.FileHandler] = None

    @classmethod
    def start_session(cls, username: str) -> None:
        """
        Begin a new session for the given user.

        - Stores the username.
        - Creates a timestamped log file under <AppData>/SolarJV/logs/
        - Attaches a FileHandler to the root logger so all subsequent
          log messages are written to the file.
        """
        cls.current_user = username

        log_dir = os.path.join(_get_app_data_dir(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"session_{timestamp}.log"
        file_path = os.path.join(log_dir, filename)

        handler = logging.FileHandler(file_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        cls._file_handler = handler

        logger.info(f"Session started for user '{username}'. Log file: {file_path}")

    @classmethod
    def end_session(cls) -> None:
        """
        End the current session:
        - Logs a final "session ended" message.
        - Removes the file handler from the root logger.
        - Clears the current user.
        """
        if cls._file_handler is not None:
            logger.info(f"Session ended for user '{cls.current_user}'.")
            logging.root.removeHandler(cls._file_handler)
            cls._file_handler.close()
            cls._file_handler = None
        cls.current_user = None


# ---------------------------------------------------------------------------
# Convenience functions
def get_current_user() -> Optional[str]:
    """Return the username of the currently logged‑in user, or None."""
    return SessionManager.current_user


def logout(instrument_manager=None) -> None:
    """
    Perform a full logout:

    1. Disconnect instruments (if an InstrumentManager instance is provided).
    2. End the session (stops log file recording).
    """
    if instrument_manager is not None:
        try:
            instrument_manager.disconnect_keithley()
            instrument_manager.disconnect_mux()
        except Exception:
            pass

    SessionManager.end_session()