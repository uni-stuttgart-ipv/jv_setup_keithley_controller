"""
SolarJV Analyzer – Application entry point.

Flow:
    Login → Calibration → Main Analyzer Window

Logout from any window returns to the login screen.
"""

import sys
import os

# Safety: ensure the project root is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from PyQt5.QtWidgets import QApplication

from solarjv_analyzer.auth import init_db, show_login_dialog
from solarjv_analyzer.auth.session import SessionManager, logout as auth_logout
from solarjv_analyzer.windows.calibration_window import CalibrationWindow
from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
from solarjv_analyzer.gui.style import DIALOG_STYLESHEET


def main():
    # 1. Initialise the SQLite user database (once per launch)
    init_db()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Apply a shared modern style to ALL popups/dialogs app-wide (QMessageBox,
    # QInputDialog, QDialog, ...) so they match the main/calibration window
    # look regardless of which window spawns them. Windows still set their
    # own full stylesheet on top of this for their own widgets.
    app.setStyleSheet(DIALOG_STYLESHEET)

    # Instrument manager may be reused across sessions
    instrument_manager = None
    relogin = True

    while relogin:
        relogin = False

        # 2. Show the login dialog (modal)
        username = show_login_dialog()
        if username is None:
            sys.exit(0)      # user closed the dialog without logging in

        # 3. Calibration window
        calib_window = CalibrationWindow(username)

        # Reuse instruments from previous session if available
        if instrument_manager and calib_window.instrument_manager:
            if instrument_manager.keithley:
                calib_window.instrument_manager.keithley = instrument_manager.keithley
            if instrument_manager.mux:
                calib_window.instrument_manager.mux = instrument_manager.mux

        main_window = None

        def launch_main_app(data):
            """Callback: runs when calibration passes or skip is clicked."""
            nonlocal main_window

            # Extract instrument manager and output directory
            instr = data.get('instrument_manager') if isinstance(data, dict) else data
            out_dir = data.get('output_directory') if isinstance(data, dict) else None

            main_window = JVAnalyzerWindow(username)

            # Configure directory manager
            if hasattr(main_window, 'dir_manager'):
                main_window.dir_manager.set_username(username)
                if out_dir:
                    main_window.dir_manager.set_base_directory(out_dir)
                else:
                    saved = main_window.dir_manager.get_user_selected_base()
                    if saved:
                        main_window.dir_manager.set_base_directory(saved)

            # Pass connected instruments to avoid reconnection
            if hasattr(main_window, 'instrument_manager'):
                if instr and instr.keithley:
                    main_window.instrument_manager.keithley = instr.keithley
                if instr and instr.mux:
                    main_window.instrument_manager.mux = instr.mux
                if hasattr(main_window, 'update_instrument_lights'):
                    main_window.update_instrument_lights()

            # Connect logout signal from main window
            main_window.logged_out.connect(lambda: handle_logout(main_window))

            main_window.show()
            calib_window.close()

        # Connect calibration window signals
        calib_window.calibration_passed.connect(launch_main_app)
        calib_window.logged_out.connect(lambda: handle_logout(calib_window))

        calib_window.show()
        app.exec_()

        # After the event loop ends, check if we need to relogin
        if SessionManager.current_user is None:
            relogin = True

        # Clean up main window if still visible
        if main_window and main_window.isVisible():
            main_window.close()

    sys.exit(0)


def handle_logout(window):
    """
    Perform logout from any window:
    1. End session (stops log file, clears user)
    2. Disconnect instruments
    3. Close the window
    4. Return to login loop
    """
    auth_logout(window.instrument_manager if hasattr(window, 'instrument_manager') else None)
    window.close()
    # Flag for main loop: no current user → relogin
    SessionManager.current_user = None


if __name__ == "__main__":
    main()