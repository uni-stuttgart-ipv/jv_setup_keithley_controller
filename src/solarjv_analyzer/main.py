"""
SolarJV Analyzer – Application entry point.

Flow:
    Login → Calibration → Main Analyzer Window

Logout from any window returns to the login screen.
"""

import logging
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
from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer import store


def main():
    # 1. Initialise the SQLite user database (once per launch)
    init_db()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Register the design-system fonts (Geist/Inter/JetBrains Mono, if
    # bundled in resources/fonts/) and the shared pyqtgraph defaults BEFORE
    # any window or plot is created, so every view renders identically.
    from solarjv_analyzer.gui.theme import load_design_fonts, apply_global_plot_config
    load_design_fonts()
    apply_global_plot_config()

    # Apply a shared modern style to ALL popups/dialogs app-wide (QMessageBox,
    # QInputDialog, QDialog, ...) so they match the main/calibration window
    # look regardless of which window spawns them. Windows still set their
    # own full stylesheet on top of this for their own widgets.
    app.setStyleSheet(DIALOG_STYLESHEET)

    # Resolve which COM port each instrument is on, before anything tries to
    # connect. config.py ships inside the packaged app, so a changed COM
    # number would otherwise be unfixable without a rebuild. Never fatal: on
    # failure the resolver falls back to the config values and the usual
    # "Hardware Disconnected — Retry" path still applies.
    try:
        from solarjv_analyzer.instruments import port_resolver
        port_resolver.resolve_at_startup()
    except Exception as exc:
        logging.getLogger(__name__).error(f"Port resolution failed: {exc}")

    # ONE InstrumentManager for the whole process. Every window is handed
    # this same object, so there is exactly one owner of the VISA session and
    # the serial port. (Previously each window built its own manager and the
    # instrument objects were copied between them; the calibration window's
    # closeEvent then closed the very session the main window was still
    # holding, and its connect_*() calls short-circuited on the dead handle.)
    instrument_manager = InstrumentManager()

    # Connect both instruments NOW, as the app opens. The rack is powered on
    # before the app is launched, so there is nothing to wait for — and
    # waiting was visible: the Keithley only connected when the calibration
    # window was built, and the MUX not until the first run, which is why its
    # status light sat red until someone pressed a button.
    #
    # On a background thread so the login dialog appears instantly, and joined
    # before the calibration window is built so nothing races the manager.
    import threading

    def _preconnect(manager):
        for name, connect in (("Keithley", manager.connect_keithley),
                              ("MUX", manager.connect_mux)):
            try:
                connect(simulation=False)
                logging.getLogger(__name__).info(f"{name} connected at startup")
            except Exception as exc:
                # Never fatal: the calibration window's "Hardware Disconnected
                # — Retry" path and the run-start connect both still apply.
                logging.getLogger(__name__).warning(
                    f"{name} not connected at startup: {exc}")

    preconnect = threading.Thread(
        target=_preconnect, args=(instrument_manager,),
        name="startup-connect", daemon=True,
    )
    preconnect.start()

    relogin = True

    while relogin:
        relogin = False

        # 2. Show the login dialog (modal)
        username = show_login_dialog()
        if username is None:
            sys.exit(0)      # user closed the dialog without logging in

        # Let the startup connection finish before any window touches the
        # manager. By now the operator has typed a password, so this is
        # normally instantaneous.
        if preconnect.is_alive():
            preconnect.join(timeout=20)

        # 3. Calibration window — shares the process-wide instrument manager,
        #    so any connection still open from a previous session is reused
        #    as-is rather than copied.
        calib_window = CalibrationWindow(
            username, instrument_manager=instrument_manager
        )
        # Redirect working files to local staging and start publishing finished
        # reports to the protected store. A no-op when disabled; never raises.
        store.attach(calib_window)

        main_window = None

        def launch_main_app(data):
            """Callback: runs when calibration passes or skip is clicked."""
            nonlocal main_window

            # Extract instrument manager and output directory
            instr = data.get('instrument_manager') if isinstance(data, dict) else data
            out_dir = data.get('output_directory') if isinstance(data, dict) else None

            # Hand over the manager ITSELF (normally the same process-wide
            # object the calibration window was given), never a copy of its
            # instrument references — see InstrumentManager's module docstring.
            main_window = JVAnalyzerWindow(
                username,
                instrument_manager=instr if instr is not None else instrument_manager,
            )

            # Configure directory manager
            if hasattr(main_window, 'dir_manager'):
                main_window.dir_manager.set_username(username)
                if out_dir:
                    main_window.dir_manager.set_base_directory(out_dir)
                else:
                    saved = main_window.dir_manager.get_user_selected_base()
                    if saved:
                        main_window.dir_manager.set_base_directory(saved)

            store.attach(main_window)

            # Reflect the inherited connection state in the status lights.
            if hasattr(main_window, 'update_instrument_lights'):
                main_window.update_instrument_lights()

            # Connect logout signal from main window
            main_window.logged_out.connect(lambda: handle_logout(main_window))

            main_window.showMaximized()
            # Start the live hardware monitor so both status lights are
            # honest from the moment the window appears, and stay honest —
            # the calibration gate is Keithley-only, so nothing has opened the
            # MUX yet, and a cable pulled later must turn its light red.
            if hasattr(main_window, 'start_hardware_monitor'):
                main_window.start_hardware_monitor()
            calib_window.close()

        # Connect calibration window signals
        calib_window.calibration_passed.connect(launch_main_app)
        calib_window.logged_out.connect(lambda: handle_logout(calib_window))

        calib_window.showMaximized()
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