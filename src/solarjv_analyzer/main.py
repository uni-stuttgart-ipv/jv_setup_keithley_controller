import sys
import os
from PyQt5.QtWidgets import QApplication

# --- PATH FIX ---
# Ensures Python can find 'solarjv_analyzer' even if running from inside the folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# ----------------

from solarjv_analyzer.utils.database import init_db
# Correct imports based on your project structure
from solarjv_analyzer.gui.login_dialog import LoginDialog
from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
# The new window we just created
from solarjv_analyzer.windows.calibration_window import CalibrationWindow

def main():
    init_db()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 1. Show Login Dialog (Modal)
    login = LoginDialog()
    if login.exec_() != LoginDialog.Accepted:
        sys.exit(0) # User cancelled login
        
    username = login.username_input.text()
    
    # 2. Show Calibration Window
    # We keep references to prevent garbage collection
    calib_window = CalibrationWindow(username)
    main_window = None

    def launch_main_app(instr_manager):
        """Callback: Runs when calibration passes OR skip is clicked."""
        nonlocal main_window
        
        # Open Main Analyzer Window
        main_window = JVAnalyzerWindow(username)  # Use captured username
        
        # Pass connected instruments to avoid reconnecting
        if hasattr(main_window, 'instrument_manager'):
             if instr_manager and instr_manager.keithley:
                 main_window.instrument_manager.keithley = instr_manager.keithley
             if instr_manager and instr_manager.mux:
                 main_window.instrument_manager.mux = instr_manager.mux
             
             # Update UI lights if the method exists
             if hasattr(main_window, 'update_instrument_lights'):
                 main_window.update_instrument_lights()
        
        main_window.show()
        calib_window.close()  # Close calibration window
    
    # Connect the signal from Calibration to the Main App launcher
    calib_window.calibration_passed.connect(launch_main_app)
    
    # Show the calibration window and start the event loop
    calib_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()