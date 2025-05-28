import sys
from PyQt5.QtWidgets import QApplication
from solarjv_analyzer.utils.database import init_db
from solarjv_analyzer.gui.login_dialog import LoginDialog
from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow as MainWindow

def main():
    init_db()
    app = QApplication(sys.argv)
    login = LoginDialog()
    if login.exec_() == LoginDialog.Accepted:
        window = MainWindow(login.username_input.text())
        window.show()
        app.exec()

if __name__ == "__main__":
    main()
