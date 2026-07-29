"""
Login and registration dialog for SolarJV Analyzer.

Modern UI with clean, flat design.
"""

from PyQt5 import QtWidgets, QtCore, QtGui

from .database import register_user, authenticate_user, reset_password
from .session import SessionManager


# ---------------------------------------------------------------------------
# Modern stylesheet
MODERN_STYLE = """
QDialog {
    background-color: #f5f7fa;
    border-radius: 12px;
}

QLabel {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #2c3e50;
    background: transparent;
}

QLabel#Title {
    font-size: 22px;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 8px;
}

QLineEdit {
    background-color: #ffffff;
    border: 1px solid #dcdde1;
    border-radius: 8px;
    padding: 6px 10px;               /* increased padding for better visibility */
    font-size: 13px;
    color: #2c3e50;
    selection-background-color: #a8d8ea;
    min-height: 20px;
}

QLineEdit:focus {
    border: 2px solid #0984e3;
    padding: 5px 9px;                /* compensate for thicker border */
}

QLineEdit::placeholder {
    color: #a0a4a8;                  /* visible grey for placeholder text */
    font-style: italic;
}

QPushButton {
    background-color: #0984e3;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #0873c7;
}

QPushButton:pressed {
    background-color: #065a9c;
}

QPushButton#SecondaryButton {
    background-color: #ecf0f1;
    color: #2c3e50;
    border: 1px solid #dcdde1;
}

QPushButton#SecondaryButton:hover {
    background-color: #dfe6e9;
}

QPushButton#DangerButton {
    background-color: #e74c3c;
}

QPushButton#DangerButton:hover {
    background-color: #c0392b;
}

QLabel#Feedback {
    font-size: 12px;
    font-weight: 500;
    min-height: 18px;
}

QFormLayout {
    spacing: 8px;
}
"""


# ---------------------------------------------------------------------------
class RegistrationForm(QtWidgets.QWidget):
    """Embedded form for new user registration."""

    registered = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(MODERN_STYLE)
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Create Account")
        title.setObjectName("Title")
        layout.addRow(title)

        self.email = QtWidgets.QLineEdit()
        self.email.setPlaceholderText("university@example.com")
        layout.addRow("University Email:", self.email)

        self.first_name = QtWidgets.QLineEdit()
        layout.addRow("First Name:", self.first_name)

        self.last_name = QtWidgets.QLineEdit()
        layout.addRow("Last Name:", self.last_name)

        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText("University ID (e.g., st123456, ac789012)")
        layout.addRow("Username:", self.username)

        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password.setPlaceholderText("At least 6 characters")
        layout.addRow("Password:", self.password)

        self.confirm_password = QtWidgets.QLineEdit()
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password.setPlaceholderText("Re-enter password")
        layout.addRow("Re-enter Password:", self.confirm_password)

        self.feedback = QtWidgets.QLabel("")
        self.feedback.setObjectName("Feedback")
        layout.addRow(self.feedback)

        btn_layout = QtWidgets.QHBoxLayout()
        self.register_btn = QtWidgets.QPushButton("Register")
        self.register_btn.clicked.connect(self._on_register)
        self.back_btn = QtWidgets.QPushButton("Back to Login")
        self.back_btn.setObjectName("SecondaryButton")
        self.back_btn.clicked.connect(self._on_back)
        btn_layout.addWidget(self.register_btn)
        btn_layout.addWidget(self.back_btn)
        layout.addRow(btn_layout)

    def _on_register(self):
        """Validate input and attempt registration."""
        self.feedback.setText("")
        try:
            register_user(
                email=self.email.text().strip(),
                first_name=self.first_name.text().strip(),
                last_name=self.last_name.text().strip(),
                username=self.username.text().strip(),
                password=self.password.text(),
                confirm_password=self.confirm_password.text(),
            )
            self.feedback.setStyleSheet("color: #27ae60; font-weight: 500;")
            self.feedback.setText("Registration successful! You can now log in.")
            self.registered.emit()
        except ValueError as e:
            self.feedback.setStyleSheet("color: #e74c3c; font-weight: 500;")
            self.feedback.setText(str(e))

    def _on_back(self):
        """Return to login screen."""
        self.registered.emit()


class ForgotPasswordDialog(QtWidgets.QDialog):
    """Dialog for resetting a forgotten password."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reset Password")
        self.setFixedSize(380, 300)
        self.setStyleSheet(MODERN_STYLE)

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Reset Password")
        title.setObjectName("Title")
        layout.addRow(title)

        self.username = QtWidgets.QLineEdit()
        self.username.setPlaceholderText("University ID")
        layout.addRow("Username:", self.username)

        self.email = QtWidgets.QLineEdit()
        self.email.setPlaceholderText("University email")
        layout.addRow("Email:", self.email)

        self.new_password = QtWidgets.QLineEdit()
        self.new_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.new_password.setPlaceholderText("At least 6 characters")
        layout.addRow("New Password:", self.new_password)

        self.confirm_password = QtWidgets.QLineEdit()
        self.confirm_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.confirm_password.setPlaceholderText("Re-enter new password")
        layout.addRow("Confirm:", self.confirm_password)

        self.feedback = QtWidgets.QLabel("")
        self.feedback.setObjectName("Feedback")
        layout.addRow(self.feedback)

        btn_layout = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset Password")
        self.reset_btn.clicked.connect(self._on_reset)
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setObjectName("SecondaryButton")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addRow(btn_layout)

    def _on_reset(self):
        """Attempt to reset the password."""
        self.feedback.setText("")
        if self.new_password.text() != self.confirm_password.text():
            self.feedback.setStyleSheet("color: #e74c3c;")
            self.feedback.setText("Passwords do not match.")
            return
        try:
            reset_password(
                username=self.username.text().strip(),
                email=self.email.text().strip(),
                new_password=self.new_password.text(),
            )
            self.feedback.setStyleSheet("color: #27ae60;")
            self.feedback.setText("Password reset successful! You can now log in.")
            QtCore.QTimer.singleShot(1500, self.accept)
        except ValueError as e:
            self.feedback.setStyleSheet("color: #e74c3c;")
            self.feedback.setText(str(e))


class LoginDialog(QtWidgets.QDialog):
    """Main dialog: login form with options to register or reset password."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SolarJV Analyzer – Login")
        self.setFixedSize(420, 380)
        self.setStyleSheet(MODERN_STYLE)
        self._logged_in_user = None

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.stacked = QtWidgets.QStackedWidget()
        self.main_layout.addWidget(self.stacked)

        # --- Login page ---
        self.login_page = QtWidgets.QWidget()
        login_layout = QtWidgets.QFormLayout(self.login_page)
        login_layout.setContentsMargins(30, 30, 30, 30)
        login_layout.setSpacing(10)

        title = QtWidgets.QLabel("Welcome Back")
        title.setObjectName("Title")
        login_layout.addRow(title)

        self.login_username = QtWidgets.QLineEdit()
        self.login_username.setPlaceholderText("University ID")
        login_layout.addRow("Username:", self.login_username)

        self.login_password = QtWidgets.QLineEdit()
        self.login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login_password.setPlaceholderText("Password")
        login_layout.addRow("Password:", self.login_password)

        self.login_feedback = QtWidgets.QLabel("")
        self.login_feedback.setObjectName("Feedback")
        login_layout.addRow(self.login_feedback)

        # Buttons row: Login, Register, Forgot Password
        btn_layout = QtWidgets.QHBoxLayout()
        self.login_btn = QtWidgets.QPushButton("Login")
        self.login_btn.clicked.connect(self._on_login)
        
        self.register_nav_btn = QtWidgets.QPushButton("Register")
        self.register_nav_btn.setObjectName("SecondaryButton")
        self.register_nav_btn.clicked.connect(self._show_registration)
        
        self.forgot_pw_btn = QtWidgets.QPushButton("Forgot Password?")
        self.forgot_pw_btn.setObjectName("SecondaryButton")
        self.forgot_pw_btn.clicked.connect(self._on_forgot_password)
        
        btn_layout.addWidget(self.login_btn)
        btn_layout.addWidget(self.register_nav_btn)
        btn_layout.addWidget(self.forgot_pw_btn)
        login_layout.addRow(btn_layout)

        # --- NEW: Hide admin buttons by default ---
        self.register_nav_btn.hide()
        self.forgot_pw_btn.hide()

        # --- NEW: Setup Admin Shortcut (Ctrl+Shift+A) ---
        self.admin_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Shift+A"), self)
        self.admin_shortcut.activated.connect(self._toggle_admin_controls)

        self.stacked.addWidget(self.login_page)

        # --- Registration page ---
        self.reg_form = RegistrationForm()
        self.reg_form.registered.connect(self._show_login)
        self.stacked.addWidget(self.reg_form)

        self.stacked.setCurrentIndex(0)

    # --- NEW: Method to toggle visibility ---
    def _toggle_admin_controls(self):
        """Toggle the visibility of the Register and Forgot Password buttons."""
        is_hidden = self.register_nav_btn.isHidden()
        self.register_nav_btn.setVisible(is_hidden)
        self.forgot_pw_btn.setVisible(is_hidden)

    def _show_registration(self):
        """Switch to the registration form."""
        self.stacked.setCurrentIndex(1)

    def _show_login(self):
        """Switch back to the login form."""
        self.stacked.setCurrentIndex(0)

    def _on_login(self):
        """Attempt authentication."""
        username = self.login_username.text().strip()
        password = self.login_password.text()

        if authenticate_user(username, password):
            SessionManager.start_session(username)
            self._logged_in_user = username
            self.accept()
        else:
            self.login_feedback.setStyleSheet("color: #e74c3c; font-weight: 500;")
            self.login_feedback.setText("Invalid username or password.")

    def _on_forgot_password(self):
        """Open the password reset dialog."""
        dlg = ForgotPasswordDialog(self)
        dlg.exec_()

    def get_logged_in_user(self) -> str:
        """Return the username of the successfully logged‑in user."""
        return self._logged_in_user


def show_login_dialog(parent=None) -> str:
    """
    Display the login dialog and return the username on success,
    or None if the user closed the dialog / wanted to exit.
    """
    dialog = LoginDialog(parent)
    result = dialog.exec_()
    if result == QtWidgets.QDialog.Accepted:
        return dialog.get_logged_in_user()
    return None