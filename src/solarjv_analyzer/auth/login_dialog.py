"""
Login, registration, and password-reset dialogs for SolarJV Analyzer.

Premium dark-themed split-screen UI.  Left: illustration panel with the ipv
lab visual and branding.  Right: frosted glass-surface form card with clean
typography and subtle depth.  All authentication logic is unchanged.
"""

import os

from PyQt5 import QtWidgets, QtCore, QtGui

from .database import register_user, authenticate_user, reset_password
from .session import SessionManager

# ---------------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------------
# Shipped inside the package under solarjv_analyzer/resources/ so it survives
# Briefcase packaging, which only bundles what lives under `sources`
# (src/solarjv_analyzer). A repo-root copy is kept as a development fallback.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.dirname(_THIS_DIR)
_IMAGE_PATH = os.path.join(_PACKAGE_ROOT, "resources", "Login.png")
if not os.path.exists(_IMAGE_PATH):
    # Development checkout: fall back to the repo-root copy.
    _IMAGE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(_PACKAGE_ROOT)), "Login.png"
    )

# ---------------------------------------------------------------------------
# Design tokens  —  dark professional palette
# ---------------------------------------------------------------------------
C_SURFACE    = "#0f172a"   # deep navy
C_SURFACE_2  = "#1e293b"   # card surface
C_PRIMARY    = "#10b981"   # mint accent
C_PRIMARY_H  = "#34d399"
C_TEXT       = "#f1f5f9"   # near-white
C_TEXT_MUTED = "#94a3b8"
C_BORDER     = "rgba(148,163,184,0.15)"
C_INPUT_BG   = "rgba(30,41,59,0.8)"
C_DANGER     = "#f87171"
C_WHITE      = "#ffffff"

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
RADIUS_CARD  = "20px"
RADIUS_INPUT = "10px"
RADIUS_BTN   = "10px"

# ---------------------------------------------------------------------------
# Shared stylesheets
# ---------------------------------------------------------------------------
_INPUT_STYLE = (
    f"QLineEdit {{"
    f"  font-family: {FONT}; font-size: 14px; color: {C_TEXT};"
    f"  background: {C_INPUT_BG};"
    f"  border: 1px solid {C_BORDER}; border-radius: {RADIUS_INPUT};"
    f"  padding: 13px 16px;"
    f"}}"
    f"QLineEdit:focus {{"
    f"  border: 1px solid {C_PRIMARY};"
    f"  background: rgba(30,41,59,0.95);"
    f"}}"
    f"QLineEdit::placeholder {{ color: #64748b; }}"
)

_LABEL_STYLE = (
    f"font-family: {FONT}; font-size: 12px; font-weight: 600;"
    f" color: {C_TEXT_MUTED}; background: transparent;"
    f" letter-spacing: 0.8px;"
)

_PRIMARY_BTN = (
    f"QPushButton {{"
    f"  background-color: {C_PRIMARY}; color: {C_WHITE}; border: none;"
    f"  border-radius: {RADIUS_BTN}; padding: 14px 0;"
    f"  font-family: {FONT}; font-size: 15px; font-weight: 600;"
    f"}}"
    f"QPushButton:hover {{ background-color: {C_PRIMARY_H}; }}"
    f"QPushButton:pressed {{ background-color: #059669; }}"
)

_SECONDARY_BTN = (
    f"QPushButton {{"
    f"  background: transparent; border: none;"
    f"  font-family: {FONT}; font-size: 13px; font-weight: 500;"
    f"  color: {C_TEXT_MUTED}; padding: 2px 0;"
    f"}}"
    f"QPushButton:hover {{ color: {C_WHITE}; }}"
)


# ======================================================================
# Left illustration panel
# ======================================================================

class _IllustrationPanel(QtWidgets.QWidget):
    """Full-bleed panel — the lab illustration fills the entire left side."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(400)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if os.path.exists(_IMAGE_PATH):
            pix = QtGui.QPixmap(_IMAGE_PATH)
            if not pix.isNull():
                scaled = pix.scaled(w, h,
                                    QtCore.Qt.KeepAspectRatioByExpanding,
                                    QtCore.Qt.SmoothTransformation)
                cx = (scaled.width() - w) // 2
                cy = (scaled.height() - h) // 2
                painter.drawPixmap(0, 0, w, h, scaled, cx, cy, w, h)

        painter.end()


# ======================================================================
# Registration form  (page 1 of right panel stack)
# ======================================================================

class RegistrationForm(QtWidgets.QWidget):
    """Embedded registration form with email OTP verification."""

    registered = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._username_check_timer = None
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 6px; }"
        )

        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title = QtWidgets.QLabel("Create Account")
        title.setStyleSheet(
            f"font-family: {FONT}; font-size: 28px; font-weight: 700;"
            f" color: {C_TEXT}; background: transparent;"
        )
        layout.addWidget(title)
        layout.addSpacing(6)

        sub = QtWidgets.QLabel("Enter your details to get started.")
        sub.setStyleSheet(
            f"font-family: {FONT}; font-size: 15px; color: {C_TEXT_MUTED};"
            f" background: transparent;"
        )
        layout.addWidget(sub)
        layout.addSpacing(28)

        fields = [
            ("EMAIL", "email", "university@example.com"),
            ("FIRST NAME", "first_name", ""),
            ("LAST NAME", "last_name", ""),
            ("USERNAME", "username", "e.g. st123456"),
            ("PASSWORD", "password", "Min 6 characters"),
            ("CONFIRM", "confirm_password", "Re-enter password"),
        ]
        self._fields = {}
        for label, attr, ph in fields:
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(_LABEL_STYLE)
            layout.addWidget(lbl)
            layout.addSpacing(6)
            le = QtWidgets.QLineEdit()
            le.setPlaceholderText(ph)
            le.setStyleSheet(_INPUT_STYLE)
            if "password" in attr.lower():
                le.setEchoMode(QtWidgets.QLineEdit.Password)
            layout.addWidget(le)
            self._fields[attr] = le
            layout.addSpacing(18)

        # Real-time username availability message (below username field)
        self._username_feedback = QtWidgets.QLabel("")
        self._username_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 11px; font-weight: 500;"
            f" background: transparent; min-height: 16px;"
        )
        layout.addWidget(self._username_feedback)

        # Connect username field to debounced availability check
        self.username.textChanged.connect(self._on_username_text_changed)

        self.feedback = QtWidgets.QLabel("")
        self.feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" background: transparent; min-height: 20px;"
            f" padding-top: 2px;"
        )
        layout.addWidget(self.feedback)

        layout.addSpacing(4)
        self.register_btn = QtWidgets.QPushButton("Create Account")
        self.register_btn.setStyleSheet(_PRIMARY_BTN)
        self.register_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.register_btn.clicked.connect(self._on_register)
        layout.addWidget(self.register_btn)

        layout.addSpacing(20)
        back = QtWidgets.QPushButton("← Back to Sign In")
        back.setStyleSheet(_SECONDARY_BTN)
        back.setCursor(QtCore.Qt.PointingHandCursor)
        back.clicked.connect(self.registered.emit)
        layout.addWidget(back, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

    @property
    def email(self):      return self._fields["email"]
    @property
    def first_name(self): return self._fields["first_name"]
    @property
    def last_name(self):  return self._fields["last_name"]
    @property
    def username(self):   return self._fields["username"]
    @property
    def password(self):   return self._fields["password"]
    @property
    def confirm_password(self): return self._fields["confirm_password"]

    # -------------------------------------------------------------------
    # Real-time username availability check (debounced)
    # -------------------------------------------------------------------

    def _on_username_text_changed(self, _text):
        if self._username_check_timer is not None:
            self._username_check_timer.stop()
        uname = self.username.text().strip()
        if not uname:
            self._username_feedback.setText("")
            return
        self._username_check_timer = QtCore.QTimer(self)
        self._username_check_timer.setSingleShot(True)
        self._username_check_timer.timeout.connect(self._check_username)
        self._username_check_timer.start(500)

    def _check_username(self):
        from .database import username_exists
        uname = self.username.text().strip()
        if not uname:
            self._username_feedback.setText("")
        elif username_exists(uname):
            self._username_feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 11px; font-weight: 500;"
                f" color: {C_DANGER}; background: transparent;"
            )
            self._username_feedback.setText("✗ This University ID is already registered.")
        else:
            self._username_feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 11px; font-weight: 500;"
                f" color: {C_PRIMARY}; background: transparent;"
            )
            self._username_feedback.setText("✓ Username is available.")

    # -------------------------------------------------------------------
    # Registration with OTP
    # -------------------------------------------------------------------

    def _on_register(self):
        self.feedback.setText("")

        # ---- local validation ------------------------------------------------
        email = self.email.text().strip()
        first = self.first_name.text().strip()
        last = self.last_name.text().strip()
        uname = self.username.text().strip()
        pw = self.password.text()
        cpw = self.confirm_password.text()

        if not email:
            self._show_error("Email address is required."); return
        if not first:
            self._show_error("First name is required."); return
        if not last:
            self._show_error("Last name is required."); return
        if not uname:
            self._show_error("University ID is required."); return
        if len(pw) < 6:
            self._show_error("Password must be at least 6 characters."); return
        if pw != cpw:
            self._show_error("Passwords do not match."); return

        # ---- server-side uniqueness checks -----------------------------------
        from .database import username_exists, email_exists
        if username_exists(uname):
            self._show_error("This University ID is already registered."); return
        if email_exists(email):
            self._show_error("An account with this email already exists."); return

        # ---- send OTP --------------------------------------------------------
        from .email_service import send_otp
        ok, msg = send_otp(email)
        if not ok:
            self._show_error(msg); return

        # ---- show OTP popup --------------------------------------------------
        from .otp_widget import OtpVerifierPopup
        popup = OtpVerifierPopup(email, self)
        if popup.exec_() == QtWidgets.QDialog.Accepted:
            # OTP verified — create account
            try:
                register_user(email, first, last, uname, pw, cpw)
                self._show_success("✓ Account created! You can now sign in.")
                # Send welcome email (fire-and-forget)
                from .email_service import send_registration_confirmation
                send_registration_confirmation(email, uname)
                QtCore.QTimer.singleShot(2000, self.registered.emit)
            except ValueError as e:
                self._show_error(str(e))
        else:
            # User cancelled OTP
            from .email_service import clear_otp
            clear_otp(email)

    def _show_error(self, msg):
        self.feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" color: {C_DANGER}; background: transparent; padding-top: 2px;"
        )
        self.feedback.setText(msg)

    def _show_success(self, msg):
        self.feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" color: {C_PRIMARY}; background: transparent; padding-top: 2px;"
        )
        self.feedback.setText(msg)


# ======================================================================
# Forgot-password dialog
# ======================================================================

class ForgotPasswordDialog(QtWidgets.QDialog):
    """2-page modal: Page 0 = identity verification + send OTP,
    Page 1 = OTP widget inline + set new password."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reset Password")
        self.setFixedSize(460, 520)
        self.setStyleSheet(
            f"QDialog {{"
            f"  background-color: {C_SURFACE_2};"
            f"  border-radius: {RADIUS_CARD};"
            f"}}"
        )
        self._verified_email = ""
        self._build_ui()

    # ---- Page 0: identity ---------------------------------------------------

    def _build_page_0(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(44, 40, 44, 36)
        layout.setSpacing(0)

        title = QtWidgets.QLabel("Reset Password")
        title.setStyleSheet(
            f"font-family: {FONT}; font-size: 28px; font-weight: 700;"
            f" color: {C_TEXT}; background: transparent;"
        )
        layout.addWidget(title)
        layout.addSpacing(6)

        sub = QtWidgets.QLabel("Enter your University ID and email to continue.")
        sub.setStyleSheet(
            f"font-family: {FONT}; font-size: 15px; color: {C_TEXT_MUTED};"
            f" background: transparent;"
        )
        layout.addWidget(sub)
        layout.addSpacing(28)

        # Username
        layout.addWidget(QtWidgets.QLabel("USERNAME"))
        layout.itemAt(layout.count() - 1).widget().setStyleSheet(_LABEL_STYLE)
        layout.addSpacing(6)
        self._fp_username = QtWidgets.QLineEdit()
        self._fp_username.setPlaceholderText("University ID")
        self._fp_username.setStyleSheet(_INPUT_STYLE)
        layout.addWidget(self._fp_username)
        layout.addSpacing(18)

        # Email
        layout.addWidget(QtWidgets.QLabel("EMAIL"))
        layout.itemAt(layout.count() - 1).widget().setStyleSheet(_LABEL_STYLE)
        layout.addSpacing(6)
        self._fp_email = QtWidgets.QLineEdit()
        self._fp_email.setPlaceholderText("university@example.com")
        self._fp_email.setStyleSheet(_INPUT_STYLE)
        layout.addWidget(self._fp_email)
        layout.addSpacing(18)

        self._fp_feedback = QtWidgets.QLabel("")
        self._fp_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" background: transparent; min-height: 20px; padding-top: 2px;"
        )
        layout.addWidget(self._fp_feedback)
        layout.addSpacing(4)

        send_btn = QtWidgets.QPushButton("Send Verification Code")
        send_btn.setStyleSheet(_PRIMARY_BTN)
        send_btn.setCursor(QtCore.Qt.PointingHandCursor)
        send_btn.clicked.connect(self._on_send_otp)
        layout.addWidget(send_btn)

        layout.addSpacing(16)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.setStyleSheet(_SECONDARY_BTN)
        cancel.setCursor(QtCore.Qt.PointingHandCursor)
        cancel.clicked.connect(self.reject)
        layout.addWidget(cancel, alignment=QtCore.Qt.AlignCenter)
        layout.addStretch()

        return page

    # ---- Page 1: OTP + new password ----------------------------------------

    def _build_page_1(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(36, 32, 36, 28)
        layout.setSpacing(0)

        # Inline OTP widget
        self._otp_widget = None  # set dynamically
        self._otp_container = QtWidgets.QVBoxLayout()
        layout.addLayout(self._otp_container)

        layout.addSpacing(16)

        # New password fields (initially disabled until OTP verified)
        for label, attr, ph in [
            ("NEW PASSWORD", "new_password", "Min 6 characters"),
            ("CONFIRM", "confirm_password", "Re-enter new password"),
        ]:
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(_LABEL_STYLE)
            layout.addWidget(lbl)
            layout.addSpacing(6)
            le = QtWidgets.QLineEdit()
            le.setPlaceholderText(ph)
            le.setStyleSheet(_INPUT_STYLE)
            le.setEchoMode(QtWidgets.QLineEdit.Password)
            le.setEnabled(False)
            layout.addWidget(le)
            setattr(self, f"_fp_{attr}", le)
            layout.addSpacing(14)

        self._fp_reset_feedback = QtWidgets.QLabel("")
        self._fp_reset_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" background: transparent; min-height: 20px; padding-top: 2px;"
        )
        layout.addWidget(self._fp_reset_feedback)
        layout.addSpacing(4)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        back_btn = QtWidgets.QPushButton("← Back")
        back_btn.setStyleSheet(_SECONDARY_BTN)
        back_btn.setCursor(QtCore.Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        btn_row.addWidget(back_btn)

        self._set_pw_btn = QtWidgets.QPushButton("Set New Password")
        self._set_pw_btn.setStyleSheet(_PRIMARY_BTN + "QPushButton { padding: 12px 24px; }")
        self._set_pw_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._set_pw_btn.clicked.connect(self._on_set_password)
        btn_row.addWidget(self._set_pw_btn, stretch=1)
        layout.addLayout(btn_row)
        layout.addStretch()

        return page

    # ---- build --------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QtWidgets.QStackedWidget()
        self._stack.setStyleSheet("background: transparent;")
        self._stack.addWidget(self._build_page_0())
        self._stack.addWidget(self._build_page_1())
        layout.addWidget(self._stack)

    # ---- actions ------------------------------------------------------------

    def _on_send_otp(self):
        uname = self._fp_username.text().strip()
        email = self._fp_email.text().strip()

        if not uname or not email:
            self._show_fp_error("Please fill in both fields."); return

        # Verify identity
        from .database import _connect
        conn = _connect()
        try:
            cur = conn.execute(
                "SELECT id FROM users WHERE username = ? AND email = ?",
                (uname, email.lower()))
            if cur.fetchone() is None:
                self._show_fp_error("No account found with that ID and email.")
                return
        finally:
            conn.close()

        # Send OTP
        from .email_service import send_otp
        ok, msg = send_otp(email)
        if not ok:
            self._show_fp_error(msg); return

        self._verified_email = email
        # Replace OTP container content
        from .otp_widget import OtpVerifierWidget
        # Clear old
        while self._otp_container.count():
            item = self._otp_container.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._otp_widget = OtpVerifierWidget(email, self, dark_theme=True)
        self._otp_widget.verified.connect(self._on_otp_verified)
        self._otp_container.addWidget(self._otp_widget)
        self._stack.setCurrentIndex(1)

    def _on_otp_verified(self):
        """Enable the password fields now that OTP is confirmed."""
        self._fp_new_password.setEnabled(True)
        self._fp_confirm_password.setEnabled(True)
        self._fp_reset_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" color: {C_PRIMARY}; background: transparent; padding-top: 2px;"
        )
        self._fp_reset_feedback.setText("✓ Email verified. Set your new password below.")

    def _on_set_password(self):
        pw = self._fp_new_password.text()
        cpw = self._fp_confirm_password.text()
        if len(pw) < 6:
            self._show_reset_error("Password must be at least 6 characters."); return
        if pw != cpw:
            self._show_reset_error("Passwords do not match."); return
        try:
            reset_password(self._fp_username.text().strip(),
                           self._verified_email, pw)
            self._fp_reset_feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
                f" color: {C_PRIMARY}; background: transparent; padding-top: 2px;"
            )
            self._fp_reset_feedback.setText("✓ Password reset successful!")
            # Send confirmation email (fire-and-forget)
            from .email_service import send_password_reset_confirmation
            send_password_reset_confirmation(self._verified_email)
            QtCore.QTimer.singleShot(1500, self.accept)
        except ValueError as e:
            self._show_reset_error(str(e))

    # ---- feedback helpers ---------------------------------------------------

    def _show_fp_error(self, msg):
        self._fp_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" color: {C_DANGER}; background: transparent; padding-top: 2px;"
        )
        self._fp_feedback.setText(msg)

    def _show_reset_error(self, msg):
        self._fp_reset_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" color: {C_DANGER}; background: transparent; padding-top: 2px;"
        )
        self._fp_reset_feedback.setText(msg)


# ======================================================================
# Login dialog
# ======================================================================

class LoginDialog(QtWidgets.QDialog):
    """Premium dark split-screen login: illustration left, glass form right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SolarJV Analyzer – Login")
        self.setFixedSize(960, 620)
        self._logged_in_user = None
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(
            f"QDialog {{"
            f"  background-color: {C_SURFACE};"
            f"  border-radius: 24px;"
            f"}}"
        )
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- left: full-bleed illustration ----------------------------------
        self._illustration = _IllustrationPanel(self)
        root.addWidget(self._illustration, stretch=1)

        # ---- right: form panel -----------------------------------------------
        right = QtWidgets.QWidget()
        right.setStyleSheet(
            f"background: qlineargradient(x1:0 y1:0, x2:0 y2:1,"
            f" stop:0 {C_SURFACE_2}, stop:1 #162032);"
            f" border-top-right-radius: {RADIUS_CARD};"
            f" border-bottom-right-radius: {RADIUS_CARD};"
        )
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(56, 48, 56, 40)
        right_layout.setSpacing(0)

        # ipv brand header
        right_layout.addWidget(self._build_right_header())
        right_layout.addSpacing(12)

        # Stack: login ↔ register
        self.stacked = QtWidgets.QStackedWidget()
        self.stacked.setStyleSheet("background: transparent;")
        self.stacked.addWidget(self._build_login_page())
        self.stacked.addWidget(RegistrationForm(self))
        self.stacked.widget(1).registered.connect(self._show_login)
        right_layout.addWidget(self.stacked, stretch=1)

        # Brand footer — centred at the bottom of the right panel
        right_layout.addSpacing(24)
        brand = QtWidgets.QLabel("SolarJV Analyzer")
        brand.setStyleSheet(
            f"font-family: {FONT}; font-size: 16px; font-weight: 700;"
            f" color: {C_TEXT}; background: transparent;"
        )
        brand.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(brand)

        tag = QtWidgets.QLabel(
            "Automated J‑V characterisation & SPO degradation testing"
        )
        tag.setStyleSheet(
            f"font-family: {FONT}; font-size: 11px; color: {C_TEXT_MUTED};"
            f" background: transparent;"
        )
        tag.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(tag)

        right_layout.addSpacing(4)
        uni = QtWidgets.QLabel("University of Stuttgart")
        uni.setStyleSheet(
            f"font-family: {FONT}; font-size: 11px; font-weight: 500;"
            f" color: #475569; background: transparent;"
        )
        uni.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(uni)

        root.addWidget(right, stretch=1)

        # Admin shortcut
        self.admin_shortcut = QtWidgets.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+A"), self
        )
        self.admin_shortcut.activated.connect(self._toggle_admin_controls)

    # -------------------------------------------------------------------
    # Right header
    # -------------------------------------------------------------------

    def _build_right_header(self):
        h = QtWidgets.QWidget()
        h.setStyleSheet("background: transparent;")
        hl = QtWidgets.QHBoxLayout(h)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)

        hl.addStretch()

        lang = QtWidgets.QComboBox()
        lang.addItems(["English (UK)", "Deutsch"])
        lang.setStyleSheet(
            f"QComboBox {{"
            f"  font-family: {FONT}; font-size: 12px; color: {C_TEXT_MUTED};"
            f"  background: transparent; border: none; padding: 2px 4px;"
            f"}}"
            f"QComboBox::drop-down {{ border: none; width: 14px; }}"
            f"QComboBox QAbstractItemView {{"
            f"  font-family: {FONT}; font-size: 12px;"
            f"  background: {C_SURFACE_2}; border: 1px solid {C_BORDER};"
            f"  selection-background-color: {C_PRIMARY}; color: {C_TEXT};"
            f"  border-radius: 8px; padding: 4px;"
            f"}}"
        )
        hl.addWidget(lang)
        return h

    # -------------------------------------------------------------------
    # Login page
    # -------------------------------------------------------------------

    def _build_login_page(self):
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Welcome
        title = QtWidgets.QLabel("Welcome back")
        title.setStyleSheet(
            f"font-family: {FONT}; font-size: 28px; font-weight: 700;"
            f" color: {C_TEXT}; background: transparent;"
        )
        layout.addWidget(title)
        layout.addSpacing(6)

        sub = QtWidgets.QLabel("Sign in to your account to continue.")
        sub.setStyleSheet(
            f"font-family: {FONT}; font-size: 15px; color: {C_TEXT_MUTED};"
            f" background: transparent;"
        )
        layout.addWidget(sub)
        layout.addSpacing(36)

        # University ID
        layout.addWidget(QtWidgets.QLabel("UNIVERSITY ID"))
        layout.itemAt(layout.count() - 1).widget().setStyleSheet(_LABEL_STYLE)
        layout.addSpacing(6)

        self.login_username = QtWidgets.QLineEdit()
        self.login_username.setPlaceholderText("e.g. st123456")
        self.login_username.setStyleSheet(_INPUT_STYLE)
        layout.addWidget(self.login_username)
        layout.addSpacing(24)

        # Password
        layout.addWidget(QtWidgets.QLabel("PASSWORD"))
        layout.itemAt(layout.count() - 1).widget().setStyleSheet(_LABEL_STYLE)
        layout.addSpacing(6)

        self.login_password = QtWidgets.QLineEdit()
        self.login_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.login_password.setPlaceholderText("Enter your password")
        self.login_password.setStyleSheet(_INPUT_STYLE)
        layout.addWidget(self.login_password)

        # Feedback
        self.login_feedback = QtWidgets.QLabel("")
        self.login_feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
            f" background: transparent; min-height: 26px;"
            f" padding-top: 6px;"
        )
        layout.addWidget(self.login_feedback)

        # Sign In button
        self.login_btn = QtWidgets.QPushButton("Sign In")
        self.login_btn.setStyleSheet(_PRIMARY_BTN)
        self.login_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.login_btn.clicked.connect(self._on_login)
        layout.addWidget(self.login_btn)

        layout.addSpacing(24)

        # Footer links
        link_row = QtWidgets.QHBoxLayout()
        link_row.setSpacing(0)

        self.register_nav_btn = QtWidgets.QPushButton("Create account")
        self.register_nav_btn.setStyleSheet(_SECONDARY_BTN)
        self.register_nav_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.register_nav_btn.clicked.connect(self._show_registration)
        self.register_nav_btn.hide()
        link_row.addWidget(self.register_nav_btn)

        link_row.addStretch()

        self.forgot_pw_btn = QtWidgets.QPushButton("Forgot password?")
        self.forgot_pw_btn.setStyleSheet(_SECONDARY_BTN)
        self.forgot_pw_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.forgot_pw_btn.clicked.connect(self._on_forgot_password)
        self.forgot_pw_btn.hide()
        link_row.addWidget(self.forgot_pw_btn)

        layout.addLayout(link_row)
        layout.addStretch()

        return page

    # -------------------------------------------------------------------
    # Navigation & Auth
    # -------------------------------------------------------------------

    def _show_registration(self):
        self.stacked.setCurrentIndex(1)

    def _show_login(self):
        self.stacked.setCurrentIndex(0)

    def _toggle_admin_controls(self):
        hidden = self.register_nav_btn.isHidden()
        self.register_nav_btn.setVisible(hidden)
        self.forgot_pw_btn.setVisible(hidden)

    def _on_forgot_password(self):
        ForgotPasswordDialog(self).exec_()

    def _on_login(self):
        username = self.login_username.text().strip()
        password = self.login_password.text()

        if authenticate_user(username, password):
            SessionManager.start_session(username)
            self._logged_in_user = username
            self.accept()
        else:
            self.login_feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 13px; font-weight: 500;"
                f" color: {C_DANGER}; background: transparent;"
                f" padding-top: 6px;"
            )
            self.login_feedback.setText("Invalid username or password.")

    def get_logged_in_user(self) -> str:
        return self._logged_in_user


# ======================================================================
# Public entry point
# ======================================================================

def show_login_dialog(parent=None) -> str:
    """Show the login dialog.  Returns the username on success, or None."""
    dialog = LoginDialog(parent)
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        return dialog.get_logged_in_user()
    return None
