"""
Reusable OTP verifier widget and modal popup for SolarJV Analyzer.

OtpVerifierWidget — 6-digit input, auto-advance, paste support, shake animation.
OtpVerifierPopup — QDialog wrapper for modal use (Registration flow).
The widget can also be embedded inline (Forgot Password flow).
"""

from PyQt5 import QtWidgets, QtCore, QtGui

from .email_service import verify_otp, send_otp, mask_email

FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Auth palette — kept in sync with login_dialog.py (the auth module owns its own
# navy/green styling and is deliberately excluded from the main app theme).
C_PRIMARY = "#10b981"
C_PRIMARY_H = "#34d399"
C_DANGER = "#f87171"
C_SURFACE = "#0f172a"
C_SURFACE_2 = "#1e293b"
C_BORDER = "rgba(148,163,184,0.15)"
C_INPUT_BG = "rgba(30,41,59,0.8)"

# Dark-theme text tokens (match login_dialog.py).
C_TEXT_DARK = "#f1f5f9"
C_TEXT_MUTED_DARK = "#94a3b8"

# Light-theme tokens (retained for the non-default light variant).
C_TEXT = "#1e293b"
C_TEXT_MUTED = "#64748b"
C_BG = "#ffffff"
C_INPUT_BG_LIGHT = "#f8fafc"


def _primary_btn_style() -> str:
    return (
        f"QPushButton {{"
        f"  background-color: {C_PRIMARY}; color: white; border: none;"
        f"  border-radius: 10px; padding: 14px 0; font-size: 15px;"
        f"  font-weight: 600; font-family: {FONT};"
        f"}}"
        f"QPushButton:hover {{ background-color: {C_PRIMARY_H}; }}"
        f"QPushButton:pressed {{ background-color: #059669; }}"
        f"QPushButton:disabled {{ background-color: #475569; color: #94a3b8; }}"
    )


def _link_btn_style(dark: bool) -> str:
    base = C_TEXT_MUTED_DARK if dark else C_TEXT_MUTED
    hover = C_TEXT_DARK if dark else C_TEXT
    return (
        f"QPushButton {{"
        f"  background: transparent; border: none; color: {base};"
        f"  font-size: 13px; font-weight: 500; font-family: {FONT};"
        f"}}"
        f"QPushButton:hover {{ color: {hover}; }}"
        f"QPushButton:disabled {{ color: #475569; }}"
    )


class OtpVerifierWidget(QtWidgets.QWidget):
    """Reusable 6-digit OTP input with verification logic.
    Emits `verified` on success, `cancelled` on user cancel."""

    verified = QtCore.pyqtSignal()
    cancelled = QtCore.pyqtSignal()

    def __init__(self, email: str, parent=None, dark_theme: bool = False):
        super().__init__(parent)
        self._email = email.strip().lower()
        self._dark = dark_theme
        self._boxes: list[QtWidgets.QLineEdit] = []
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        bg = C_SURFACE_2 if self._dark else C_BG
        txt = C_TEXT_DARK if self._dark else C_TEXT
        mut = C_TEXT_MUTED_DARK if self._dark else C_TEXT_MUTED
        input_bg = C_INPUT_BG if self._dark else C_INPUT_BG_LIGHT

        self.setStyleSheet(f"background: {bg};")

        # Title
        title = QtWidgets.QLabel("Verify Your Email")
        title.setStyleSheet(
            f"font-family: {FONT}; font-size: 20px; font-weight: 700;"
            f" color: {txt}; background: transparent;"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # Masked email
        masked = mask_email(self._email)
        email_lbl = QtWidgets.QLabel(f"Code sent to {masked}")
        email_lbl.setStyleSheet(
            f"font-family: {FONT}; font-size: 13px; color: {mut};"
            f" background: transparent;"
        )
        email_lbl.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(email_lbl)

        layout.addSpacing(4)

        # 6 digit boxes
        box_row = QtWidgets.QHBoxLayout()
        box_row.setSpacing(6)
        box_row.addStretch()
        for i in range(6):
            box = QtWidgets.QLineEdit()
            box.setFixedSize(44, 52)
            box.setMaxLength(1)
            box.setAlignment(QtCore.Qt.AlignCenter)
            box.setStyleSheet(
                f"QLineEdit {{"
                f"  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;"
                f"  font-size: 22px; font-weight: 700;"
                f"  color: {txt}; background: {input_bg};"
                f"  border: 1px solid {C_BORDER}; border-radius: 10px;"
                f"}}"
                f"QLineEdit:focus {{ border: 1px solid {C_PRIMARY}; }}"
            )
            box.textChanged.connect(lambda text, idx=i: self._on_digit_changed(text, idx))
            box.keyPressEvent = self._make_key_handler(i, box.keyPressEvent)
            box_row.addWidget(box)
            self._boxes.append(box)
        box_row.addStretch()
        layout.addLayout(box_row)

        # Feedback
        self._feedback = QtWidgets.QLabel("")
        self._feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 12px; font-weight: 500;"
            f" background: transparent; min-height: 18px;"
        )
        self._feedback.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self._feedback)

        # Verify button
        self._verify_btn = QtWidgets.QPushButton("Verify")
        self._verify_btn.setStyleSheet(_primary_btn_style())
        self._verify_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._verify_btn.clicked.connect(self._on_verify)
        layout.addWidget(self._verify_btn)

        # Resend + Cancel row
        link_row = QtWidgets.QHBoxLayout()
        link_row.setSpacing(16)
        link_row.addStretch()

        self._resend_btn = QtWidgets.QPushButton("Resend Code")
        self._resend_btn.setStyleSheet(_link_btn_style(self._dark))
        self._resend_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._resend_btn.clicked.connect(self._on_resend)
        self._resend_btn.setEnabled(False)
        link_row.addWidget(self._resend_btn)

        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(_link_btn_style(self._dark))
        cancel_btn.setCursor(QtCore.Qt.PointingHandCursor)
        cancel_btn.clicked.connect(self.cancelled.emit)
        link_row.addWidget(cancel_btn)
        link_row.addStretch()
        layout.addLayout(link_row)

        # Start resend cooldown
        self._cooldown = 60
        self._start_cooldown()
        # Auto-focus first box
        QtCore.QTimer.singleShot(100, self._safe(lambda: self._boxes[0].setFocus()))

    # -------------------------------------------------------------------
    # Digit handling
    # -------------------------------------------------------------------

    def _on_digit_changed(self, text: str, idx: int):
        if text and idx < 5:
            self._boxes[idx + 1].setFocus()
        # If all 6 filled, auto-verify after debounce
        if self._all_filled():
            QtCore.QTimer.singleShot(300, self._safe(self._on_verify))

    def _make_key_handler(self, idx: int, original_handler):
        def handler(event):
            if event.key() == QtCore.Qt.Key_Backspace:
                if not self._boxes[idx].text() and idx > 0:
                    self._boxes[idx - 1].setFocus()
                    self._boxes[idx - 1].clear()
                    return
            if event.key() == QtCore.Qt.Key_V and event.modifiers() == QtCore.Qt.ControlModifier:
                clipboard = QtWidgets.QApplication.clipboard().text().strip()
                if len(clipboard) == 6 and clipboard.isdigit():
                    for i, ch in enumerate(clipboard):
                        self._boxes[i].setText(ch)
                    return
            original_handler(event)
        return handler

    def _all_filled(self) -> bool:
        return all(b.text().strip() for b in self._boxes)

    def _get_code(self) -> str:
        return "".join(b.text().strip() for b in self._boxes)

    # -------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------

    def _on_verify(self):
        code = self._get_code()
        if len(code) != 6:
            self._show_error("Please enter the full 6-digit code.")
            return

        self._verify_btn.setEnabled(False)
        self._verify_btn.setText("Verifying…")
        QtCore.QTimer.singleShot(200, self._safe(lambda: self._do_verify(code)))

    def _do_verify(self, code: str):
        ok, msg = verify_otp(self._email, code)
        if ok:
            self._feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 12px; font-weight: 500;"
                f" color: {C_PRIMARY}; background: transparent;"
            )
            self._feedback.setText("✓ Verified!")
            self.verified.emit()
        else:
            self._shake_boxes()
            self._show_error(msg)
        self._verify_btn.setEnabled(True)
        self._verify_btn.setText("Verify")

    def _on_resend(self):
        ok, msg = send_otp(self._email)
        if ok:
            self._feedback.setStyleSheet(
                f"font-family: {FONT}; font-size: 12px; font-weight: 500;"
                f" color: {C_PRIMARY}; background: transparent;"
            )
            self._feedback.setText("✓ New code sent!")
            self._cooldown = 60
            self._start_cooldown()
            for b in self._boxes:
                b.clear()
            self._boxes[0].setFocus()
        else:
            self._show_error(msg)

    @staticmethod
    def _safe(fn):
        """Wrap a callback so it silently no-ops if the widget was destroyed."""
        def wrapper():
            try:
                fn()
            except RuntimeError:
                pass
        return wrapper

    def _start_cooldown(self):
        def tick():
            self._cooldown -= 1
            if self._cooldown <= 0:
                self._resend_btn.setEnabled(True)
                self._resend_btn.setText("Resend Code")
                return
            self._resend_btn.setText(f"Resend in 0:{self._cooldown:02d}")
            QtCore.QTimer.singleShot(1000, self._safe(tick))

        self._resend_btn.setEnabled(False)
        self._resend_btn.setText(f"Resend in 0:{self._cooldown:02d}")
        QtCore.QTimer.singleShot(1000, self._safe(tick))

    # -------------------------------------------------------------------
    # UI effects
    # -------------------------------------------------------------------

    def _show_error(self, msg: str):
        self._feedback.setStyleSheet(
            f"font-family: {FONT}; font-size: 12px; font-weight: 500;"
            f" color: {C_DANGER}; background: transparent;"
        )
        self._feedback.setText(msg)

    def _shake_boxes(self):
        """Horizontal shake animation on all digit boxes."""
        original_positions = [b.pos() for b in self._boxes]
        shakes = [(3, 0), (-3, 0), (3, 0), (-3, 0), (0, 0)]

        def step(i=0):
            if i >= len(shakes):
                return
            dx, dy = shakes[i]
            for b, orig in zip(self._boxes, original_positions):
                b.move(orig.x() + dx, orig.y() + dy)
            QtCore.QTimer.singleShot(60, self._safe(lambda: step(i + 1)))

        step()

        # Flash red borders briefly
        for b in self._boxes:
            b.setStyleSheet(b.styleSheet() +
                            "QLineEdit { border: 2px solid " + C_DANGER + "; }")
        QtCore.QTimer.singleShot(600, self._safe(self._reset_box_borders))

    def _reset_box_borders(self):
        txt = C_TEXT_DARK if self._dark else C_TEXT
        bg = C_INPUT_BG if self._dark else C_INPUT_BG_LIGHT
        for b in self._boxes:
            b.setStyleSheet(
                f"QLineEdit {{"
                f"  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;"
                f"  font-size: 22px; font-weight: 700;"
                f"  color: {txt}; background: {bg};"
                f"  border: 1px solid {C_BORDER}; border-radius: 10px;"
                f"}}"
                f"QLineEdit:focus {{ border: 1px solid {C_PRIMARY}; }}"
            )


# ======================================================================
# Modal popup wrapper — used in Registration flow
# ======================================================================

class OtpVerifierPopup(QtWidgets.QDialog):
    """Modal dialog wrapping OtpVerifierWidget for standalone use."""

    def __init__(self, email: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Verify Email")
        self.setFixedSize(400, 320)
        self.setStyleSheet(
            f"QDialog {{"
            f"  background-color: {C_SURFACE_2};"
            f"  border-radius: 16px;"
            f"}}"
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)

        self._widget = OtpVerifierWidget(email, parent=self, dark_theme=True)
        self._widget.verified.connect(self.accept)
        self._widget.cancelled.connect(self.reject)
        layout.addWidget(self._widget)
