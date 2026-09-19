"""render_auth_previews.py — offscreen renders of the auth flow screens.

Renders every screen in the login / registration / forgot-password / OTP
flow to PNGs under tools/previews/, for vision-based UI-consistency review
(the auth dialogs are deliberately excluded from the theme, so they are not
covered by tools/render_preview.py).

Usage:
    python3 tools/render_auth_previews.py

No hardware or display needed. Mirrors main.py's setup order (Fusion style,
design fonts, global plot config, DIALOG_STYLESHEET) so the renders match
production pixel-for-pixel.
"""
import logging
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

PREVIEW_DIR = os.path.join(REPO_ROOT, "tools", "previews")

from PyQt5 import QtWidgets  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402


def _snapshot_logging():
    root = logging.getLogger()
    return list(root.handlers), root.level


def _restore_logging(saved_handlers, saved_level):
    root = logging.getLogger()
    for h in list(root.handlers):
        if h not in saved_handlers:
            root.removeHandler(h)
    for h in saved_handlers:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(saved_level)


def _build_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setStyle("Fusion")

    from solarjv_analyzer.gui.theme import load_design_fonts, apply_global_plot_config
    from solarjv_analyzer.gui.style import DIALOG_STYLESHEET

    load_design_fonts()
    apply_global_plot_config()
    app.setStyleSheet(DIALOG_STYLESHEET)
    return app


def _grab(widget, path):
    widget.show()
    widget.raise_()
    QTest.qWait(160)
    QtWidgets.QApplication.processEvents()
    pix = widget.grab()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return pix.save(path, "PNG")


def main():
    saved = _snapshot_logging()
    app = _build_app()

    from solarjv_analyzer.auth.login_dialog import (
        LoginDialog,
        ForgotPasswordDialog,
    )
    from solarjv_analyzer.auth.otp_widget import OtpVerifierWidget, OtpVerifierPopup

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    shots = []

    # 1. Login dialog — login page (default state)
    dlg = LoginDialog()
    dlg._grab_path = os.path.join(PREVIEW_DIR, "auth_login.png")
    _grab(dlg, dlg._grab_path)
    shots.append("auth_login.png")

    # 2. Login dialog — login page with the footer links revealed (the
    #    Ctrl+Shift+A admin toggle makes "Create account" / "Forgot password?"
    #    visible; shown here so the flow entry points can be reviewed).
    dlg._toggle_admin_controls()
    _grab(dlg, os.path.join(PREVIEW_DIR, "auth_login_links.png"))
    shots.append("auth_login_links.png")
    dlg.close()

    # 3. Login dialog — registration page (stack index 1)
    dlg = LoginDialog()
    dlg.stacked.setCurrentIndex(1)
    _grab(dlg, os.path.join(PREVIEW_DIR, "auth_register.png"))
    shots.append("auth_register.png")
    dlg.close()

    # 4. Forgot-password dialog — page 0 (identity + send OTP)
    fp = ForgotPasswordDialog()
    _grab(fp, os.path.join(PREVIEW_DIR, "auth_forgot_0.png"))
    shots.append("auth_forgot_0.png")

    # 5. Forgot-password dialog — page 1 (inline OTP + new password).
    #    In production the OTP widget is injected only after a successful
    #    identity check; inject it directly to avoid SMTP.
    otp_inline = OtpVerifierWidget("test.student@example.com", fp, dark_theme=True)
    fp._otp_container.addWidget(otp_inline)
    fp._stack.setCurrentIndex(1)
    _grab(fp, os.path.join(PREVIEW_DIR, "auth_forgot_1.png"))
    shots.append("auth_forgot_1.png")
    fp.close()

    # 6. OTP popup (registration flow)
    pop = OtpVerifierPopup("test.student@example.com")
    _grab(pop, os.path.join(PREVIEW_DIR, "auth_otp_popup.png"))
    shots.append("auth_otp_popup.png")
    pop.close()

    _restore_logging(*saved)

    print("rendered auth screens:")
    for s in shots:
        print(f"  tools/previews/{s}")
    print("PASS" if shots else "FAIL")


if __name__ == "__main__":
    main()
