"""
Authentication package for SolarJV Analyzer.

Provides:
- Database initialization (SQLite) with schema migration
- User registration with email OTP verification (argon2id)
- User authentication with session logging
- Forgot-password flow with OTP-gated reset
- OTP email service (SMTP, rate limiting)
- OTP verifier widget and modal popup
- Login dialog (PyQt5, dark themed)
"""

from .database import (
    init_db, register_user, authenticate_user,
    username_exists, email_exists,
)
from .session import SessionManager, get_current_user, logout
from .login_dialog import show_login_dialog
from .email_service import (
    send_otp, verify_otp, clear_otp, mask_email,
    send_registration_confirmation, send_password_reset_confirmation,
    is_configured as is_email_configured,
)

__all__ = [
    "init_db",
    "register_user",
    "authenticate_user",
    "username_exists",
    "email_exists",
    "SessionManager",
    "get_current_user",
    "logout",
    "show_login_dialog",
    "send_otp",
    "verify_otp",
    "clear_otp",
    "mask_email",
    "send_registration_confirmation",
    "send_password_reset_confirmation",
    "is_email_configured",
]