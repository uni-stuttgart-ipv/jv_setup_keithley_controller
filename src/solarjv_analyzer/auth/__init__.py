"""
Authentication package for SolarJV Analyzer.

Provides:
- Database initialization (SQLite)
- User registration (argon2id)
- User authentication
- Session management with log file recording
- Logout handling
- Login dialog (PyQt5)
"""

from .database import init_db, register_user, authenticate_user
from .session import SessionManager, get_current_user, logout
from .login_dialog import show_login_dialog

__all__ = [
    "init_db",
    "register_user",
    "authenticate_user",
    "SessionManager",
    "get_current_user",
    "logout",
    "show_login_dialog",
]