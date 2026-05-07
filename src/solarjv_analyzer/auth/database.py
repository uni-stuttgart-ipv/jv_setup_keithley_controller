r"""
SQLite database for user accounts and schema versioning.

Location: %LOCALAPPDATA%\SolarJV\auth.db (Windows)
          ~/.local/share/SolarJV/auth.db (Linux)

WAL mode enabled.
"""

import os
import sys
import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# ---------------------------------------------------------------------------
# Password policy constants
MIN_PASSWORD_LENGTH = 6

# ---------------------------------------------------------------------------
# Argon2id hasher (defaults: time_cost=3, memory_cost=64 MiB, parallelism=1)
_hasher = PasswordHasher()

# ---------------------------------------------------------------------------
def _get_app_data_dir() -> str:
    """Return the platform‑appropriate SolarJV data directory."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "SolarJV")


def _get_db_path() -> str:
    """Return the full path to the SQLite database file."""
    data_dir = _get_app_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "auth.db")


def _connect() -> sqlite3.Connection:
    """Open (or create) the database with WAL mode enabled."""
    db_path = _get_db_path()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """
    Create or upgrade the database schema.
    Must be called once before any other database operations.
    """
    conn = _connect()
    try:
        # Ensure schema_version table exists
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            "    version INTEGER PRIMARY KEY"
            ")"
        )

        cur = conn.execute("SELECT MAX(version) FROM schema_version")
        current_version = cur.fetchone()[0] or 0

        # Version 0 → 1: initial schema
        if current_version < 1:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "    username TEXT UNIQUE NOT NULL,"
                "    password_hash TEXT NOT NULL,"
                "    email TEXT NOT NULL,"
                "    first_name TEXT NOT NULL,"
                "    last_name TEXT NOT NULL,"
                "    created_at TEXT NOT NULL DEFAULT (datetime('now')),"
                "    last_login TEXT"
                ")"
            )
            conn.execute("INSERT OR REPLACE INTO schema_version (version) VALUES (1)")

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Password validation
def _validate_password(password: str) -> None:
    """
    Raise ValueError if password does not meet the policy:
    - at least MIN_PASSWORD_LENGTH characters
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters long.")


# ---------------------------------------------------------------------------
# Public API
def register_user(
    email: str,
    first_name: str,
    last_name: str,
    username: str,
    password: str,
    confirm_password: str,
) -> None:
    """
    Register a new user.

    Args:
        email: University email address.
        first_name: First name.
        last_name: Last name.
        username: Desired username (must be unique).
        password: Password (will be validated and hashed).
        confirm_password: Must match `password`.

    Raises:
        ValueError: If validation fails (email empty, username taken,
                     password too short, passwords mismatch).
    """
    if not email.strip():
        raise ValueError("Email address is required.")
    if not username.strip():
        raise ValueError("Username is required.")

    if password != confirm_password:
        raise ValueError("Passwords do not match.")

    _validate_password(password)

    conn = _connect()
    try:
        # Check username uniqueness
        cur = conn.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone() is not None:
            raise ValueError("Username already exists. Please choose a different one.")

        password_hash = _hasher.hash(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, email, first_name, last_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, email.strip(), first_name.strip(), last_name.strip()),
        )
        conn.commit()
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> bool:
    """
    Verify username and password.

    Returns:
        True if authentication successful, False otherwise.

    Side effects:
        Updates last_login timestamp on success.
    """
    if not username or not password:
        return False

    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE username = ?", (username,)
        )
        row = cur.fetchone()
        if row is None:
            return False

        stored_hash = row[0]
        try:
            _hasher.verify(stored_hash, password)
        except VerifyMismatchError:
            return False

        # Update last_login
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE username = ?",
            (username,),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def reset_password(username: str, email: str, new_password: str) -> None:
    """
    Reset a user's password after verifying their username and email.

    Args:
        username: The account username.
        email: The email address associated with the account (must match exactly).
        new_password: The new password (will be validated and hashed).

    Raises:
        ValueError: If the username/email combination is not found or
                     the new password fails validation.
    """
    if not username or not email:
        raise ValueError("Username and email are required.")

    _validate_password(new_password)

    conn = _connect()
    try:
        cur = conn.execute(
            "SELECT id FROM users WHERE username = ? AND email = ?",
            (username.strip(), email.strip())
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError("No account found with that username and email combination.")

        password_hash = _hasher.hash(new_password)
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, row[0])
        )
        conn.commit()
    finally:
        conn.close()