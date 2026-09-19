"""Tests for auth/database.py — user registration, auth, password reset."""

import os
import sqlite3

import pytest
from solarjv_analyzer.auth.database import (
    init_db, register_user, authenticate_user, reset_password,
    _get_db_path, _get_app_data_dir, MIN_PASSWORD_LENGTH,
)


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Redirect the auth database to a temp file for each test."""
    db_path = str(tmp_path / "auth_test.db")

    def fake_get_db_path():
        return db_path

    def fake_get_app_data_dir():
        return str(tmp_path)

    monkeypatch.setattr(
        "solarjv_analyzer.auth.database._get_db_path", fake_get_db_path
    )
    monkeypatch.setattr(
        "solarjv_analyzer.auth.database._get_app_data_dir", fake_get_app_data_dir
    )
    # Re-init with the redirected path
    init_db()
    yield db_path
    # Cleanup
    try:
        os.remove(db_path)
    except OSError:
        pass


class TestRegistration:
    def test_successful_registration(self):
        register_user("a@b.com", "First", "Last", "st123456", "pass123", "pass123")
        assert authenticate_user("st123456", "pass123")

    def test_duplicate_username_raises(self):
        register_user("a@b.com", "F", "L", "dup", "pass123", "pass123")
        with pytest.raises(ValueError, match="already exists"):
            register_user("b@c.com", "X", "Y", "dup", "pass456", "pass456")

    def test_password_mismatch_raises(self):
        with pytest.raises(ValueError, match="do not match"):
            register_user("a@b.com", "F", "L", "st", "abc123", "xyz789")

    def test_short_password_raises(self):
        with pytest.raises(ValueError, match=f"at least {MIN_PASSWORD_LENGTH}"):
            register_user("a@b.com", "F", "L", "st", "12345", "12345")

    def test_empty_email_raises(self):
        with pytest.raises(ValueError, match="Email"):
            register_user("", "F", "L", "st", "pass123", "pass123")

    def test_empty_username_raises(self):
        with pytest.raises(ValueError, match="Username"):
            register_user("a@b.com", "F", "L", "", "pass123", "pass123")


class TestAuthentication:
    def test_valid_credentials(self):
        register_user("x@y.com", "A", "B", "valid_user", "secret12", "secret12")
        assert authenticate_user("valid_user", "secret12")

    def test_invalid_password(self):
        register_user("x@y.com", "A", "B", "u1", "secret12", "secret12")
        assert not authenticate_user("u1", "wrong")

    def test_nonexistent_user(self):
        assert not authenticate_user("ghost", "anything")

    def test_empty_credentials(self):
        assert not authenticate_user("", "")
        assert not authenticate_user("user", "")
        assert not authenticate_user("", "pass")


class TestPasswordReset:
    def test_successful_reset(self):
        register_user("r@x.com", "R", "S", "reset_me", "oldpass1", "oldpass1")
        reset_password("reset_me", "r@x.com", "newpass1")
        assert authenticate_user("reset_me", "newpass1")
        assert not authenticate_user("reset_me", "oldpass1")

    def test_wrong_email_fails(self):
        register_user("e@x.com", "E", "F", "user_x", "pass123", "pass123")
        with pytest.raises(ValueError, match="No account found"):
            reset_password("user_x", "wrong@x.com", "newpass1")

    def test_wrong_username_fails(self):
        register_user("e@x.com", "E", "F", "user_y", "pass123", "pass123")
        with pytest.raises(ValueError, match="No account found"):
            reset_password("ghost", "e@x.com", "newpass1")

    def test_short_new_password_fails(self):
        register_user("e@x.com", "E", "F", "user_z", "pass123", "pass123")
        with pytest.raises(ValueError, match=f"at least {MIN_PASSWORD_LENGTH}"):
            reset_password("user_z", "e@x.com", "12345")


class TestEmailVerifiedMigration:
    """Schema migration V1→V2 adds email_verified column."""

    def test_email_verified_column_exists(self, isolated_db):
        import sqlite3
        conn = sqlite3.connect(isolated_db)
        cur = conn.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cur.fetchall()}
        assert "email_verified" in columns
        conn.close()

    def test_existing_users_grandfathered(self, isolated_db):
        """Existing users default to email_verified = 1 after migration."""
        register_user("g@x.com", "G", "H", "grandfather", "pass123", "pass123")
        import sqlite3
        conn = sqlite3.connect(isolated_db)
        cur = conn.execute(
            "SELECT email_verified FROM users WHERE username = ?", ("grandfather",)
        )
        row = cur.fetchone()
        assert row is not None and row[0] == 1
        conn.close()


class TestLookupHelpers:
    """username_exists and email_exists."""

    def test_username_exists_true(self, isolated_db):
        from solarjv_analyzer.auth.database import username_exists
        register_user("x@y.com", "X", "Y", "existing_user", "pass123", "pass123")
        assert username_exists("existing_user")

    def test_username_exists_false(self, isolated_db):
        from solarjv_analyzer.auth.database import username_exists
        assert not username_exists("nonexistent_user")

    def test_username_exists_empty(self, isolated_db):
        from solarjv_analyzer.auth.database import username_exists
        assert not username_exists("")

    def test_email_exists_true(self, isolated_db):
        from solarjv_analyzer.auth.database import email_exists
        register_user("unique@test.com", "U", "T", "u1", "pass123", "pass123")
        assert email_exists("unique@test.com")

    def test_email_exists_false(self, isolated_db):
        from solarjv_analyzer.auth.database import email_exists
        assert not email_exists("no@test.com")
