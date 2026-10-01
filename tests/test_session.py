"""Tests for auth/session.py — session start/end and user tracking."""

import os

import pytest

from solarjv_analyzer.auth.session import (
    SessionManager, get_current_user, logout,
)


@pytest.fixture
def fresh_session(monkeypatch, tmp_path):
    """Redirect log directory and reset session state."""
    monkeypatch.setattr(
        "solarjv_analyzer.auth.session._get_app_data_dir",
        lambda: str(tmp_path),
    )
    SessionManager.current_user = None
    if SessionManager._file_handler:
        SessionManager._file_handler.close()
        SessionManager._file_handler = None
    yield
    if SessionManager._file_handler:
        SessionManager._file_handler.close()
        SessionManager._file_handler = None
    SessionManager.current_user = None


class TestSessionManager:
    def test_start_session_sets_user(self, fresh_session):
        SessionManager.start_session("test_user")
        assert SessionManager.current_user == "test_user"

    def test_start_session_creates_log_file(self, fresh_session, tmp_path):
        SessionManager.start_session("logger")
        log_dir = str(tmp_path / "logs")
        assert os.path.isdir(log_dir)
        files = os.listdir(log_dir)
        assert len(files) == 1
        assert files[0].startswith("session_")
        assert files[0].endswith(".log")

    def test_end_session_clears_user(self, fresh_session):
        SessionManager.start_session("temp")
        SessionManager.end_session()
        assert SessionManager.current_user is None

    def test_get_current_user_returns_none_initially(self, fresh_session):
        assert get_current_user() is None

    def test_get_current_user_after_login(self, fresh_session):
        SessionManager.start_session("active")
        assert get_current_user() == "active"

    def test_logout_clears_session(self, fresh_session):
        SessionManager.start_session("to_logout")
        logout(instrument_manager=None)
        assert get_current_user() is None
