"""Tests for auth/email_service.py — OTP generation, verification, rate limiting."""

import time
import pytest
from solarjv_analyzer.auth.email_service import (
    send_otp, verify_otp, clear_otp, mask_email,
    _otp_store, _request_tracker, _send_email, _get_smtp_config,
    OTP_LENGTH, MAX_FAILED_ATTEMPTS, is_configured,
)


@pytest.fixture(autouse=True)
def clean_store(monkeypatch):
    """Wipe in-memory stores and mock SMTP for each test."""
    _otp_store.clear()
    _request_tracker.clear()
    # Mock _send_email to always succeed without real SMTP
    monkeypatch.setattr(
        "solarjv_analyzer.auth.email_service._send_email",
        lambda to, subj, html: (True, ""),
    )
    yield


class TestSendOtp:
    def test_send_otp_stores_and_returns(self):
        ok, msg = send_otp("test@uni-stuttgart.de")
        assert ok
        assert msg == ""
        assert "test@uni-stuttgart.de" in _otp_store
        assert len(_otp_store["test@uni-stuttgart.de"]["code"]) == OTP_LENGTH

    def test_rate_limit_enforced(self):
        email = "ratelimit@uni-stuttgart.de"
        for _ in range(3):
            ok, _ = send_otp(email)
            assert ok
            if email in _otp_store:
                _otp_store[email]["last_sent"] = 0
        ok, msg = send_otp(email)
        assert not ok
        assert "Too many requests" in msg

    def test_resend_cooldown_enforced(self):
        email = "cooldown@uni-stuttgart.de"
        ok, _ = send_otp(email)
        assert ok
        ok, msg = send_otp(email)
        assert not ok
        assert "wait" in msg.lower()


class TestVerifyOtp:
    def test_correct_otp_succeeds(self):
        send_otp("verify@uni-stuttgart.de")
        code = _otp_store["verify@uni-stuttgart.de"]["code"]
        ok, msg = verify_otp("verify@uni-stuttgart.de", code)
        assert ok
        assert "verify@uni-stuttgart.de" not in _otp_store

    def test_wrong_otp_fails(self):
        send_otp("wrong@uni-stuttgart.de")
        ok, msg = verify_otp("wrong@uni-stuttgart.de", "000000")
        assert not ok

    def test_three_failed_attempts_invalidates(self):
        email = "exhaust@uni-stuttgart.de"
        send_otp(email)
        for _ in range(MAX_FAILED_ATTEMPTS):
            ok, msg = verify_otp(email, "000000")
            assert not ok
        assert email not in _otp_store
        assert "Too many" in msg

    def test_expired_otp_fails(self):
        email = "expired@uni-stuttgart.de"
        send_otp(email)
        _otp_store[email]["expiry"] = time.time() - 10
        ok, msg = verify_otp(email, _otp_store[email]["code"])
        assert not ok
        assert "expired" in msg.lower()

    def test_no_otp_for_email_fails(self):
        ok, msg = verify_otp("never_sent@uni-stuttgart.de", "123456")
        assert not ok


class TestHtmlEmails:
    def test_email_wrapper_contains_branding(self):
        from solarjv_analyzer.auth.email_service import _EMAIL_WRAPPER
        assert "ipv" in _EMAIL_WRAPPER
        assert "University of Stuttgart" in _EMAIL_WRAPPER

    def test_otp_email_contains_code_placeholder(self):
        from solarjv_analyzer.auth.email_service import OTP_EMAIL_BODY
        assert "%s" in OTP_EMAIL_BODY

    def test_registration_email_contains_username_placeholder(self):
        from solarjv_analyzer.auth.email_service import REGISTRATION_SUCCESS_BODY
        assert "%s" in REGISTRATION_SUCCESS_BODY

    def test_is_configured_detects_no_creds(self, monkeypatch):
        monkeypatch.setattr(
            "solarjv_analyzer.auth.email_service._smtp_config", None
        )
        monkeypatch.setenv("SMTP_USER", "")
        monkeypatch.setenv("SMTP_PASSWORD", "")
        assert not is_configured()


class TestMaskEmail:
    def test_standard_email(self):
        masked = mask_email("st000000@stud.uni-stuttgart.de")
        assert "@" in masked
        assert "stud.uni-stuttgart.de" in masked
        assert masked.count("*") >= 4


class TestClearOtp:
    def test_clear_removes_entry(self):
        send_otp("clear@uni-stuttgart.de")
        assert "clear@uni-stuttgart.de" in _otp_store
        clear_otp("clear@uni-stuttgart.de")
        assert "clear@uni-stuttgart.de" not in _otp_store
