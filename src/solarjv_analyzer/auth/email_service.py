"""
Email Service for SolarJV Analyzer authentication.

Sends HTML-formatted emails via SMTP for OTP verification,
registration confirmation, and password-reset confirmation.
Credentials are read from environment variables; never hardcoded.
"""

import logging
import os
import random
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory OTP store
# ---------------------------------------------------------------------------
_otp_store: dict = {}
_request_tracker: dict = {}

OTP_LENGTH = 6
OTP_EXPIRY_SECONDS = 600
MAX_FAILED_ATTEMPTS = 3
MAX_REQUESTS_PER_WINDOW = 3
RATE_WINDOW_SECONDS = 600
RESEND_COOLDOWN_SECONDS = 60

# ---------------------------------------------------------------------------
# SMTP configuration
# ---------------------------------------------------------------------------

_DEFAULT_SMTP = {
    "host": "smtp.uni-stuttgart.de",
    "port": 587,
    "user": "",
    "password": "",
    "from_addr": "solarjv@ipv.uni-stuttgart.de",
    "from_name": "SolarJV Analyzer",
    "use_tls": True,
}

_smtp_config = None


def _resolve_config_paths() -> list:
    """Return ordered list of paths to check for email_config.json.
    S: drive root takes priority on Windows (lab deployment)."""
    import sys
    paths = []
    if sys.platform == "win32":
        paths.append("S:\\solarjv_email_config.json")
    paths.append(os.path.join(os.path.expanduser("~"), ".solarjv", "email_config.json"))
    return paths


def _get_smtp_config() -> dict:
    """Read SMTP settings from config file or environment variables.
    Priority: S: drive (lab) → env vars → ~/.solarjv/ (fallback)."""
    global _smtp_config
    if _smtp_config is not None:
        return _smtp_config

    cfg = dict(_DEFAULT_SMTP)

    # 1. Try file-based configs (S: drive first, then local)
    import json
    for path in _resolve_config_paths():
        try:
            with open(path) as f:
                file_cfg = json.load(f)
            for k in ("host", "port", "user", "password", "from_addr", "from_name"):
                if k in file_cfg:
                    cfg[k] = file_cfg[k]
            cfg["port"] = int(cfg["port"])
            logger.info(f"SMTP config loaded from {path}")
            break
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            continue

    # 2. Environment variable overrides (for admin/debug use)
    for key in ("host", "port", "user", "password", "from_addr", "from_name"):
        env_val = os.environ.get(f"SMTP_{key.upper()}")
        if env_val is not None:
            cfg[key] = env_val
    cfg["port"] = int(cfg["port"])
    cfg["use_tls"] = os.environ.get("SMTP_USE_TLS", "1") == "1"

    _smtp_config = cfg
    return cfg


def is_configured() -> bool:
    """Return True if SMTP credentials are set."""
    cfg = _get_smtp_config()
    return bool(cfg.get("user") and cfg.get("password"))


# ---------------------------------------------------------------------------
# HTML email templates
# ---------------------------------------------------------------------------

_EMAIL_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
<table width="100%%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:40px 0;">
<tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">
  <!-- header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%%,#065f46 100%%);padding:28px 32px;text-align:center;">
      <span style="color:#10b981;font-size:20px;font-weight:800;letter-spacing:2px;">ipv</span>
    </td>
  </tr>
  <!-- body -->
  <tr>
    <td style="padding:32px;">
      %s
    </td>
  </tr>
  <!-- footer -->
  <tr>
    <td style="background-color:#f8fafc;padding:18px 32px;text-align:center;border-top:1px solid #e2e8f0;">
      <p style="margin:0;font-size:11px;color:#94a3b8;">
        Institut für Photovoltaik (ipv) &middot; University of Stuttgart<br>
        Pfaffenwaldring 47 &middot; 70569 Stuttgart &middot; Germany
      </p>
    </td>
  </tr>
</table>
</td></tr>
</table>
</body>
</html>"""

OTP_EMAIL_BODY = """\
<h2 style="margin:0 0 8px;font-size:20px;color:#1e293b;">Verify Your Email</h2>
<p style="margin:0 0 24px;font-size:14px;color:#64748b;line-height:1.5;">
  Use the verification code below to complete your action in SolarJV Analyzer.
</p>
<div style="background-color:#f0fdf4;border:1px solid #d1fae5;border-radius:12px;padding:20px;text-align:center;margin-bottom:24px;">
  <span style="font-family:'SF Mono',Menlo,Consolas,monospace;font-size:32px;font-weight:800;color:#10b981;letter-spacing:8px;">%s</span>
</div>
<p style="margin:0;font-size:13px;color:#94a3b8;line-height:1.5;">
  This code will expire in <strong>10 minutes</strong>.<br>
  If you did not request this, please ignore this email.
</p>"""

REGISTRATION_SUCCESS_BODY = """\
<div style="text-align:center;margin-bottom:20px;">
  <div style="display:inline-block;width:56px;height:56px;background-color:#d1fae5;border-radius:50%%;line-height:56px;font-size:28px;">&#10003;</div>
</div>
<h2 style="margin:0 0 8px;font-size:20px;color:#1e293b;text-align:center;">Registration Successful</h2>
<p style="margin:0 0 16px;font-size:14px;color:#64748b;line-height:1.5;text-align:center;">
  Welcome to SolarJV Analyzer! Your account has been created.
</p>
<div style="background-color:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:20px;">
  <table width="100%%" cellpadding="4" cellspacing="0">
    <tr><td style="font-size:13px;color:#64748b;">University ID</td><td style="font-size:14px;color:#1e293b;font-weight:600;">%s</td></tr>
  </table>
</div>
<p style="margin:0;font-size:13px;color:#94a3b8;line-height:1.5;text-align:center;">
  You can now sign in and start measuring.<br>
  Questions? Contact your lab administrator.
</p>"""

PASSWORD_RESET_BODY = """\
<div style="text-align:center;margin-bottom:20px;">
  <div style="display:inline-block;width:56px;height:56px;background-color:#fef3c7;border-radius:50%%;line-height:56px;font-size:28px;">&#128274;</div>
</div>
<h2 style="margin:0 0 8px;font-size:20px;color:#1e293b;text-align:center;">Password Reset Confirmed</h2>
<p style="margin:0 0 16px;font-size:14px;color:#64748b;line-height:1.5;text-align:center;">
  Your SolarJV Analyzer account password has been successfully changed.
</p>
<div style="background-color:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:16px;margin-bottom:20px;">
  <p style="margin:0;font-size:13px;color:#dc2626;line-height:1.5;">
    <strong>&#9888; If you did not make this change</strong>, please contact your lab administrator immediately to secure your account.
  </p>
</div>
<p style="margin:0;font-size:13px;color:#94a3b8;line-height:1.5;text-align:center;">
  You can now sign in with your new password.
</p>"""


# ---------------------------------------------------------------------------
# Core: send an HTML email
# ---------------------------------------------------------------------------


def _send_email(to_email: str, subject: str, html_body: str) -> Tuple[bool, str]:
    """Build a multipart email with plain-text fallback and send via SMTP."""
    cfg = _get_smtp_config()

    if not cfg["user"] or not cfg["password"]:
        logger.warning("SMTP not configured — email not sent.")
        return False, "Email service is not configured. Contact the lab administrator."

    # Strip HTML tags for plain-text fallback
    import re
    plain = re.sub(r"<[^>]+>", "", html_body)
    plain = re.sub(r"\s+", " ", plain).strip()

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f'{cfg["from_name"]} <{cfg["from_addr"]}>'
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(plain, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        if cfg["port"] == 465:
            server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=10)
        else:
            server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=10)
            if cfg["use_tls"]:
                server.starttls()

        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["from_addr"], to_email, msg.as_string())
        server.quit()
        logger.info(f"Email sent to {to_email}: {subject}")
        return True, ""

    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP authentication failed")
        return False, "Email service configuration error. Contact the lab administrator."
    except smtplib.SMTPConnectError:
        logger.error(f"SMTP connection failed to {cfg['host']}:{cfg['port']}")
        return False, "Could not connect to email server. Please try again later."
    except Exception as e:
        logger.error(f"SMTP error: {e}")
        return False, "Could not send email. Please try again."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def send_otp(email: str) -> Tuple[bool, str]:
    """Generate an OTP, store it in memory, and send it via SMTP."""
    email = email.strip().lower()

    now = time.time()
    cutoff = now - RATE_WINDOW_SECONDS
    _request_tracker.setdefault(email, [])
    _request_tracker[email] = [t for t in _request_tracker[email] if t > cutoff]
    if len(_request_tracker[email]) >= MAX_REQUESTS_PER_WINDOW:
        return False, "Too many requests. Please wait before trying again."

    existing = _otp_store.get(email)
    if existing and (now - existing["last_sent"]) < RESEND_COOLDOWN_SECONDS:
        remaining = int(RESEND_COOLDOWN_SECONDS - (now - existing["last_sent"]))
        return False, f"Please wait {remaining}s before requesting a new code."

    code = _generate_otp()
    _otp_store[email] = {
        "code": code, "expiry": now + OTP_EXPIRY_SECONDS,
        "attempts": 0, "last_sent": now,
    }
    _request_tracker[email].append(now)

    html = _EMAIL_WRAPPER % (OTP_EMAIL_BODY % code)
    ok, err = _send_email(email, "SolarJV Analyzer — Verification Code", html)

    if not ok and is_configured():
        # SMTP failed — clean up the stored OTP
        _otp_store.pop(email, None)
    return ok, err


def send_registration_confirmation(email: str, username: str) -> None:
    """Send a welcome email confirming successful registration."""
    html = _EMAIL_WRAPPER % (REGISTRATION_SUCCESS_BODY % username)
    _send_email(email, "Welcome to SolarJV Analyzer", html)


def send_password_reset_confirmation(email: str) -> None:
    """Send a confirmation email after a successful password reset."""
    html = _EMAIL_WRAPPER % PASSWORD_RESET_BODY
    _send_email(email, "SolarJV Analyzer — Password Reset Confirmed", html)


def verify_otp(email: str, code: str) -> Tuple[bool, str]:
    """Verify a submitted OTP."""
    email = email.strip().lower()
    entry = _otp_store.get(email)

    if entry is None:
        return False, "No verification code was requested for this email."
    if time.time() > entry["expiry"]:
        del _otp_store[email]
        return False, "Code has expired. Please request a new one."
    if entry["attempts"] >= MAX_FAILED_ATTEMPTS:
        del _otp_store[email]
        return False, "Too many failed attempts. Please request a new code."
    if code.strip() != entry["code"]:
        entry["attempts"] += 1
        remaining = MAX_FAILED_ATTEMPTS - entry["attempts"]
        if remaining <= 0:
            del _otp_store[email]
            return False, "Too many failed attempts. Please request a new code."
        return False, f"Invalid code. {remaining} attempt{'s' if remaining > 1 else ''} remaining."

    del _otp_store[email]
    return True, ""


def clear_otp(email: str) -> None:
    _otp_store.pop(email.strip().lower(), None)


def mask_email(email: str) -> str:
    """Privacy-safe masked email."""
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    visible = local[:2] if len(local) > 2 else local[0]
    return f"{visible}{'*' * max(4, len(local) - (1 if len(local) <= 2 else 2))}@{domain}"


def _generate_otp() -> str:
    return str(random.randint(0, 999999)).zfill(OTP_LENGTH)
