"""
Email Service for SolarJV Analyzer authentication.

Sends HTML-formatted emails via SMTP for OTP verification,
registration confirmation, and password-reset confirmation.

THE SMTP PASSWORD IS NEVER STORED IN THIS FILE, IN A CONFIG FILE, OR ON THE
S: DRIVE. It lives in the operating system's credential store (Windows
Credential Manager via DPAPI, Keychain on macOS), encrypted at rest under a
key the OS derives from the machine and account -- a key that is not in this
program and does not ship in the installer.

An admin puts it there once per lab PC:

    python tools/setup_smtp_credential.py

Everything else -- host, port, sender address -- is NOT secret and may still
come from a shared config file, so those stay centrally editable.

If no credential is found the app says so and refuses to send. It does NOT
fall back to a built-in value: a silent fallback is exactly how a real
password came to be committed here in the first place.
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

# Non-secret defaults only. There is deliberately no "password" key here --
# see the structural test in tests/auth/test_email_service.py, which fails if
# one reappears.
_DEFAULT_SMTP = {
    "host": "smtp.uni-stuttgart.de",
    "port": 587,
    "user": "",
    "from_addr": "",
    "from_name": "SolarJV Analyzer",
    "use_tls": True,
}

# Identifies this application's entry in the OS credential store.
KEYRING_SERVICE = "SolarJV Analyzer SMTP"

# Settings that may come from a shared config file. "password" is absent on
# purpose: a shared file is readable by everyone the share is readable by.
_FILE_SETTINGS = ("host", "port", "user", "from_addr", "from_name")

_smtp_config = None


def _resolve_config_paths() -> list:
    """Where to look for email_config.json: the local machine, only.

    The S: drive used to be checked first. It is gone: the setup script
    writes this file during commissioning, so every machine has its own, and
    a shared config on the share only ever masked it.
    """
    return [os.path.join(os.path.expanduser("~"), ".solarjv", "email_config.json")]


def _password_from_keyring(user: str) -> str:
    """Ask the OS credential store for this user's SMTP password.

    Returns "" for every failure -- not installed, nothing stored, store
    locked. The caller treats "not configured" and "could not be read" the
    same way: refuse to send and say so.
    """
    if not user:
        return ""
    try:
        import keyring
    except ImportError:
        logger.warning(
            "keyring is not installed, so the SMTP password cannot be read "
            "from the OS credential store.")
        return ""
    try:
        return keyring.get_password(KEYRING_SERVICE, user) or ""
    except Exception as exc:                          # noqa: BLE001
        logger.error(f"Could not read the SMTP password from the credential "
                     f"store: {exc}")
        return ""


def _get_smtp_config() -> dict:
    """Assemble the SMTP settings.

    Non-secret settings: built-in defaults, then a config file, then
    environment variables. The password comes from the OS credential store,
    or from SMTP_PASSWORD when that is set (for development and CI).
    """
    global _smtp_config
    if _smtp_config is not None:
        return _smtp_config

    cfg = dict(_DEFAULT_SMTP)

    import json
    # EVERY config file is consulted, in priority order, and each key is taken
    # from the FIRST file that defines it.
    #
    # This used to `break` at the first readable file. On a lab PC that meant
    # a shared S: config -- which may predate per-machine commissioning and
    # need not name a user at all -- silently masked the local file the setup
    # script writes. The password sat in the vault under a username the
    # application never saw, and it reported "not configured" with no way to
    # tell why. Falling through keeps S: authoritative for what it does
    # define, without letting it hide what it does not.
    sources = {}
    for path in _resolve_config_paths():
        try:
            with open(path) as f:
                file_cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            continue
        if "password" in file_cfg:
            # Refusing rather than using it is the point. This file lives on
            # a share every whitelisted lab user can read; honouring it would
            # keep the plaintext password alive and nothing would improve.
            logger.error(
                f"{path} contains a 'password' field. It is being IGNORED -- "
                f"a shared config file is readable by everyone the share is "
                f"readable by. Remove that field, and store the password with "
                f"tools/setup_smtp_credential.py instead.")
        taken = []
        for k in _FILE_SETTINGS:
            if k in file_cfg and file_cfg[k] not in ("", None) and k not in sources:
                cfg[k] = file_cfg[k]
                sources[k] = path
                taken.append(k)
        if taken:
            logger.info(f"SMTP settings {', '.join(taken)} from {path}")
        else:
            logger.debug(f"SMTP config {path} added nothing new.")

    for key in _FILE_SETTINGS:
        env_val = os.environ.get(f"SMTP_{key.upper()}")
        if env_val is not None:
            cfg[key] = env_val
    try:
        cfg["port"] = int(cfg["port"])
    except (TypeError, ValueError):
        logger.error(f"Bad SMTP port {cfg['port']!r}; falling back to 587.")
        cfg["port"] = 587
    cfg["use_tls"] = os.environ.get("SMTP_USE_TLS", "1") == "1"

    # The secret, last: an explicit env var wins (development, CI), otherwise
    # the OS credential store. Nothing else, and no built-in fallback.
    cfg["password"] = (os.environ.get("SMTP_PASSWORD")
                       or _password_from_keyring(cfg.get("user", "")))

    _smtp_config = cfg
    return cfg


def reset_config_cache() -> None:
    """Forget the cached settings so the next call re-reads them.

    The config is read once and cached for the life of the process. After an
    admin stores a new password the running app would otherwise keep using
    the old one.
    """
    global _smtp_config
    _smtp_config = None


def is_configured() -> bool:
    """True when there is a username AND a password to send with."""
    cfg = _get_smtp_config()
    return bool(cfg.get("user") and cfg.get("password"))


def configuration_problem() -> str:
    """A sentence naming what is missing, or "" when all is well.

    Worth having: "could not send the email" is a dead end for whoever is
    standing at the lab PC, and the fix is usually one command.
    """
    cfg = _get_smtp_config()
    if not cfg.get("user"):
        return ("No SMTP username is configured. Set it in the email config "
                "file or the SMTP_USER environment variable.")
    if not cfg.get("password"):
        return (f"No SMTP password is stored on this machine for "
                f"'{cfg['user']}'. An administrator can add it by running:\n\n"
                f"    python tools/setup_smtp_credential.py")
    return ""


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

    problem = configuration_problem()
    if problem:
        # Log the specific cause for whoever has to fix it; tell the operator
        # something they can act on rather than a dead end.
        logger.warning(f"SMTP not configured — email not sent. {problem}")
        return False, ("Email service is not configured on this machine. "
                       "Contact the lab administrator.")

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
