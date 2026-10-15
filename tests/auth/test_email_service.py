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


# ---------------------------------------------------------------------------
# Where the SMTP password comes from
#
# A real university password was committed in this module once. It got there
# because _DEFAULT_SMTP held a working fallback, so nothing ever failed and
# nothing ever complained. These tests exist so that cannot happen quietly
# again.
# ---------------------------------------------------------------------------

import ast
import json
import os as _os
import pathlib

import pytest

from solarjv_analyzer.auth import email_service as _es


class _MemoryKeyring:
    """Stands in for Windows Credential Manager."""

    def __init__(self):
        self.store = {}

    def set_password(self, service, user, password):
        self.store[(service, user)] = password

    def get_password(self, service, user):
        return self.store.get((service, user))

    def delete_password(self, service, user):
        self.store.pop((service, user), None)


@pytest.fixture
def smtp_env(monkeypatch, tmp_path):
    """A clean slate: no env vars, no config file, empty credential store."""
    for key in ("HOST", "PORT", "USER", "PASSWORD", "FROM_ADDR",
                "FROM_NAME", "USE_TLS"):
        monkeypatch.delenv(f"SMTP_{key}", raising=False)
    monkeypatch.setattr(_es, "_resolve_config_paths", lambda: [])
    _es.reset_config_cache()

    fake = _MemoryKeyring()
    import keyring
    monkeypatch.setattr(keyring, "get_password",
                        lambda s, u: fake.get_password(s, u))
    monkeypatch.setattr(keyring, "set_password",
                        lambda s, u, p: fake.set_password(s, u, p))
    yield fake
    _es.reset_config_cache()


def test_no_credential_is_hardcoded_in_the_auth_package():
    """A structural guard, not a spot check.

    Walks every assignment in the auth package and fails on a non-empty
    string literal assigned to anything that looks like a secret. This is the
    test that would have caught the committed password.
    """
    suspicious = ("password", "passwd", "secret", "token", "api_key", "apikey")
    offenders = []
    auth_dir = pathlib.Path(_es.__file__).parent

    def looks_like_a_secret(value):
        """A credential is short and has no whitespace.

        Without this, every email template constant whose name contains
        "password" -- PASSWORD_RESET_BODY and friends -- would be flagged,
        and a guard that cries wolf gets deleted by the next person.
        """
        return (isinstance(value, str) and value
                and len(value) <= 100
                and not any(c.isspace() for c in value))

    for path in sorted(auth_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # key: "literal" inside a dict
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                            and key.value.lower() in suspicious
                            and isinstance(value, ast.Constant)
                            and looks_like_a_secret(value.value)):
                        offenders.append(f"{path.name}:{key.lineno} {key.value}")
            # NAME = "literal"
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Name)
                            and any(s in target.id.lower() for s in suspicious)
                            and isinstance(node.value, ast.Constant)
                            and looks_like_a_secret(node.value.value)):
                        offenders.append(f"{path.name}:{node.lineno} {target.id}")

    assert not offenders, (
        "a credential appears to be hardcoded in the auth package: "
        + ", ".join(offenders))


def test_without_a_stored_password_the_app_refuses_and_says_how_to_fix_it(smtp_env,
                                                                          monkeypatch):
    monkeypatch.setenv("SMTP_USER", "fn-example")
    _es.reset_config_cache()

    assert _es.is_configured() is False
    problem = _es.configuration_problem()
    assert "fn-example" in problem
    assert "setup_smtp_credential" in problem, (
        "the message should name the command that fixes it")


def test_the_password_comes_from_the_credential_store(smtp_env, monkeypatch):
    monkeypatch.setenv("SMTP_USER", "fn-example")
    smtp_env.set_password(_es.KEYRING_SERVICE, "fn-example", "stored-secret")
    _es.reset_config_cache()

    assert _es.is_configured() is True
    assert _es._get_smtp_config()["password"] == "stored-secret"
    assert _es.configuration_problem() == ""


def test_an_env_var_overrides_the_credential_store(smtp_env, monkeypatch):
    """So a developer can run against a test mailbox without touching it."""
    monkeypatch.setenv("SMTP_USER", "fn-example")
    monkeypatch.setenv("SMTP_PASSWORD", "from-the-environment")
    smtp_env.set_password(_es.KEYRING_SERVICE, "fn-example", "stored-secret")
    _es.reset_config_cache()

    assert _es._get_smtp_config()["password"] == "from-the-environment"


def test_a_password_in_a_shared_config_file_is_ignored(smtp_env, monkeypatch,
                                                       tmp_path, caplog):
    """The S: config is readable by every whitelisted lab user.

    Honouring a password there would keep the plaintext alive and nothing
    would have improved, so it is read, refused, and logged.
    """
    cfg_file = tmp_path / "solarjv_email_config.json"
    cfg_file.write_text(json.dumps({
        "host": "smtp.example.invalid",
        "user": "fn-example",
        "password": "plaintext-on-a-share",
        "from_addr": "no-reply@example.invalid",
    }), encoding="utf-8")
    monkeypatch.setattr(_es, "_resolve_config_paths", lambda: [str(cfg_file)])
    _es.reset_config_cache()

    with caplog.at_level("ERROR"):
        cfg = _es._get_smtp_config()

    # the non-secret settings are honoured...
    assert cfg["host"] == "smtp.example.invalid"
    assert cfg["user"] == "fn-example"
    assert cfg["from_addr"] == "no-reply@example.invalid"
    # ...and the password is not
    assert cfg["password"] != "plaintext-on-a-share"
    assert not _es.is_configured()
    assert any("IGNORED" in r.message for r in caplog.records), (
        "ignoring it silently is nearly as bad as honouring it")


def test_a_broken_credential_store_does_not_crash_the_app(smtp_env, monkeypatch):
    """A locked or missing store must degrade to "not configured"."""
    monkeypatch.setenv("SMTP_USER", "fn-example")
    import keyring

    def boom(service, user):
        raise RuntimeError("credential store unavailable")

    monkeypatch.setattr(keyring, "get_password", boom)
    _es.reset_config_cache()

    assert _es.is_configured() is False      # not an exception


def test_the_username_must_be_discoverable_or_the_password_is_useless(smtp_env,
                                                                      monkeypatch,
                                                                      tmp_path):
    """The app looks the password up BY USERNAME.

    Storing a credential without making the username discoverable leaves
    `is_configured()` False with a perfectly good password sitting in the
    vault -- which is what happened when the hardcoded username was removed
    and nothing replaced it.
    """
    smtp_env.set_password(_es.KEYRING_SERVICE, "fn-noreply", "stored-secret")
    _es.reset_config_cache()

    # No username anywhere: the vault entry cannot be found.
    assert _es.is_configured() is False

    # A config file supplying only the NON-SECRET username is enough.
    cfg_file = tmp_path / "email_config.json"
    cfg_file.write_text(json.dumps({
        "user": "fn-noreply",
        "from_addr": "noreply@example.invalid",
    }), encoding="utf-8")
    monkeypatch.setattr(_es, "_resolve_config_paths", lambda: [str(cfg_file)])
    _es.reset_config_cache()

    assert _es.is_configured() is True
    assert _es._get_smtp_config()["password"] == "stored-secret"


def test_the_setup_script_never_writes_a_password_to_the_config():
    """A structural guard on the commissioning script."""
    import ast
    import pathlib

    script = (pathlib.Path(_es.__file__).parents[2].parent
              / "tools" / "setup_smtp_credential.py")
    if not script.exists():
        pytest.skip("tools/ is not present in this checkout")

    source = script.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # the only thing allowed to receive the secret is keyring.set_password
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("dump", "dumps", "write"):
                dumped = ast.dump(node)
                assert "secret" not in dumped, (
                    "the setup script appears to serialise the password")
    assert 'settings.pop("password", None)' in source, (
        "the script must strip any password from the config it writes")


def _setup_script_path():
    import pathlib
    return (pathlib.Path(_es.__file__).parents[2].parent
            / "tools" / "setup_smtp_credential.py")


def test_the_setup_script_holds_config_but_never_a_password():
    """The lab constants are fine; the password is not.

    Host, port, account name and sender address are not secrets and belong
    in a file so commissioning is one command. The password is the one thing
    that must not be there -- this file is tracked by git and the repository
    pushes to GitHub, which is exactly how the last credential got out.
    """
    import ast

    script = _setup_script_path()
    if not script.exists():
        pytest.skip("tools/ is not present in this checkout")
    tree = ast.parse(script.read_text(encoding="utf-8"))

    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    constants[target.id] = node.value.value

    # the config knobs exist...
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_FROM_ADDR"):
        assert name in constants, f"{name} is missing from the lab config block"

    # ...and nothing password-shaped carries a value
    for name, value in constants.items():
        if any(w in name.lower() for w in ("password", "passwd", "secret", "token")):
            assert not (isinstance(value, str) and value.strip()), (
                f"{name} has a value in the setup script -- it is tracked by "
                f"git and pushes to GitHub")


def test_the_script_refuses_to_run_with_a_hardcoded_password():
    """Not just absent -- actively rejected, so it cannot creep back."""
    script = _setup_script_path()
    if not script.exists():
        pytest.skip("tools/ is not present in this checkout")
    source = script.read_text(encoding="utf-8")
    assert "_refuse_a_hardcoded_password" in source
    assert "if _refuse_a_hardcoded_password():" in source, (
        "the guard exists but is never called")


def test_a_password_file_outside_the_repo_is_required():
    """--password-file exists for commissioning several machines, but it must
    not be allowed to sit in the working tree."""
    script = _setup_script_path()
    if not script.exists():
        pytest.skip("tools/ is not present in this checkout")
    source = script.read_text(encoding="utf-8")
    assert "--password-file" in source
    assert "is inside the repository" in source, (
        "nothing stops the password file being kept next to the code")


def test_the_real_credential_file_is_gitignored():
    """The password lives in tools/smtp_local_config.py, and the ONLY thing
    keeping it out of the university's GitHub org is this .gitignore line.

    `git add .` is the realistic way it would get committed, so the rule has
    to be there, not merely remembered.
    """
    import pathlib

    root = pathlib.Path(_es.__file__).parents[2].parent
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        pytest.skip("no .gitignore in this checkout")

    rules = gitignore.read_text(encoding="utf-8")
    assert "tools/smtp_local_config.py" in rules, (
        "the file that holds the SMTP password is NOT gitignored")

    example = root / "tools" / "smtp_local_config.example.py"
    if example.exists():
        source = example.read_text(encoding="utf-8")
        import ast
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if (isinstance(target, ast.Name)
                            and "password" in target.id.lower()):
                        assert not (isinstance(node.value.value, str)
                                    and node.value.value.strip()), (
                            "the EXAMPLE file is tracked and has a real "
                            "password in it")


def test_a_shared_config_does_not_hide_the_local_one(smtp_env, monkeypatch,
                                                     tmp_path):
    """Reported from the lab: the setup script wrote
    C:\\Users\\<id>\\.solarjv\\email_config.json, the credential was in the
    vault, and the app still said "not configured".

    The loader stopped at the FIRST readable config file. A shared S: config
    -- which predates per-machine commissioning and need not name a user --
    masked the local file entirely, so the username was never found and the
    vault lookup had nothing to key on.
    """
    shared = tmp_path / "S_solarjv_email_config.json"
    shared.write_text(json.dumps({
        "host": "smtp.shared.invalid",          # the shared file DOES set this
        "from_name": "Lab",
    }), encoding="utf-8")

    local = tmp_path / "local_email_config.json"
    local.write_text(json.dumps({
        "user": "fn-noreply",                   # only the LOCAL file has this
        "from_addr": "noreply@example.invalid",
    }), encoding="utf-8")

    monkeypatch.setattr(_es, "_resolve_config_paths",
                        lambda: [str(shared), str(local)])
    smtp_env.set_password(_es.KEYRING_SERVICE, "fn-noreply", "stored-secret")
    _es.reset_config_cache()

    cfg = _es._get_smtp_config()
    assert cfg["host"] == "smtp.shared.invalid", (
        "the shared config should still win for what it defines")
    assert cfg["user"] == "fn-noreply", (
        "the local config was masked -- this is the reported bug")
    assert _es.is_configured() is True


def test_the_shared_config_still_wins_where_both_define_a_key(smtp_env,
                                                              monkeypatch,
                                                              tmp_path):
    """Falling through must not invert the priority order."""
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({"user": "shared-account"}), encoding="utf-8")
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"user": "local-account"}), encoding="utf-8")

    monkeypatch.setattr(_es, "_resolve_config_paths",
                        lambda: [str(shared), str(local)])
    _es.reset_config_cache()

    assert _es._get_smtp_config()["user"] == "shared-account"


def test_an_empty_value_does_not_count_as_defined(smtp_env, monkeypatch,
                                                  tmp_path):
    """A config file with "user": "" must not block a later one."""
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps({"user": "", "host": "h"}), encoding="utf-8")
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"user": "fn-noreply"}), encoding="utf-8")

    monkeypatch.setattr(_es, "_resolve_config_paths",
                        lambda: [str(shared), str(local)])
    _es.reset_config_cache()

    assert _es._get_smtp_config()["user"] == "fn-noreply"
