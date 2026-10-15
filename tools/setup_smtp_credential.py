#!/usr/bin/env python3
"""Store this machine's SMTP password in the OS credential store.

Run ONCE per lab PC, as the account the app runs under:

    python tools/setup_smtp_credential.py

The password is typed in, never passed as an argument (arguments show up in
shell history and in the process list) and never echoed. It goes straight
into Windows Credential Manager -- encrypted at rest by DPAPI under a key
Windows derives from this machine and account -- and is never written to a
file, to the repository, or to the S: drive.

    --user NAME   set the username instead of taking it from the config
    --check       report whether a password is stored, without changing it
    --test        after storing, try a real SMTP login to prove it works
    --delete      remove the stored password from this machine

Nothing here prints the password, and --test reports only success or the
server's own error message.
"""

import argparse
import getpass
import os
import sys

# ---------------------------------------------------------------------------
# LAB CONFIGURATION -- fill in once for this institute, then this script runs
# with no arguments and asks only for the password.
#
# None of these is a secret: a hostname, a port, an account name and a sender
# address. They are exactly the things that belong in a file.
#
# THE PASSWORD DOES NOT GO HERE, and the script refuses to run if someone
# adds it. This file is tracked by git and the repository pushes to GitHub --
# which is precisely how the last credential got out -- and the lab PCs share
# one Windows account, so a file in that account's Documents folder is not
# "admin only" in any useful sense. The password is typed once per machine
# and lives in Windows Credential Manager, encrypted to that machine.
# ---------------------------------------------------------------------------
SMTP_HOST = "smtp.uni-stuttgart.de"
SMTP_PORT = 587
SMTP_USER = ""                       # <-- the no-reply account name
SMTP_FROM_ADDR = ""                  # <-- address the OTP mail comes from
SMTP_FROM_NAME = "SolarJV Analyzer"


def _local_config():
    """`tools/smtp_local_config.py`, if the admin has created one.

    That file is gitignored, so it is the one place a password can sit
    without being swept into a commit. See smtp_local_config.example.py.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import smtp_local_config as local
        return local
    except ImportError:
        return None


def _lab_defaults() -> dict:
    """The constants above, overridden by the local config when present."""
    values = {
        "host": SMTP_HOST,
        "port": SMTP_PORT,
        "user": SMTP_USER,
        "from_addr": SMTP_FROM_ADDR,
        "from_name": SMTP_FROM_NAME,
    }
    local = _local_config()
    if local is not None:
        for key, attr in (("host", "SMTP_HOST"), ("port", "SMTP_PORT"),
                          ("user", "SMTP_USER"),
                          ("from_addr", "SMTP_FROM_ADDR"),
                          ("from_name", "SMTP_FROM_NAME")):
            found = getattr(local, attr, None)
            if found not in ("", None):
                values[key] = found
    return {k: v for k, v in values.items() if v not in ("", None)}


def _local_password() -> str:
    """The password from the gitignored local config, or ""."""
    local = _local_config()
    if local is None:
        return ""
    return (getattr(local, "SMTP_PASSWORD", "") or "").strip()


def _refuse_a_hardcoded_password() -> bool:
    """Stop if a password constant has been added to this file.

    Not paranoia: the previous credential reached GitHub exactly this way,
    and a script that quietly accepted one would make that the easy path
    again.
    """
    for name in ("SMTP_PASSWORD", "SMTP_PASS", "PASSWORD", "SMTP_SECRET"):
        value = globals().get(name)
        if isinstance(value, str) and value.strip():
            print(f"ERROR: {name} is set in this file.\n"
                  f"       This file is tracked by git and the repository "
                  f"pushes to GitHub.\n"
                  f"       Remove it -- the password is typed in, once per "
                  f"machine.", file=sys.stderr)
            return True
    return False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", help="SMTP username to store the password under")
    parser.add_argument("--check", action="store_true",
                        help="report whether a password is stored")
    parser.add_argument("--diagnose", action="store_true",
                        help="show exactly what the APPLICATION sees and why "
                             "it decides configured / not configured. Run this "
                             "with the same Python that runs the app.")
    parser.add_argument("--test", action="store_true",
                        help="verify by logging in to the SMTP server")
    parser.add_argument("--delete", action="store_true",
                        help="remove the stored password from this machine")
    parser.add_argument("--password-file", metavar="PATH",
                        help="read the password from this file instead of "
                             "typing it. Keep it OUTSIDE the repository -- a "
                             "USB stick or a protected folder -- and delete "
                             "it when the machines are commissioned.")
    args = parser.parse_args()

    if _refuse_a_hardcoded_password():
        return 2

    if args.diagnose:
        return _diagnose()

    try:
        import keyring
    except ImportError:
        print("ERROR: the 'keyring' package is not installed.\n"
              "       pip install keyring", file=sys.stderr)
        return 2

    from solarjv_analyzer.auth import email_service

    # Lab constants first, then anything the machine already provides
    # (a shared S: config or SMTP_* environment variables still win).
    cfg = dict(_lab_defaults())
    cfg.update({k: v for k, v in email_service._get_smtp_config().items()
                if v not in ("", None)})
    user = args.user or cfg.get("user") or ""
    if not user and not (args.check or args.delete):
        # Ask rather than refuse. This script is how a lab PC gets
        # commissioned, and erroring out on the one thing it could simply
        # ask for turns a 30-second job into a documentation hunt.
        print("No SMTP username is configured on this machine yet.")
        try:
            user = input("SMTP username: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.", file=sys.stderr)
            return 1
    if not user:
        print("ERROR: no SMTP username. Pass --user NAME, or set one in the\n"
              "       email config file.", file=sys.stderr)
        return 2

    service = email_service.KEYRING_SERVICE
    try:
        backend_obj = keyring.get_keyring()
    except Exception as exc:                          # noqa: BLE001
        print(f"ERROR: no usable credential store on this machine: {exc}",
              file=sys.stderr)
        return 2
    backend = type(backend_obj).__name__
    if "fail" in type(backend_obj).__module__:
        # Windows and macOS always have one. A bare Linux box often does not.
        print("ERROR: Python found no credential store on this machine.\n"
              "       On Windows this should not happen -- check that the\n"
              "       'keyring' package installed correctly.", file=sys.stderr)
        return 2
    print(f"Credential store : {backend}")
    print(f"Service          : {service}")
    print(f"Username         : {user}")
    print()

    if args.delete:
        try:
            keyring.delete_password(service, user)
            print("Deleted. This machine can no longer send mail until a "
                  "password is stored again.")
        except Exception as exc:                      # noqa: BLE001
            print(f"Nothing deleted: {exc}")
        return 0

    if args.check:
        try:
            stored = keyring.get_password(service, user)
        except Exception as exc:                      # noqa: BLE001
            print(f"ERROR: could not read the credential store: {exc}",
                  file=sys.stderr)
            return 2
        print("A password IS stored for this user on this machine."
              if stored else
              "NO password is stored for this user on this machine.")
        return 0 if stored else 1

    # --- store ----------------------------------------------------------
    secret = _local_password()
    if secret:
        print("Using the password from tools/smtp_local_config.py "
              "(gitignored).")
    elif args.password_file:
        real = os.path.abspath(args.password_file)
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if real.startswith(repo + os.sep):
            print(f"ERROR: {real}\n"
                  f"       is inside the repository. Put it somewhere git "
                  f"will never see.", file=sys.stderr)
            return 2
        try:
            with open(real, encoding="utf-8") as handle:
                secret = handle.read().strip()
        except OSError as exc:
            print(f"ERROR: could not read {real}: {exc}", file=sys.stderr)
            return 2
        if not secret:
            print(f"ERROR: {real} is empty.", file=sys.stderr)
            return 2
        print(f"Read the password from {real}")
    else:
        print("The password will not be shown as you type.")
        secret = getpass.getpass("SMTP password: ")
        if not secret:
            print("Nothing entered; no change made.", file=sys.stderr)
            return 1
        if secret != getpass.getpass("Confirm      : "):
            print("The two entries did not match; no change made.",
                  file=sys.stderr)
            return 1

    try:
        keyring.set_password(service, user, secret)
    except Exception as exc:                          # noqa: BLE001
        print(f"ERROR: could not write to the credential store: {exc}",
              file=sys.stderr)
        return 2

    # Read it back through the same path the app uses, rather than trusting
    # that set_password worked.
    email_service.reset_config_cache()
    if email_service._password_from_keyring(user) != secret:
        print("ERROR: stored, but reading it back gave something different.",
              file=sys.stderr)
        return 3
    print("\nStored, and read back correctly.")

    if args.test:
        print("\nTesting a real login...")
        ok, detail = _try_login(cfg, user, secret)
        print("  Login succeeded." if ok else f"  Login FAILED: {detail}")
        if not ok:
            return 4

    # The application finds the stored password by looking it up UNDER THIS
    # USERNAME, which it reads from a config file. Storing the password
    # without making the username discoverable leaves the app reporting
    # "not configured" with a perfectly good credential sitting in the vault.
    _ensure_local_config(email_service, cfg, user)

    print("\nDone. This machine can now send OTP emails.")
    return 0


def _ensure_local_config(email_service, cfg, user) -> None:
    """Make sure the app can discover the username (and sender address).

    Writes only NON-SECRET settings, and only when nothing already supplies
    them -- a shared S: config or SMTP_* environment variables win.
    """
    import json

    already = email_service._get_smtp_config()
    email_service.reset_config_cache()
    if already.get("user") == user and already.get("from_addr"):
        return                            # something already provides both

    from_addr = already.get("from_addr") or cfg.get("from_addr") or ""
    if not from_addr:
        print("\nNo sender address is configured. Mail needs a From: address.")
        try:
            from_addr = input(f"Sender address [{user}]: ").strip() or user
        except (EOFError, KeyboardInterrupt):
            print("\nSkipped; set from_addr in the config file yourself.",
                  file=sys.stderr)
            return

    path = os.path.join(os.path.expanduser("~"), ".solarjv", "email_config.json")
    settings = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                settings = json.load(handle)
        except (OSError, ValueError):
            settings = {}
    settings.pop("password", None)        # never, under any circumstances
    settings.update({
        "host": already.get("host") or cfg.get("host", ""),
        "port": int(already.get("port") or cfg.get("port", 587)),
        "user": user,
        "from_addr": from_addr,
        "from_name": already.get("from_name") or cfg.get("from_name", ""),
    })
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, indent=2)
        print(f"\nSaved the non-secret settings to {path}")
        print("  (username, host, port, sender -- never the password)")
    except OSError as exc:
        print(f"\nWARNING: could not write {path}: {exc}\n"
              f"         The app may not find the username.", file=sys.stderr)
    email_service.reset_config_cache()


def _diagnose() -> int:
    """Print the application's view of its own configuration.

    Never prints the password -- only whether one was found, and how long it
    is, which is enough to tell "missing" from "wrong".
    """
    import sys as _sys
    from solarjv_analyzer.auth import email_service

    print("=" * 62)
    print("WHAT THE APPLICATION SEES")
    print("=" * 62)
    print(f"python          : {_sys.executable}")
    print(f"version         : {_sys.version.split()[0]}")

    try:
        import keyring
        version = getattr(keyring, "__version__", "?")
        try:
            backend = type(keyring.get_keyring()).__name__
        except Exception as exc:                      # noqa: BLE001
            backend = f"unavailable: {exc}"
        print(f"keyring         : {version} ({backend})")
    except ImportError:
        print("keyring         : *** NOT INSTALLED FOR THIS PYTHON ***")
        print("                  The app cannot read the vault at all.")
        print("                  -> pip install keyring, in THIS interpreter.")
        return 1

    print()
    print("Config files, in the order the app tries them:")
    chosen = None
    for path in email_service._resolve_config_paths():
        if os.path.exists(path):
            marker = "USED  " if chosen is None else "masked"
            if chosen is None:
                chosen = path
            try:
                import json
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                keys = ", ".join(sorted(data))
                user_in = data.get("user", "(no user key)")
                print(f"  [{marker}] {path}")
                print(f"           user={user_in!r}  keys: {keys}")
                if "password" in data:
                    print("           *** contains a 'password' key -- "
                          "IGNORED by the app, remove it ***")
            except Exception as exc:                  # noqa: BLE001
                print(f"  [BROKEN] {path}: {exc}")
        else:
            print(f"  [absent] {path}")
    if chosen is None:
        print("  -> no config file at all; only SMTP_* env vars can supply "
              "the username.")
    else:
        print(f"  -> the app stops at the FIRST one it can read: {chosen}")
        print("     Anything below it is never consulted.")

    print()
    for key in ("HOST", "PORT", "USER", "PASSWORD", "FROM_ADDR"):
        value = os.environ.get(f"SMTP_{key}")
        if value is not None:
            shown = "(set)" if key == "PASSWORD" else repr(value)
            print(f"env SMTP_{key:<10}: {shown}  (overrides the file)")

    email_service.reset_config_cache()
    cfg = email_service._get_smtp_config()
    print()
    print(f"resolved user   : {cfg.get('user')!r}")
    print(f"resolved host   : {cfg.get('host')!r}:{cfg.get('port')}")
    print(f"resolved sender : {cfg.get('from_addr')!r}")

    stored = ""
    if cfg.get("user"):
        try:
            stored = keyring.get_password(
                email_service.KEYRING_SERVICE, cfg["user"]) or ""
        except Exception as exc:                      # noqa: BLE001
            print(f"vault read FAILED: {exc}")
    print(f"vault entry for that user: "
          f"{'FOUND (%d chars)' % len(stored) if stored else 'NOT FOUND'}")

    print()
    problem = email_service.configuration_problem()
    if problem:
        print("VERDICT: NOT configured")
        print("  " + problem.replace("\n", "\n  "))
        if cfg.get("user") and not stored:
            print()
            print("  A username resolved but the vault has nothing under it.")
            print("  Either the password was stored under a DIFFERENT "
                  "username,")
            print("  or under a different Windows account, or a config file")
            print("  higher in the list above is supplying the wrong user.")
            print(f"  Stored entries are keyed on: "
                  f"{email_service.KEYRING_SERVICE!r} + the username.")
        return 1

    print("VERDICT: configured. The app can send mail.")
    return 0


def _try_login(cfg, user, secret):
    """Connect and authenticate, nothing more. No mail is sent."""
    import smtplib
    try:
        server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=20)
        try:
            server.ehlo()
            if cfg.get("use_tls"):
                server.starttls()
                server.ehlo()
            server.login(user, secret)
        finally:
            try:
                server.quit()
            except Exception:                         # noqa: BLE001
                pass
        return True, ""
    except Exception as exc:                          # noqa: BLE001
        return False, str(exc)       # the server's message, never the secret


if __name__ == "__main__":
    sys.exit(main())
