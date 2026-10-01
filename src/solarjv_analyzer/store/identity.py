"""Who is running this — the APPLICATION login, not the Windows account.

The store folder name is the attribution for every file in it. It used to come
from the Windows process token (`GetUserNameW`), which no one can spoof — but
the lab now runs every machine under ONE shared Windows account, so that name
is identical for everybody and would file all users' data into a single folder
with no attribution at all.

The identity is therefore the name the operator signed in with, set by
`set_app_user()` after a successful login and cleared on logout. Two
consequences follow, and both are deliberate:

* **It is not cached like a process identity.** `main.py` runs a re-login loop,
  so one process can serve several operators in turn. A cached value would
  publish the second person's data into the first person's folder — silently,
  onto a share that refuses deletions.
* **There is no fallback to the Windows name.** Falling back would quietly
  write to a *different* folder than the one the operator expects. Nothing
  publishes until somebody has logged in; `active_user()` raises instead.

`windows_user()` remains for diagnostics and as the identity of last resort for
callers that explicitly want the OS account.
"""

import getpass
import logging
import re
import sys

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_cached: str = ""


def _from_token() -> str:
    """Windows account name from the process token, or "" if unavailable."""
    if sys.platform != "win32":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        size = wintypes.DWORD(0)
        # First call fails on purpose and fills in the required buffer size.
        ctypes.windll.advapi32.GetUserNameW(None, ctypes.byref(size))
        buf = ctypes.create_unicode_buffer(size.value)
        if ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(size)):
            return buf.value or ""
    except Exception as exc:
        logger.warning(f"GetUserNameW failed, falling back to getpass: {exc}")
    return ""


def _sanitise(name: str) -> str:
    """Reduce an account name to something usable as a single folder name."""
    if not name:
        return ""
    # DOMAIN\user  and  user@domain  both reduce to `user`.
    name = name.split("\\")[-1].split("/")[-1].split("@")[0].strip()
    name = _SAFE.sub("_", name).strip("._-")
    return name


# The signed-in application user. Empty means "nobody is logged in", which is
# a refusal to publish, not an invitation to guess.
_app_user: str = ""


def set_app_user(name: str) -> str:
    """Record who just logged in. Pass "" (or None) on logout to clear it.

    The name is sanitised exactly as a Windows account name would be: a
    username is free text typed at registration, so without this it could
    contain path separators and escape the store tree.

    Returns:
        str: the sanitised name actually stored ("" when cleared).
    """
    global _app_user
    cleaned = _sanitise(name or "")
    if cleaned != _app_user:
        if cleaned:
            logger.info(f"Store identity: {cleaned!r} (application login)")
        else:
            logger.info("Store identity cleared (logged out)")
    _app_user = cleaned
    return _app_user


def app_user() -> str:
    """The signed-in application user, or "" if nobody is logged in."""
    return _app_user


def active_user() -> str:
    """The name every store path is built from.

    Raises:
        RuntimeError: when nobody is logged in. Callers treat this as "hold,
            do not publish yet" — it is how pre-login session logs are kept
            back until they can be attributed to someone.
    """
    if _app_user:
        return _app_user
    raise RuntimeError(
        "No user is signed in; refusing to publish to an unattributed folder."
    )


def windows_user(refresh: bool = False) -> str:
    """The account this process runs as, safe to use as a folder name.

    Cached: the identity of a running process cannot change, and this is
    called on every sweep.

    Raises:
        RuntimeError: if no usable name can be determined at all. Publishing
            to an unattributed folder would be worse than not publishing.
    """
    global _cached
    if _cached and not refresh:
        return _cached

    name = _sanitise(_from_token())
    source = "process token"
    if not name:
        try:
            name = _sanitise(getpass.getuser())
            source = "getpass.getuser()"
        except Exception:
            name = ""

    if not name:
        raise RuntimeError(
            "Cannot determine the current user; refusing to publish to an "
            "unattributed folder."
        )

    if _cached != name:
        logger.info(f"Store identity: {name!r} (via {source})")
    _cached = name
    return name
