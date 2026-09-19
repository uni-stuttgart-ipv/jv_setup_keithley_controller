"""Who is running this — resolved from the Windows process token.

The store folder name is the attribution for every file in it, so it must not
come from anything the user can change. `%USERNAME%` is an ordinary
environment variable and can be set to any string before launching the app;
`GetUserNameW` reads the access token the process is actually running under.
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
