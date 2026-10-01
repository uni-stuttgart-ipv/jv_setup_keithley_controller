r"""Find finished files and publish them.

There is no hook anywhere in the application. A file is considered finished
when three cheap conditions hold, and between them they cover every case:

  * it is not named `*_temp.csv` — those are the per-channel intermediates the
    app merges and deletes itself. An aborted JV run never reaches the merge,
    so temps are all it leaves and nothing is published;
  * it has not been modified for `STORE_QUIET_SECONDS` — an SPO raw CSV grows
    every sampling interval, so a running hold is never picked up mid-run,
    while a crashed one goes quiet and is rescued on the next sweep;
  * it is not empty.

Session logs are handled separately: they live outside staging, are still open
while the session runs, and are never deleted locally.
"""

import logging
import os
import time

from solarjv_analyzer import config
from . import paths
from .identity import active_user
from .journal import open_journal
from .publisher import PublishError, publish_file, sha256_of

logger = logging.getLogger(__name__)

NOISY_AFTER_ATTEMPTS = 3          # demote repeated identical failures to debug
LOG_QUIET_SECONDS = 300.0         # a session log must be idle this long
_INTERNAL_NAMES = ("_publish_journal.json", "_publish_journal.json.tmp")


def _claim(src: str) -> str:
    """Take exclusive ownership of a staged file by renaming it, or return "".

    This is the structural guard. On Windows a file another process holds open
    cannot be renamed or deleted — Python's `open()` does not grant
    FILE_SHARE_DELETE — so a successful rename *proves* nobody else is using
    it. Publishing from the renamed path then means the sweeper can never
    read a half-written file, and can never delete one out from under the
    application.

    That second failure mode is not hypothetical: a name-matching bug once let
    the sweeper publish live pymeasure temp files and remove one mid-sweep,
    which surfaced as `KeyError: 'Voltage (V)'` from pandas reading a file that
    had just vanished. Name matching alone is not a sufficient guard.

    On POSIX a rename always succeeds, so there the quiet-time check remains
    the protection; the sequencing is identical either way.
    """
    claimed = src + paths.CLAIM_SUFFIX
    try:
        os.replace(src, claimed)
        return claimed
    except OSError as exc:
        logger.debug(f"Still in use, leaving for the next sweep: {src} ({exc})")
        return ""


def _release(claimed: str) -> None:
    """Give a claimed file back its original name after a failed publish."""
    original = paths.original_name(claimed)
    try:
        os.replace(claimed, original)
    except OSError as exc:
        # It stays as *.publishing; find_finished() picks those up again.
        logger.debug(f"Could not restore {original}: {exc}")


def extra_target() -> str:
    """The folder chosen with "Also save a copy to…", or "" if none.

    Stored in the same per-user config file DirectoryManager already uses, but
    under its own key so it cannot be confused with the legacy
    `base_directory` preference.
    """
    try:
        import json
        cfg = os.path.join(os.path.expanduser("~"), ".solarjv", "config.json")
        with open(cfg, "r", encoding="utf-8") as handle:
            value = json.load(handle).get("store_extra_copy", "")
        return value if isinstance(value, str) else ""
    except (FileNotFoundError, ValueError, OSError):
        return ""


def set_extra_target(folder: str) -> None:
    """Remember (or clear, with "") the second copy destination."""
    import json
    directory = os.path.join(os.path.expanduser("~"), ".solarjv")
    os.makedirs(directory, exist_ok=True)
    cfg = os.path.join(directory, "config.json")
    data = {}
    try:
        with open(cfg, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            data = loaded
    except (FileNotFoundError, ValueError, OSError):
        pass
    data["store_extra_copy"] = folder or ""
    with open(cfg, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _is_settled(path: str, quiet_seconds: float) -> bool:
    try:
        stat = os.stat(path)
    except OSError:
        return False
    if stat.st_size == 0:
        return False
    return (time.time() - stat.st_mtime) >= quiet_seconds


def find_finished(staging: str = None, quiet_seconds: float = None) -> list:
    """Staged files that look finished, oldest first."""
    staging = staging or paths.staging_root()
    quiet_seconds = config.STORE_QUIET_SECONDS if quiet_seconds is None else quiet_seconds
    found = []
    if not os.path.isdir(staging):
        return found
    for directory, _subdirs, files in os.walk(staging):
        for name in files:
            if name in _INTERNAL_NAMES or paths.is_temp_name(name):
                continue
            full = os.path.join(directory, name)
            # A file left claimed by an interrupted sweep is already ours and
            # is retried immediately, regardless of how recently it was touched.
            if name.endswith(paths.CLAIM_SUFFIX) or _is_settled(full, quiet_seconds):
                found.append(full)
    found.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    return found


def _prune_empty_dirs(path: str, stop_at: str) -> None:
    """Remove now-empty staging folders, never climbing above `stop_at`."""
    directory = os.path.dirname(os.path.abspath(path))
    stop_at = os.path.abspath(stop_at)
    while directory.startswith(stop_at) and directory != stop_at:
        try:
            os.rmdir(directory)
        except OSError:
            return
        directory = os.path.dirname(directory)


def _active_log_path() -> str:
    """The session log the running app currently holds open, or ""."""
    try:
        from solarjv_analyzer.auth.session import SessionManager
        handler = SessionManager._file_handler
        return os.path.abspath(getattr(handler, "baseFilename", "")) if handler else ""
    except Exception:
        return ""


def _logs_dir() -> str:
    try:
        from solarjv_analyzer.auth.database import _get_app_data_dir
        return os.path.join(_get_app_data_dir(), "logs")
    except Exception:
        return ""


def sweep(staging: str = None, store: str = None, include_logs: bool = True) -> dict:
    """Publish everything that is ready. Safe to call from a worker thread.

    Returns a summary dict: published, pending, failed, errors.
    """
    staging = staging or paths.staging_root()
    store = store or paths.store_root()
    summary = {"published": 0, "pending": 0, "failed": 0, "errors": []}

    try:
        user = active_user()
    except RuntimeError:
        # Nobody is signed in yet. This is the normal state between process
        # start and the login dialog being answered, so it is a quiet HOLD,
        # not a failure: finished files and session logs stay in staging and
        # publish on a later tick once they can be attributed to someone.
        # Counting it as an error here would fill the log with one entry
        # every sweep interval before anyone had even logged in.
        summary["pending"] += 1
        logger.debug("Store sweep held: no user signed in yet.")
        return summary

    journal = open_journal(staging)
    second = extra_target()

    for candidate in find_finished(staging):
        # Claim it first: a rename that succeeds proves the application has
        # closed it. Never publish or delete a file we do not own.
        claimed = candidate if candidate.endswith(paths.CLAIM_SUFFIX) else _claim(candidate)
        if not claimed:
            summary["pending"] += 1
            continue

        final_name = os.path.basename(paths.original_name(claimed))
        date_str, mode = paths.classify_staged(claimed, staging)
        targets = [paths.destination_dir(user, date_str, mode, store)]
        if second:
            # "where they choose" — straight into the chosen folder, not a
            # mirrored user/date/mode tree.
            targets.append(second)

        try:
            for dest in targets:
                publish_file(claimed, dest, final_name)
        except (PublishError, OSError) as exc:
            attempts = journal.record_failure(candidate, str(exc))
            level = logger.warning if attempts <= NOISY_AFTER_ATTEMPTS else logger.debug
            level(f"Publish deferred ({attempts} attempts) for {final_name}: {exc}")
            _release(claimed)
            summary["pending"] += 1
            summary["failed"] += 1
            summary["errors"].append(f"{final_name}: {exc}")
            continue

        journal.clear_failure(candidate)
        try:
            os.remove(claimed)                  # local staging — delete allowed
            _prune_empty_dirs(claimed, staging)
        except OSError as exc:
            # Published safely; only the local copy lingers. Not a failure.
            logger.warning(f"Published but could not remove the staged copy: {exc}")
        summary["published"] += 1

    if include_logs:
        _sweep_logs(journal, user, store, summary)

    journal.save()
    return summary


def _sweep_logs(journal, user: str, store: str, summary: dict) -> None:
    """Publish session logs. The local copies are never deleted."""
    directory = _logs_dir()
    if not directory or not os.path.isdir(directory):
        return
    active = _active_log_path()
    for name in sorted(os.listdir(directory)):
        if not name.lower().endswith(".log"):
            continue
        full = os.path.abspath(os.path.join(directory, name))
        if full == active or journal.log_published(full):
            continue
        if not _is_settled(full, LOG_QUIET_SECONDS):
            continue
        dest = paths.destination_dir(user, paths.log_date(name), paths.LOGS_MODE, store)
        try:
            written = publish_file(full, dest)
        except (PublishError, OSError) as exc:
            logger.debug(f"Session log not published yet ({name}): {exc}")
            continue
        journal.mark_log_published(full, written)
        summary["published"] += 1


def pending_count(staging: str = None) -> int:
    """How many finished files are still waiting to reach the store."""
    return len(find_finished(staging))
