r"""Write a file so that its final name appears only when it is complete.

The problem this solves: the sweeper has to decide whether a staged file is
finished. It used to guess, by waiting `STORE_QUIET_SECONDS` for the file to
stop changing. A guess is wrong in both directions — it delays every report by
the length of the window, and a writer that merely pauses for longer than the
window looks finished when it is not.

The rule here replaces the guess with a fact. A writer builds its file under
`<final><PART_SUFFIX>`, closes it, and only then renames it to `<final>`. The
sweeper skips anything carrying the suffix, so **a file under its final name is
complete because the writer closed it before renaming it**.

Two notes on why the close is what matters:

  * The close is the guarantee. On Windows a rename of a file that another
    handle still holds open normally fails, because Python's `open()` does not
    grant FILE_SHARE_DELETE — but that is a *backstop* that turns a
    mis-ordering into a loud failure, not the guarantee itself. A handle opened
    with FILE_SHARE_DELETE can be renamed while open, so a successful rename
    must never be read as proof that nobody holds the file.
  * The suffix is `.part`, not `.tmp`. `.tmp` sits in the default exclude lists
    of antivirus and disk-cleaning tools, and `FILE_ATTRIBUTE_TEMPORARY` is an
    unrelated Windows flag that asks the cache manager to avoid writing to
    disk. Neither is wanted for a file holding measurement data.

`PART_SUFFIX` lives in `paths` so this module and the sweeper import the same
constant and cannot drift apart.
"""

import contextlib
import logging
import os
import time

from . import paths

logger = logging.getLogger(__name__)

_RENAME_ATTEMPTS = 5
_RENAME_BACKOFF = 0.05        # seconds; 50, 100, 150, 200 ms between tries


@contextlib.contextmanager
def published_file(final_path, mode="w", **kwargs):
    r"""Open `final_path` for writing, via a `.part` sibling in the same folder.

    Use exactly like `open()`::

        with published_file(report_path, newline="") as handle:
            handle.write(...)

    On a clean exit the handle is closed, flushed to disk and renamed to
    `final_path`. On an exception nothing appears under the final name and the
    `.part` is removed, so a half-written report can never be mistaken for a
    finished one.

    Raises:
        FileExistsError: if `final_path` already exists. Report names carry a
            timestamp, so a collision means two runs claimed one name — a bug
            worth seeing rather than silently resolving by overwriting a file
            the sweeper may already be reading.
    """
    part = final_path + paths.PART_SUFFIX
    parent = os.path.dirname(final_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    started = time.monotonic()
    handle = open(part, mode, **kwargs)
    completed = False
    try:
        yield handle
        handle.flush()
        _fsync(handle)
        completed = True
    finally:
        # Close before renaming, always: the close is the completion
        # guarantee, and Windows refuses to rename a file it still holds open.
        handle.close()
        closed_at = time.monotonic()
        if completed:
            promote(part, final_path)
            # T0/T1: how long the write took, and how long it then took for
            # the report to carry its final name. Everything after this point
            # belongs to discovery and the copy to the store, timed there.
            logger.debug(
                "published_file: %s written in %.3f s, renamed in %.3f s",
                os.path.basename(final_path),
                closed_at - started,
                time.monotonic() - closed_at,
            )
        else:
            discard(part)


def promote(part_path, final_path):
    r"""Rename a finished `.part` to its final name.

    The `.part` is left untouched if this fails, so the data is still on disk
    and the next sweep's janitor — or a later retry — can deal with it.
    """
    # `os.rename` raises FileExistsError on Windows when the target exists but
    # overwrites silently on POSIX. Checking first makes a development Mac and
    # the lab PC behave identically, and makes a name collision loud on both.
    if os.path.exists(final_path):
        raise FileExistsError(
            f"refusing to overwrite an existing report: {final_path}")
    # Windows hands out transient failures here: antivirus and the search
    # indexer both open a file the moment it is closed, and a rename while
    # they hold it fails with PermissionError. It clears in milliseconds, and
    # not retrying would risk a finished report being left as a `.part` for
    # the janitor to delete.
    for attempt in range(_RENAME_ATTEMPTS):
        try:
            os.rename(part_path, final_path)
            return
        except PermissionError:
            if attempt == _RENAME_ATTEMPTS - 1:
                raise
            time.sleep(_RENAME_BACKOFF * (attempt + 1))


def discard(part_path):
    """Remove an abandoned `.part`. Never raises — the janitor is the backstop."""
    try:
        os.remove(part_path)
    except OSError as exc:
        logger.debug(f"Could not remove the abandoned part file {part_path}: {exc}")


def _fsync(handle):
    """Push the bytes at the OS, where the filesystem supports it.

    Costs about a millisecond at report sizes and means a power cut cannot
    leave a file that carries its final name but holds nothing.
    """
    try:
        os.fsync(handle.fileno())
    except (OSError, ValueError, AttributeError):
        # Not every filesystem or file-like object supports fsync; flush()
        # alone is still enough for another process to read the content.
        pass
