r"""Copying a finished file into the protected store.

Measured behaviour of the real share (S:\Data\JV, 2026-09-09) that this code
is written against:

  * delete is REFUSED — nothing here may ever call os.remove/os.replace on a
    store path, and a mistake cannot be cleaned up afterwards;
  * modifying an existing file is ALLOWED — so `shutil.copyfile` would happily
    truncate someone else's report. It is never used on a store path;
  * rename is REFUSED (it needs Delete on the source name) — so the atomic
    write-to-a-temp-name-then-rename trick is unavailable and we must write
    directly to the final name;
  * a partial file CAN be rewritten in place — which is the only way to clean
    up after an interrupted copy, since it cannot be deleted.

Hence the algorithm in `publish_file`.
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

CHUNK = 1 << 20
MAX_COLLISION_ATTEMPTS = 50


class PublishError(RuntimeError):
    """Raised when a file could not be placed in a target with integrity."""


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_into(src: str, handle) -> None:
    """Stream `src` into an already-open binary handle, then flush and fsync."""
    with open(src, "rb") as source:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        # Some network redirectors refuse fsync. flush() has already handed
        # the bytes to the OS; not worth failing a publish over.
        pass


def _is_partial_of(target: str, src: str) -> bool:
    """True if `target` looks like an interrupted copy of `src`.

    Shorter than the source, and byte-identical as far as it goes. Anything
    else is a different file and must not be touched.
    """
    try:
        target_size = os.path.getsize(target)
        src_size = os.path.getsize(src)
    except OSError:
        return False
    if target_size == 0:
        return True
    if target_size >= src_size:
        return False
    with open(target, "rb") as existing, open(src, "rb") as source:
        remaining = target_size
        while remaining > 0:
            want = min(CHUNK, remaining)
            if existing.read(want) != source.read(want):
                return False
            remaining -= want
    return True


def _candidate_names(filename: str):
    """`report.csv`, then `report_2.csv`, `report_3.csv`, …"""
    stem, ext = os.path.splitext(filename)
    yield filename
    for n in range(2, MAX_COLLISION_ATTEMPTS + 1):
        yield f"{stem}_{n}{ext}"


def publish_file(src: str, dest_dir: str, filename: str = None) -> str:
    """Place `src` in `dest_dir` with integrity. Returns the path written.

    Never overwrites a complete file and never deletes anything. Verifies the
    result by hash before returning, so a caller may delete its own staged
    copy on success.

    Raises:
        PublishError: on any failure. The staged copy must then be kept.
    """
    filename = filename or os.path.basename(src)
    our_hash = sha256_of(src)

    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        raise PublishError(f"cannot create {dest_dir}: {exc}") from exc

    last_error = None
    for candidate in _candidate_names(filename):
        target = os.path.join(dest_dir, candidate)
        try:
            # Exclusive create: the OS enforces no-clobber, so two runs
            # finishing in the same second cannot both win the same name.
            with open(target, "xb") as handle:
                _copy_into(src, handle)
        except FileExistsError:
            try:
                if sha256_of(target) == our_hash:
                    logger.info(f"Already published, skipping: {target}")
                    return target
                if _is_partial_of(target, src):
                    # The wreck of an interrupted copy. It cannot be deleted,
                    # so repair it rather than leaving it beside the good file
                    # with nothing to tell them apart.
                    logger.warning(f"Repairing an incomplete published file: {target}")
                    with open(target, "r+b") as handle:
                        handle.seek(0)
                        _copy_into(src, handle)
                        handle.truncate()
                else:
                    logger.warning(
                        f"Name already taken by a different file, trying the "
                        f"next: {target}"
                    )
                    continue
            except OSError as exc:
                last_error = exc
                logger.error(f"Cannot inspect or repair {target}: {exc}")
                continue
        except OSError as exc:
            last_error = exc
            logger.error(f"Cannot write {target}: {exc}")
            # A permission or space problem will not be fixed by another name.
            break

        try:
            written = sha256_of(target)
        except OSError as exc:
            raise PublishError(f"cannot read back {target}: {exc}") from exc
        if written != our_hash:
            raise PublishError(
                f"verification failed for {target}: expected {our_hash[:16]}, "
                f"got {written[:16]}"
            )
        logger.info(f"Published {os.path.basename(src)} -> {target}")
        return target

    raise PublishError(
        f"could not publish {src} into {dest_dir}"
        + (f": {last_error}" if last_error else " (all candidate names taken)")
    )
