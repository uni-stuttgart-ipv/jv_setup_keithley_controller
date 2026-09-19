#!/usr/bin/env python3
"""
S-drive store probe — verifies the protected-store requirements empirically.

Self-contained: no imports from solarjv_analyzer, no third-party packages, so
it runs on a bare lab PC before a single line of the feature is written. Its
job is to answer, on the real share, the questions the design depends on:

  * who does the process token say we are, and is that a usable folder name?
  * can we create  <root>\\<user>\\<yyyy-mm-dd>\\<mode>  ?
  * can we write a file, read it back, and does it hash the same?
  * what happens when a name already exists — can we clobber it?
  * can we modify an existing file in place?
  * can we delete?  (the answer decides whether "write-once" is real)
  * how fast is a realistic report-sized copy?

Usage
-----
    py -3.13 tools/store_probe.py                    # read-only checks
    py -3.13 tools/store_probe.py --write            # also write/verify/delete
    py -3.13 tools/store_probe.py --root D:\\fakeS --write   # rehearse elsewhere
    py -3.13 tools/store_probe.py --write --keep     # leave the probe files
    py -3.13 tools/store_probe.py --write --big-mb 0 # skip the throughput blob

Default is READ-ONLY. `--write` creates a handful of small files under
<root>\\<user>\\<date>\\_store_probe\\ and tries to remove them afterwards; on a
no-delete share they will stay there permanently, which is expected and is
itself one of the results. Rehearse with --root somewhere harmless first.

Exit code 0 = every required check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

DEFAULT_ROOT = r"S:\Data\JV"
MODES = ("Calibration", "Main", "SPO")
PROBE_DIRNAME = "_store_probe"
REACH_TIMEOUT_S = 10

_results: list[tuple[str, str, str]] = []   # (level, title, detail)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def record(level: str, title: str, detail: str = "") -> None:
    _results.append((level, title, detail))
    tag = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INFO": "[INFO]", "WARN": "[WARN]"}[level]
    print(f"{tag} {title}" + (f"\n       {detail}" if detail else ""), flush=True)


def section(name: str) -> None:
    print(f"\n--- {name} " + "-" * max(0, 62 - len(name)), flush=True)


# ---------------------------------------------------------------------------
# 1. identity
# ---------------------------------------------------------------------------
def windows_identity() -> tuple[str, str]:
    """(account_name, how_it_was_obtained). Reads the process token on Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes
            size = wintypes.DWORD(0)
            ctypes.windll.advapi32.GetUserNameW(None, ctypes.byref(size))
            buf = ctypes.create_unicode_buffer(size.value)
            if ctypes.windll.advapi32.GetUserNameW(buf, ctypes.byref(size)):
                return buf.value, "process token (GetUserNameW)"
        except Exception as exc:                              # pragma: no cover
            record("WARN", "GetUserNameW failed", str(exc))
    import getpass
    return getpass.getuser(), "getpass.getuser() fallback"


def check_identity() -> str:
    section("identity")
    name, how = windows_identity()
    record("INFO", f"resolved user: {name!r}", f"via {how}")

    if sys.platform == "win32":
        env_user = os.environ.get("USERNAME")
        if env_user and env_user != name:
            record("WARN", "%USERNAME% disagrees with the process token",
                   f"token={name!r} env={env_user!r} — the token is authoritative")
        else:
            record("INFO", f"%USERNAME% = {env_user!r}")

    bare = name.split("\\")[-1].split("@")[0].strip()
    bad = set(bare) & set('<>:"/\\|?*') or not bare
    if bad:
        record("FAIL", "user name is not usable as a folder name", f"{bare!r}")
    else:
        record("PASS", f"folder-safe user name: {bare!r}")
    return bare


# ---------------------------------------------------------------------------
# 2. paths
# ---------------------------------------------------------------------------
def check_paths(root: str, user: str) -> dict:
    section("path layout")
    date = datetime.now().strftime("%Y-%m-%d")
    record("PASS" if len(date) == 10 else "FAIL", f"date folder: {date}",
           "format is yyyy-mm-dd")
    dirs = {m: os.path.join(root, user, date, m) for m in MODES}
    for mode, path in dirs.items():
        record("INFO", f"{mode:<12} -> {path}")
    return dirs


def reachable(root: str) -> bool:
    """os.path.isdir on a dead mapped drive can block for a long time."""
    section("reachability")
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(os.path.isdir, root)
        started = time.perf_counter()
        try:
            ok = fut.result(timeout=REACH_TIMEOUT_S)
        except FuturesTimeout:
            record("FAIL", f"{root} did not answer within {REACH_TIMEOUT_S}s",
                   "a mapped-but-dead drive behaves like this — the app must "
                   "probe on a worker thread, never the GUI thread")
            return False
    took = (time.perf_counter() - started) * 1000
    if ok:
        record("PASS", f"{root} exists", f"stat took {took:.0f} ms")
        return True
    anc = nearest_existing_ancestor(root)
    if anc:
        record("WARN", f"{root} does not exist",
               f"nearest existing ancestor is {anc!r} — the probe can create "
               "the rest of the tree from there")
    else:
        record("FAIL", f"nothing along {root} exists",
               "the drive letter itself is not present — check the mapping")
    return False


# ---------------------------------------------------------------------------
# 3. write / verify / collision / modify / delete
# ---------------------------------------------------------------------------
def nearest_existing_ancestor(path: str) -> str | None:
    """Deepest existing directory on the way to `path`, or None.

    On Windows this bottoms out at the drive root, so a missing S: mapping is
    distinguishable from a store tree that simply has not been created yet.
    """
    cur = os.path.abspath(path)
    seen = set()
    while cur and cur not in seen:
        if os.path.isdir(cur):
            return cur
        seen.add(cur)
        cur = os.path.dirname(cur)
    return None


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def publish_copy(src: str, dest_dir: str, name: str) -> str:
    """Copy `src` into `dest_dir` as `name`, never overwriting.

    This is the exact rule the real publisher will use: if the name is taken,
    add _2, _3 ... rather than clobbering, because on a write-once store an
    existing name means a double publish or a clock problem, not permission to
    replace someone's data.

    The destination is opened "xb" — exclusive create — so the no-clobber
    guarantee is enforced by the OS rather than by an os.path.exists() check
    beforehand, which is racy: two processes finishing a run in the same second
    could both see the name as free.

    fsync is called on the WRITE handle before closing. Calling it on a
    read-only handle after the copy raises OSError(EBADF) on Windows, which is
    exactly the bug this probe caught on its first real run.
    """
    os.makedirs(dest_dir, exist_ok=True)
    stem, ext = os.path.splitext(name)
    attempt = 1
    while True:
        candidate = name if attempt == 1 else f"{stem}_{attempt}{ext}"
        target = os.path.join(dest_dir, candidate)
        try:
            with open(src, "rb") as fsrc, open(target, "xb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 1 << 20)
                fdst.flush()
                try:
                    os.fsync(fdst.fileno())
                except OSError:
                    # Some network redirectors refuse fsync. flush() has
                    # already handed the bytes to the OS; not worth failing
                    # a publish over.
                    pass
            return target
        except FileExistsError:
            attempt += 1
            if attempt > 50:
                raise


def check_write(dirs: dict, keep: bool, big_mb: int) -> None:
    section("create / write / verify")
    probe_dirs = {}
    for mode, path in dirs.items():
        probe = os.path.join(path, PROBE_DIRNAME)
        try:
            os.makedirs(probe, exist_ok=True)
            record("PASS", f"created {mode} folder tree", probe)
            probe_dirs[mode] = probe
        except Exception as exc:
            record("FAIL", f"cannot create {mode} folder tree", f"{type(exc).__name__}: {exc}")
    if not probe_dirs:
        return

    probe = probe_dirs.get("Main") or next(iter(probe_dirs.values()))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # a local staging file, exactly as the real flow will have
    staging = os.path.join(
        os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp",
        f"store_probe_{stamp}.csv",
    )
    payload = ("[[ MEASUREMENT DATA ]]\n"
               "Channel,Voltage (V),Current (A)\n"
               + "".join(f"1,{v/100:.2f},-0.0{v%9}\n" for v in range(200))
               + "# non-ascii check: µ ° ä\n")
    with open(staging, "w", encoding="utf-8", newline="") as fh:
        fh.write(payload)
    src_hash = sha256(staging)
    record("INFO", "staged a local test report",
           f"{staging} ({os.path.getsize(staging)} bytes)")

    # --- publish + verify -------------------------------------------------
    name = f"probe_{stamp}.csv"
    try:
        first = publish_copy(staging, probe, name)
        record("PASS", "wrote a file into the store", first)
    except Exception as exc:
        record("FAIL", "cannot write into the store", f"{type(exc).__name__}: {exc}")
        return

    try:
        if sha256(first) == src_hash:
            record("PASS", "read back and hash matches", src_hash[:16] + "…")
        else:
            record("FAIL", "hash mismatch after copy", "the store altered the bytes")
    except Exception as exc:
        record("FAIL", "cannot read the file back", f"{type(exc).__name__}: {exc}")

    # --- collision --------------------------------------------------------
    try:
        second = publish_copy(staging, probe, name)
        if os.path.basename(second) != os.path.basename(first) and os.path.exists(first):
            record("PASS", "name collision handled without overwriting",
                   f"second copy went to {os.path.basename(second)}")
        else:
            record("FAIL", "collision rule did not hold", f"{first} vs {second}")
    except Exception as exc:
        record("FAIL", "collision probe raised", f"{type(exc).__name__}: {exc}")
        second = None

    # --- can an existing file be modified in place? -----------------------
    # "No delete" and "cannot be changed" are different permissions. If this
    # succeeds, the store is append/truncate-able and write-once is NOT
    # enforced by the filesystem — the publisher's never-overwrite rule is then
    # the only thing protecting existing data.
    # Use a throwaway file, never the one just verified — on a permissive
    # share this test alters the target, and the verified copy must stay
    # byte-identical to what was staged.
    try:
        victim = publish_copy(staging, probe, f"probe_{stamp}_modify.csv")
        with open(victim, "a", encoding="utf-8") as fh:
            fh.write("# probe append\n")
        record("WARN", "an existing file CAN be appended to",
               "the share does not enforce write-once, so the publisher's "
               "never-overwrite rule is the ONLY thing protecting existing "
               "data; ask IT to deny Write Data/Append Data on existing files")
    except Exception as exc:
        victim = None
        record("PASS", "an existing file cannot be modified",
               f"append refused: {type(exc).__name__}")

    # --- can a partial file be repaired in place? -------------------------
    # This is the recovery path that matters most. If a publish is interrupted,
    # the store holds a truncated file that cannot be deleted. Repairing it
    # requires either truncate-and-rewrite (needs write on an existing file) or
    # a rename (needs Delete on the source name, which this share denies). We
    # reuse the already-sacrificed victim file so no extra artifacts are left.
    if victim:
        try:
            with open(victim, "r+b") as fh:
                fh.seek(0)
                fh.write(b"# repaired\n")
                fh.truncate()
            if os.path.getsize(victim) == len(b"# repaired\n"):
                record("PASS", "a partial file CAN be repaired in place",
                       "truncate + rewrite works, so an interrupted publish is "
                       "recoverable without needing delete")
            else:
                record("WARN", "truncate did not take effect",
                       f"size is {os.path.getsize(victim)}")
        except Exception as exc:
            record("FAIL", "a partial file CANNOT be repaired in place",
                   f"truncate refused: {type(exc).__name__}: {exc} — an "
                   "interrupted copy would strand a corrupt file permanently; "
                   "the publisher must write under a temp name first")

        # rename/move on the share — the other possible recovery primitive.
        # NTFS needs DELETE on the source name to rename it, so on a
        # no-delete share this is expected to fail.
        renamed = victim + ".renamed"
        try:
            os.rename(victim, renamed)
            record("INFO", "files CAN be renamed inside the store",
                   "so publish-to-temp-name-then-rename is available "
                   "(the cleanest way to make a publish atomic)")
            victim = renamed
        except Exception as exc:
            record("INFO", "files cannot be renamed inside the store",
                   f"{type(exc).__name__} — rename needs Delete on the source "
                   "name; the publisher must write directly to the final name "
                   "and repair in place if interrupted")

    # --- throughput -------------------------------------------------------
    big_dest = None
    if big_mb <= 0:
        record("INFO", "throughput probe skipped (--big-mb 0)")
    else:
        big = staging + ".big"
        try:
            with open(big, "wb") as fh:
                fh.write(os.urandom(big_mb * 1024 * 1024))
            t0 = time.perf_counter()
            big_dest = publish_copy(big, probe, f"probe_{stamp}_{big_mb}MB.bin")
            dt = time.perf_counter() - t0
            record("INFO", f"{big_mb} MB copy took {dt:.2f}s",
                   f"{big_mb/max(dt, 1e-6):.1f} MB/s — publish must run off the GUI thread")
        except Exception as exc:
            record("WARN", "throughput probe failed", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                os.remove(big)
            except OSError:
                pass

    # --- staging delete (must work — staging is local) --------------------
    try:
        os.remove(staging)
        record("PASS", "staging copy can be deleted",
               "required: the publisher deletes staging only after verifying")
    except Exception as exc:
        record("FAIL", "cannot delete the local staging file", f"{type(exc).__name__}: {exc}")

    # --- store delete (informational: this is the protection being tested)
    if keep:
        record("INFO", "--keep given: leaving probe files in the store", probe)
        return

    section("delete protection")
    removed, refused = [], []
    for path in [p for p in (first, second, victim, big_dest) if p]:
        try:
            os.remove(path)
            removed.append(os.path.basename(path))
        except Exception as exc:
            refused.append(f"{os.path.basename(path)} ({type(exc).__name__})")
    if refused and not removed:
        record("PASS", "the store refused every delete",
               "no-delete is enforced. The probe files below stay permanently:\n       "
               + "\n       ".join(refused))
    elif removed and not refused:
        record("WARN", "the store allowed files to be deleted",
               "no-delete is NOT enforced yet: " + ", ".join(removed)
               + "\n       (fine for a rehearsal root; on the real S: ask IT to deny Delete"
                 " and Delete Subfolders and Files)")
    else:
        record("WARN", "delete was partially allowed",
               f"removed={removed} refused={refused}")
    try:
        os.rmdir(probe)
    except OSError:
        pass


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("SOLARJV_STORE_ROOT", DEFAULT_ROOT),
                    help=f"store root to probe (default: {DEFAULT_ROOT})")
    ap.add_argument("--write", action="store_true",
                    help="perform the write/verify/collision/delete probes")
    ap.add_argument("--keep", action="store_true",
                    help="with --write, leave the probe files in place")
    ap.add_argument("--big-mb", type=int, default=1, metavar="N",
                    help="size of the throughput test file in MB (default 1, "
                         "0 to skip). On a no-delete share this blob is "
                         "permanent, so keep it small.")
    args = ap.parse_args()

    print("=" * 70)
    print(f"SolarJV store probe   root={args.root}   mode={'WRITE' if args.write else 'READ-ONLY'}")
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    print("=" * 70)

    user = check_identity()
    dirs = check_paths(args.root, user)
    root_ok = reachable(args.root)

    if args.write:
        if root_ok or nearest_existing_ancestor(args.root):
            check_write(dirs, args.keep, args.big_mb)
        else:
            record("FAIL", "store root unreachable — skipping write probes",
                   "fix the drive mapping, or rehearse with --root")
    else:
        section("write probes skipped")
        record("INFO", "re-run with --write to test creating and writing files",
               "a few small files may remain permanently on a no-delete share")

    section("summary")
    fails = [t for lvl, t, _ in _results if lvl == "FAIL"]
    warns = [t for lvl, t, _ in _results if lvl == "WARN"]
    print(f"  {sum(1 for l, _, _ in _results if l == 'PASS')} passed, "
          f"{len(warns)} warnings, {len(fails)} failed")
    for t in warns:
        print(f"  WARN  {t}")
    for t in fails:
        print(f"  FAIL  {t}")
    print()
    if fails:
        print("RESULT: FAIL — the store cannot support the design as-is (see above).")
        return 1
    print("RESULT: PASS" + ("  (with warnings — read them, they are about the ACL)" if warns else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
