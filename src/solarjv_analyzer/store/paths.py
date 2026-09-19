r"""Where things go. Pure path arithmetic — no I/O, no side effects.

Two trees are involved.

**Staging** is where the application does all of its work. `DirectoryManager`
builds `<base>/<app username>/<dd-mm-yyyy>/<Mode>/` and the sidecar points its
base at the staging root, so temp files, merges and finished reports all land
there with no change to any producer.

**The store** is the S drive, laid out `<root>/<windows user>/<yyyy-mm-dd>/<Mode>/`.

Note the two differences that have to be bridged: the staging tree is named
after the *application* login with a `dd-mm-yyyy` folder, the store after the
*Windows* login with `yyyy-mm-dd`. The date in the staging path is what the
app stamped when the run started, which is exactly the date the file belongs
under — so a run that crosses midnight cannot be split, for free.
"""

import os
import re
from datetime import datetime

from solarjv_analyzer import config

MODES = ("Calibration", "Main", "SPO")
LOGS_MODE = "Logs"           # session logs have no measurement mode of their own

STORE_DATE_FMT = "%Y-%m-%d"
STAGING_DATE_FMT = "%d-%m-%Y"     # what DirectoryManager writes

_STAGING_DATE_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
CLAIM_SUFFIX = ".publishing"

# `_temp` immediately before an optional extension. A regex rather than
# splitext() so a filename containing dots elsewhere cannot fool it.
_TEMP_RE = re.compile(r"_temp(\.[^.]*)?$", re.IGNORECASE)


def _env(name: str, fallback: str) -> str:
    value = os.environ.get(name)
    return value if value else fallback


def store_root() -> str:
    """Root of the protected store. SOLARJV_STORE_ROOT overrides config."""
    return _env("SOLARJV_STORE_ROOT", config.STORE_ROOT)


def staging_root() -> str:
    """Root of the local working area. SOLARJV_STAGING_ROOT overrides config."""
    return _env("SOLARJV_STAGING_ROOT", config.STAGING_ROOT)


def store_date(when: datetime = None) -> str:
    """Today's store-format date folder (yyyy-mm-dd)."""
    return (when or datetime.now()).strftime(STORE_DATE_FMT)


def destination_dir(user: str, date_str: str, mode: str, root: str = None) -> str:
    """`<root>/<user>/<yyyy-mm-dd>/<mode>`."""
    return os.path.join(root or store_root(), user, date_str, mode)


def is_temp_name(filename: str) -> bool:
    """True for the per-channel intermediates the app merges and deletes itself.

    `app_controller` builds these as `{base}_{stamp}_ch{n}_{direction}_temp{ext}`
    where `ext` comes from `os.path.splitext(user filename)` — so a user who
    types `Test` with no extension gets `Test_..._ch1_forward_temp` with **no
    extension at all**. Matching on the stem rather than on `_temp.csv` is
    therefore essential: the extension-based version of this check published
    live temp files to the store and deleted one while pymeasure still had it
    open, which surfaced as `KeyError: 'Voltage (V)'` from pandas.

    Excluding these by name is also what makes an aborted JV run publish
    nothing: an abort never reaches the merge step, so temps are all it leaves.
    """
    return bool(_TEMP_RE.search(original_name(filename)))


def original_name(filename: str) -> str:
    """Strip the in-progress claim suffix, if present."""
    if filename.endswith(CLAIM_SUFFIX):
        return filename[:-len(CLAIM_SUFFIX)]
    return filename


def classify_staged(path: str, root: str = None) -> tuple:
    """Work out where a staged file belongs in the store.

    Returns (date_str, mode). Falls back to the file's own modification date
    and `Main` when the staging path does not have the expected shape, so an
    unexpected layout still publishes to a sane place rather than being
    skipped.
    """
    root = root or staging_root()
    try:
        rel = os.path.relpath(path, root)
    except ValueError:                      # different drive
        rel = os.path.basename(path)
    parts = [p for p in rel.replace("\\", "/").split("/") if p][:-1]  # drop filename

    mode = next((p for p in reversed(parts) if p in MODES), "Main")

    date_str = ""
    for part in parts:
        if _STAGING_DATE_RE.match(part):
            try:
                date_str = datetime.strptime(part, STAGING_DATE_FMT).strftime(STORE_DATE_FMT)
            except ValueError:
                date_str = ""
            break

    if not date_str:
        try:
            date_str = datetime.fromtimestamp(os.path.getmtime(path)).strftime(STORE_DATE_FMT)
        except OSError:
            date_str = store_date()

    return date_str, mode


def log_date(filename: str) -> str:
    """Date folder for a session log named `session_YYYY-MM-DD_HH-MM-SS.log`."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else store_date()
