"""Protected-store publishing (the S drive).

Every finished report is copied to
`<STORE_ROOT>\\<windows user>\\<yyyy-mm-dd>\\<mode>\\`, and to a second folder
as well if the operator has chosen one. The store refuses deletes, so the
application does all of its work in a local staging area and only finished
files are copied across.

Nothing in the measurement, analysis, instrument or report-writing path is
involved: `attach(window)` redirects `DirectoryManager`'s base root to staging
and runs a sweeper on a timer. See `docs/plans/s-drive-publishing.md`.
"""

from .identity import windows_user
from .paths import destination_dir, staging_root, store_date, store_root
from .publisher import PublishError, publish_file, sha256_of
from .sidecar import attach, detach, is_enabled
from .sweeper import extra_target, find_finished, pending_count, set_extra_target, sweep

__all__ = [
    "attach",
    "detach",
    "is_enabled",
    "sweep",
    "find_finished",
    "pending_count",
    "extra_target",
    "set_extra_target",
    "publish_file",
    "sha256_of",
    "PublishError",
    "windows_user",
    "store_root",
    "staging_root",
    "store_date",
    "destination_dir",
]
