# Plan — publish every generated file to the protected S drive

Status: **implemented** (2026-09-09). `src/solarjv_analyzer/store/` plus
`tools/store_probe.py`; 46 tests, whole suite 248 green, `render_preview.py`
PASS. First real run on the lab PC found two bugs, both fixed — see §4c.
Awaiting a clean end-to-end run. Phase 3 (UI polish) is still open.

## 1. The requirement

Every data file the app generates must end up on the S drive, laid out by
Windows user and date:

```
S:\Data\JV\<windows user id>\<yyyy-mm-dd>\<Calibration|Main|SPO>\
```

The store is write-once: no delete, no modifying a file once written. Only
whitelisted users can log into the measurement PC, each with their own
`stXXXXXX` account, so the Windows identity is the attribution.

| Scenario | Result |
|---|---|
| **A — normal** | run → temps + final in local staging → final copied to S → staging copy deleted. Report exists **only** on S. |
| **B — user picked a folder** | same, but the final is copied to S **and** that folder. Two identical copies. |
| **C — S unreachable** | the run still completes. The final stays in staging with a receipt; a warning shows *N files awaiting publish*; the app retries at startup and on a timer until it lands. |

Decisions already taken: Windows identity from the process token (not
`%USERNAME%`, which the user can change); `yyyy-mm-dd`, pinned at run start so a
run crossing midnight does not split; published artifacts are JV merged reports,
SPO reports, SPO raw CSVs, calibration reports and session logs; plot PNGs are
**not** published (the user picks their location in a dialog and that is the only
copy).

## 2. The overriding constraint: do not touch core functionality

Nothing in the measurement, analysis, instrument or report-writing path may
change. That rules out the obvious implementation — editing `app_controller`,
`spo_procedure`, `spo_report` and `calibration_window` to redirect their writes
and call a publisher.

Reading the code for a choke point instead turns up something much better.

**Every output path in the app resolves through one method.**

```
JV          file_panel.directory_input  ←  _update_save_directory()
                                        ←  dir_manager.get_current_directory()
SPO         dir_manager.get_current_directory(create=True)
Calibration dir_manager.get_file_path()
                                              ↓ all three
                    _get_dated_dir() → _get_user_dir() → get_base_root()
```

And `DirectoryManager` already exposes a public setter for it:

```python
def set_base_root(self, base_root):
    self._base_root = base_root
    self._update_display_directory()
```

**`set_base_root()` has zero callers anywhere in the codebase.** So a single
external call — `dir_manager.set_base_root(STAGING_ROOT)` — redirects every
temp file and every finished report of all three producers into staging, with
**no edit to any existing file**. The app carries on merging and deleting its
temps exactly as it does today; it simply does it somewhere else.

That is the whole trick. Everything else is a new package.

### Files this feature touches

| File | Change |
|---|---|
| `src/solarjv_analyzer/config.py` | 3 new annotated constants (`STORE_ROOT`, `STAGING_ROOT`, `STORE_ENABLED`) |
| `src/solarjv_analyzer/main.py` | 2 lines: import the store sidecar, `store.attach(window)` |

### Files this feature does **not** touch

`gui/app_controller.py` · `procedures/jv_procedure.py` · `spo/spo_procedure.py` ·
`spo/spo_report.py` · `spo/spo_analysis.py` · `spo/spo_widget.py` ·
`windows/calibration_window.py` · `analysis/analysis.py` ·
`instruments/*` · `gui/jv_analyzer_window.py` · `gui/widgets/*` ·
`utils/directory_manager.py`

Everything else lives in a new `src/solarjv_analyzer/store/` package. With
`STORE_ENABLED = False` the feature is inert and the app behaves exactly as it
does today — which is also the rollback: flip one flag.

## 3. The sidecar

```
src/solarjv_analyzer/store/
├── identity.py     resolve the Windows account from the process token
├── paths.py        <root>/<user>/<date>/<mode>  — pure, no I/O
├── receipts.py     the small JSON file that says where a staged file belongs
├── publisher.py    copy → verify → delete staging; never overwrite
├── sweeper.py      find finished files in staging, publish, retry failures
└── sidecar.py      attach() — wires it all to a window from the outside
```

### `attach(window)`

Called once per window from `main.py`. It:

1. calls `window.dir_manager.set_base_root(STAGING_ROOT)` — the redirect;
2. starts a `QTimer` that runs the sweeper on a worker thread;
3. adds a small status label + *"N awaiting publish"* indicator to the window;
4. rewires the file panel's **Browse** button so it records a second publish
   target instead of redirecting the app's writes.

Everything is wrapped in `hasattr` guards and try/except, and logs and gives up
rather than raising: a widget rename must degrade to "the store feature is off",
never to a broken app. (This is the codebase's documented trap — controller code
binds to widget attribute *names*.)

Point 4 is the one place the sidecar reaches into an existing widget. The
alternative is a five-line edit to `file_panel.py`; doing it from outside keeps
the touched-file count at two, at the cost of being less discoverable to someone
reading `file_panel.py`. Worth a comment there if you prefer — I'd take the
external version and note it in `docs/architecture.md`.

### How a file is recognised as finished — no hooks needed

The sweeper scans the staging tree and publishes anything that is:

- **not** named `*_temp.csv` — those are the per-channel intermediates the app
  merges and deletes itself, and
- **unchanged for at least 15 seconds**, and
- non-empty.

That is all the signal required, and it falls out remarkably cleanly:

- A JV run's temps are excluded by name; only the merged report is published.
- An **aborted** JV run never produces a merged report, so nothing is published
   — the "aborted runs publish nothing" rule holds for free.
- An SPO raw CSV grows every sampling interval, so during a six-hour hold it is
  never "unchanged for 15 s" and is never published mid-run.
- A **crashed** SPO hold leaves its raw CSV stable in staging, so the next
  sweep publishes it. The data survives.

### Publishing one file

1. Resolve `<root>/<user>/<date>/<mode>` from the receipt (the date is the one
   recorded when the file was created, not today's).
2. **Never overwrite**, using exclusive create (`"xb"`), not
   `shutil.copyfile`. If the name is taken, compare hashes: identical means
   already published (nothing to do); a *different, complete* file gets `_2`,
   `_3`… with a loud log line, because on this store an existing name means a
   double publish or a clock problem; a *partial* file is the wreck of an
   interrupted attempt and gets repaired in place (see §4b).
3. Copy, flush, `fsync`.
4. **Verify**: re-read the destination, compare size and SHA-256.
5. Only when every target verifies, delete the staging copy.

Deleting last is deliberate — it is the inverse of audit finding **A2** (raw
data deleted before the report was written), which is the same bug in the other
direction.

Publishing runs on a worker thread. A dead mapped drive can block a bare
`os.path.isdir('S:\\')` for tens of seconds, and the project rule against
blocking hardware I/O on the GUI thread applies to SMB for exactly the same
reason. The probe measures this.

### Receipts and retries

A receipt is a small JSON file written beside each staged artifact naming the
user, the pinned date, the mode, the extra target if any, and the hash. The
sweeper publishes anything with an unsatisfied receipt and leaves everything
else alone. Receipts are what make scenario C work with no state in memory: kill
the app mid-publish and the next start picks up exactly where it stopped.

### Session logs

Session logs live outside staging, in `%LOCALAPPDATA%\SolarJV\logs\`. The
sweeper watches that folder too and publishes a log once it has been idle for a
few minutes (i.e. the session that owned it has ended). No change to
`auth/session.py`.

## 4. Honest deviations from the earlier decisions

- **An aborted SPO hold will be published.** `SpoReport.finalize()` is
  abort-safe by design and writes a complete, well-formed report for the shorter
  hold, and the sweeper cannot tell that from a normal one without touching SPO
  code. I'd accept it: it is real data, correctly reported. If you want it
  excluded, that needs a marker written by the SPO path — a core change.
- **A crashed SPO hold will also be published** (its raw CSV), which is
  strictly better than the "publish nothing unless complete" rule we discussed
  and which I flagged as its cost. Confirm you are happy with it.
- **The file panel will display the staging path** until the Phase 3 UI work,
  and its "Open Folder" button will open a folder that is empty after
  publishing. Cosmetic, and worth fixing, but it does not affect where data
  lands.

## 4b. What the real S drive actually does (measured 2026-09-09)

`tools/store_probe.py --write` against `S:\Data\JV` on the lab PC:

| Property | Result |
|---|---|
| Identity from the process token | `st000000`, agrees with `%USERNAME%`, folder-safe |
| `S:\Data\JV` exists | yes — IT has provisioned the parent; we create `<user>\<date>\<mode>` |
| Create the three mode folders | works |
| Write a file, read back, SHA-256 | matches |
| Name collision | `_2` — exclusive create refused the clobber |
| **Delete** | **refused (`PermissionError`)** — no-delete is genuinely enforced |
| **Modify an existing file** | **allowed** — write-once is NOT enforced |
| **Repair a partial file in place** | **allowed** — `r+b` + `truncate()` works |
| **Rename inside the store** | **refused (`PermissionError`)** — rename needs Delete on the source name |
| Throughput | 10.5 MB/s (vs 348 MB/s local); a 1 MB report ≈ 0.1 s; stat 2–30 ms |

All 12 checks passed. The single warning — existing files can be modified — is
an ACL matter for IT, not a blocker: the publisher never modifies a file it did
not just create incomplete.

Two consequences.

**Never-overwrite is load-bearing, not a nicety.** Since existing files can be
rewritten, the only thing standing between a name collision and destroyed data
is the publisher opening its destination with `"xb"` (exclusive create) and
falling back to `_2`. That must never be relaxed to `shutil.copyfile`, which
happily truncates an existing target. There is a test for exactly this.

**An interrupted publish would otherwise strand a corrupt file forever.** If the
copy dies halfway — network blip, PC sleeps — the store holds a truncated
`report.csv` that cannot be deleted, and the next sweep's `"xb"` would sidestep
it and write `report_2.csv`, leaving a corrupt file permanently beside the good
one with nothing to distinguish them. The publisher must therefore:

1. Write, then **verify by hash before considering the publish done** (already
   in the design), and
2. on finding an existing destination whose hash does **not** match what we are
   publishing, treat it as a wreck from an interrupted attempt and **repair it
   in place** — `open(target, "r+b")`, seek(0), rewrite, `truncate()` — rather
   than skipping to `_2`. Only a destination that already matches our hash is
   "already published, nothing to do"; only a *different, complete* file gets
   the `_2` treatment.

Both recovery primitives were then measured. **Repair in place works; rename
does not** (`PermissionError` — it needs Delete on the source name). So the
atomic write-to-a-temp-name-then-rename approach is unavailable, and the
publisher writes directly to the final name.

### The publish algorithm, settled

For each file, for each target:

```
if destination does not exist:
    open(dest, "xb"), copy, flush, fsync          # exclusive create
    verify by SHA-256; on mismatch -> repair path below
else:
    h = sha256(destination)
    if h == our hash:        already published — nothing to do, treat as success
    elif size < ours and our content starts with theirs:
                             wreck of an interrupted copy ->
                             open(dest, "r+b"), seek(0), rewrite, truncate(), verify
    else:                    a different, complete file ->
                             try the next name (_2, _3 ...), log loudly
```

Delete the staging copy only once every target has verified. Never call
`shutil.copyfile` on a store path — it truncates an existing target, and on
this share that destroys data that cannot be restored.

## 4c. First real run — what broke (2026-09-09)

The first run on the lab PC published **live pymeasure temp files** to the store
and deleted one mid-sweep. Two separate defects, both mine.

**1. The temp-file check assumed an extension.** `app_controller` builds temps
as `{base}_{stamp}_ch{n}_{direction}_temp{ext}`, where `ext` comes from
`os.path.splitext(the user's filename)`. A user who types `Test` — no
extension — gets `Test_..._ch1_forward_temp` with **no extension at all**, and
`endswith("_temp.csv")` missed it. Fixed with a regex anchored on `_temp`
before an optional extension, which is also immune to dots elsewhere in the
name. Tests use the exact failing filenames.

**2. Name matching was the only guard.** Because the check failed, the sweeper
happily published a file pymeasure was still writing, and then removed it —
which surfaced as `KeyError: 'Voltage (V)'` from pandas reading a file that had
just vanished, cascading through every plot update.

The second is the more important lesson: a *predicate* about filenames cannot
be the only thing protecting a file the application still owns. The sweeper now
**claims before it publishes** — it renames the staged file to `*.publishing`
first, and only proceeds if that succeeds. On Windows a file another process
holds open cannot be renamed (Python's `open()` does not grant
FILE_SHARE_DELETE), so a successful rename *proves* nobody else is using it.
Publishing from the claimed path means the sweeper can never read a
half-written file and can never delete one out from under the app, whatever a
name predicate says. A publish failure renames it back; a crash mid-claim
leaves a `*.publishing` file, which the next sweep retries immediately.

**Permanent residue.** Two temp files reached `S:\Data\JV\st000000\2026-09-09\Main\`
and cannot be deleted. They are junk in the store forever — the unavoidable
cost of a write-once store meeting a bug.

**Still open:** temps of a *failed* run are never published (correct) and never
deleted by the sweeper (also correct — they are the app's), so they accumulate
in staging. A stale-temp cleanup after N days would be reasonable follow-up
work; it is not a correctness problem.

## 5. Verifying it — the probe, first

`tools/store_probe.py` is written and tested. It imports nothing from the app
and needs no third-party packages, so it runs on a bare lab PC **before any of
the above is built**, and answers empirically what the design assumes:

```powershell
py -3.13 tools\store_probe.py                        # read-only checks
py -3.13 tools\store_probe.py --root D:\fakeS --write  # rehearse safely first
py -3.13 tools\store_probe.py --write                # the real S drive
```

It checks: the identity the process token reports (and warns if `%USERNAME%`
disagrees), that the name is folder-safe, the `yyyy-mm-dd` layout, whether the
root answers a stat call within 10 s, whether all three mode folders can be
created, that a written file reads back with a matching SHA-256, that a name
collision produces `_2` instead of clobbering, whether an existing file can be
appended to, throughput on a 5 MB copy, that the local staging file can be
deleted, and finally whether the store refuses deletes.

The last three are reported as warnings rather than failures, because they
describe the ACL rather than our code — but they are the answers that tell you
whether "write-once" is actually enforced or whether the publisher's
never-overwrite rule is the only protection.

Default is read-only. `--write` leaves a few small files under
`<user>\<date>\<mode>\_store_probe\`, permanently if deletes are refused, so
rehearse against `--root` somewhere harmless first.

## 6. Tests for the sidecar

All run against `SOLARJV_STORE_ROOT` pointed at `tmp_path` — no S drive, no
Windows, so they work on the Mac and in the existing suite.

- **paths**: identity + pinned date + mode; a hostile `%USERNAME%` does not
  change the resolved folder; the date survives a simulated midnight crossing.
- **publisher**: hashes match; a truncated copy keeps staging; **no `os.remove`
  or `os.replace` is ever called with a path under the store root**; a collision
  produces `_2` and leaves the original byte-identical.
- **sweeper**: `*_temp.csv` ignored; a file still being written (mtime fresh)
  ignored; a stable file published then removed from staging; an unreachable
  root leaves the receipt for the next run.
- **sidecar**: `attach()` on a window whose `file_panel` has been renamed logs
  and disables itself instead of raising; with `STORE_ENABLED = False`,
  `get_base_root()` is untouched.
- **integration**: drive a fake run — write temps and a merged report into
  staging exactly as `app_controller` does, sweep, and assert the file is on the
  fake store under the right path and gone from staging. Scenario B asserts two
  byte-identical copies.

## 7. Order of work

1. **`tools/store_probe.py`** — done. Run it on the lab PC; its output decides
   nothing changes below but tells you what the share actually permits.
2. **`store/` package**, pure and fully unit-tested, wired to nothing.
   `STORE_ENABLED = False`.
3. **`config.py` + `main.py`** — the two-line integration. Enable the flag and
   run a real JV sweep against a local fake root.
4. **Switch the root to `S:\Data\JV`** on the lab PC and do a real run.
5. **Phase 3, optional polish**: file panel shows the destination, Browse reads
   as *"Also save a copy to…"*, Open Folder opens the store, a store-status
   indicator beside the instrument lights.

Nothing in steps 1–4 changes a line of measurement, analysis or instrument code.
