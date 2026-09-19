# CLAUDE.md — SolarJV Analyzer

PyQt5 app that drives a **Keithley 2400 + 6-channel MUX** for J‑V sweeps and SPO
degradation tests on solar cells. It commands real lab hardware: a wrong sign or
an unhandled exception can hold a cell at the wrong polarity or leave the source
energized.

This file is the always-loaded part. Everything else is on demand — **read the
one doc that covers what you're about to touch, not all of them.**

| Before you touch | Read |
|---|---|
| `analysis/`, `spo/` metric math | [`docs/metrics.md`](docs/metrics.md) + `physics-change` skill |
| `gui/`, `windows/`, `spo/spo_widget.py`, theme | [`docs/ui.md`](docs/ui.md) |
| `store/` (S-drive publishing) | [`docs/plans/s-drive-publishing.md`](docs/plans/s-drive-publishing.md) — §4b has the measured share behaviour |
| COM ports / instrument addresses | `instruments/port_resolver.py` — resolved at startup, **not** from `config.py`; spec in [`docs/plans/hardware-status-and-ports.md`](docs/plans/hardware-status-and-ports.md) |
| structure, a subsystem you don't know, "where does X live" | [`docs/architecture.md`](docs/architecture.md) |
| anything, if you hit a weird failure | [`docs/testing.md`](docs/testing.md), then grep [`AUDIT.md`](AUDIT.md) |

## Commands

```bash
pip install -e .                          # dev install
python -m solarjv_analyzer.main           # run (or: solarjv / hatch run start)
QT_QPA_PLATFORM=offscreen python3 -m pytest tests/ -q     # whole suite (~6-16 s)
QT_QPA_PLATFORM=offscreen python3 tools/render_preview.py # offscreen UI geometry checks
briefcase create/build/package macOS      # or windows — needs Python 3.9
```

## Verify what you changed

Match the check to the change; don't run the whole ceremony for a one-liner.
A PostToolUse hook already runs the matching subset after each edit and stays
silent when it passes.

| Changed | Run |
|---|---|
| `analysis/`, `spo/*analysis*` | `pytest tests/analysis tests/spo -q` — **and write the failing test first** ([`docs/metrics.md`](docs/metrics.md)) |
| `gui/`, `windows/`, `spo/spo_widget.py`, `theme/` | `pytest tests/gui -q` **and** `tools/render_preview.py` (must print PASS) |
| `procedures/`, `instruments/` | `pytest tests/procedures tests/instruments -q` |
| `store/` | `pytest tests/store -q` |
| `instruments/` port resolution | `pytest tests/instruments -q` |
| `auth/` | `pytest tests/auth -q` |
| `utils/`, `config.py` | `pytest tests/utils tests/config -q` |
| several areas, or you're handing the work over | the whole suite (+ `render_preview.py` if any UI changed) |
| comments, docstrings, markdown | nothing |

All pytest commands need `QT_QPA_PLATFORM=offscreen`. A change isn't done until
its row is green. Say plainly what you ran — and never imply a hardware path is
verified from a sandbox: threading, abort timing and instrument behaviour need a
real run on the rig.

## Hard rules

**Threads.** Never call PyVISA/PySerial on the GUI thread. JV sweeps go through
PyMeasure `Manager`/`Worker`; SPO uses `SpoWorker(QThread)`. Cross-thread data
moves by Qt signal. Always `worker.abort()` then `worker.wait(timeout)` *before*
disconnecting instruments.

**Abort.** On any hardware exception, `:OUTP OFF` then `:ABOR` —
`JVProcedure._safety_abort()` does it. Validate inputs *before* mutating state
like `is_busy`: an unhandled exception in a Qt slot kills the process (qFatal)
with the source possibly still on.

**One `InstrumentManager` per process.** `main.py` creates it and passes it to
every window. Never copy instrument objects between managers — two owners of one
VISA session means whichever disconnects first silently invalidates the other's
handle. Test readiness with `is_keithley_alive()` / `is_mux_alive()`, **never**
`keithley is not None`: a closed session leaves the attribute set.

**Live hardware only.** No simulated fallback in production paths;
`SIMULATION_MODE` is deprecated and `instruments/simulated/` must not grow.
`connect_*()` raise on failure; always `disconnect_*()` after use.

**Vmpp is signed** (negative for p-i-n). SPO holds at that value directly —
`abs()` reintroduces a wrong-polarity-hold bug. **SPO power is `P_gen = −(V·I)`**,
so degradation reads as negative drift. Voc/Isc are `NaN` when the sweep has no
zero crossing — never extrapolate one.

**Calibration is Keithley-only** (reference cell wired straight to the
SourceMeter, `mux=None`). Don't re-add MUX gating to its readiness check.

**Colors live in `gui/theme/tokens.py`** — never inline, never in the
`gui/style.py` shim. Interactive controls are teal; green is status-only
(connected, PASS); red is error/abort/logout.

**The protected store refuses delete and rename, but allows existing files to
be rewritten.** So in `store/`, never use `shutil.copyfile` on a store path —
it truncates an existing target and the data cannot be recovered. Exclusive
create (`"xb"`) is the only thing protecting published reports; there is a test
for it. Nothing may call `os.remove`/`os.replace` on a path under the store root.

**Don't modify the CSV block tags** — `AppController.load_files()` and
`SpoReport.finalize()` parse them: `[[ EXPERIMENTAL PARAMETERS ]]`,
`[[ ANALYSIS SUMMARY ]]`, `[[ MEASUREMENT DATA ]]`, `[[ SPO METRICS ]]`,
`[[ TIME SERIES DATA ]]`.

**`utils/database.py` is legacy plaintext auth** — don't use or extend it. Real
auth is `auth/database.py` (Argon2id + email OTP).

**Editing discipline.** Small, single-purpose diffs. Never mix a refactor with a
behavior change. Preserve widget attribute names and signal signatures —
controller code binds to them by name, so a rename fails silently at runtime.

## Keeping docs honest

Update the doc you invalidated, in the same change — nothing else:

- metric formula changed → README § "Solar Cell Metrics" + `docs/metrics.md`
- a contract or invariant changed → this file (keep it short) or the relevant `docs/` page
- an AUDIT finding fixed or newly found → move/add its row in `AUDIT.md` (one row, not a paragraph)

## Skills

`physics-change` (test-first metric math) · `ui-preview` (offscreen GUI
verification) · `release-check` (final gate before handing work over). They
encode lessons that cost real debugging time — follow them when their topic
comes up, and skip them when it doesn't.
