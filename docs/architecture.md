# Architecture

Read this before making a structural change, adding a subsystem, or working in
a part of the app you haven't touched before. For day-to-day rules see
[`CLAUDE.md`](../CLAUDE.md).

## Identity

| Property | Value |
|----------|-------|
| Domain | Automated J‑V sweep characterization & SPO (Set‑Point Operation) degradation testing of solar cells |
| Hardware | Keithley 2400 SourceMeter + 6-channel multiplexer (hex protocol over serial) |
| UI | PyQt5 + pyqtgraph |
| Instrument control | PyVISA (Keithley), PySerial (MUX) |
| Measurement engine | PyMeasure (Procedure/Manager/Worker) |
| Auth | SQLite + Argon2id + email OTP (SMTP) |
| Packaging | Briefcase (.dmg / .msi), Hatchling build backend |
| Python | ≥3.8 to develop; exactly 3.9 for Briefcase (`[tool.briefcase]`) |

## Application flow

```
main.py  →  LoginDialog  →  CalibrationWindow  →  JVAnalyzerWindow
                │                    │                      │
                │      mandatory pre-measurement gate:       │
                │      reference-cell Isc within tolerance   │
                │                                            │
                └──── SessionManager tracks login state ─────┘
                      Logout from any window returns to login
```

`main.py` creates **one** `InstrumentManager` and passes it to every window.

## Key patterns

**One `InstrumentManager` per process.** Never copy instrument objects between
managers: two managers holding one Keithley means two owners of one VISA
session — whichever disconnects first closes it and the survivor can't tell,
because the attribute is still set. Readiness is `is_keithley_alive()` /
`is_mux_alive()` (they probe the session/port), never `keithley is not None`.
`connect_keithley()`/`connect_mux()` drop a stale handle and reconnect;
`disconnect_keithley()` on an already-closed session just releases the
reference. `get_keithley()` closes the adapter if init fails, so a retry
doesn't hit a busy COM port.

**The store sidecar is additive by construction** (`store/`): every finished
report is published to `S:\Data\JV\<windows user>\<yyyy-mm-dd>\<mode>\`, and
none of the producers know about it. `store.attach(window)` calls
`DirectoryManager.set_base_root(<staging>)` — the one method every output path
resolves through — so all temps and reports land in a local staging area, and a
timer sweeps staging on a worker thread and copies finished files across. The
store refuses delete and rename but allows existing files to be rewritten, so
the publisher uses exclusive create (`"xb"`) and **never** `shutil.copyfile`;
see `docs/plans/s-drive-publishing.md` §4b for the measured share behaviour and
the publish algorithm. The sidecar also rewires the file panel's Browse button
from outside to mean "also save a copy here"; every such reach-in is guarded and
degrades to "feature off" rather than raising.

**Singleton `DirectoryManager`** (`utils/directory_manager.py`): process-wide,
owns every output path as `Base/Username/Date/{Calibration|Main|SPO}/`. All
paths flow through it — never hardcode. The SPO module temporarily switches its
mode to `"SPO"` and restores it in a `finally` (keep the `finally`: an
exception must not strand the singleton in SPO mode).

**SPO is optional and self-contained** (`spo/`): the whole package can be
deleted and the JV app must still compile and run. `jv_analyzer_window.py` and
`app_controller.py` guard the import with try/except (`SPO_AVAILABLE`); SPO
features become no-ops when absent. Keep it that way — SPO never leaks into the
JV core.

**PyMeasure `Manager` for JV sweeps, a custom `QThread` for SPO.** JV sweeps are
a queue of experiments, which is what `Manager`/`Worker` is for. An SPO run is
one long hold, not a queue, so it uses `SpoWorker(QThread)`.

**Hardware-controlled staircase sweeps.** `JVProcedure` configures the
Keithley's own sweep (trigger model + buffer) instead of stepping voltage in
software. NPLC is derived from the requested sweep rate:
`time_per_point = voltage_range / sweep_rate / num_points`,
`NPLC = time_per_point * line_frequency`.

**Frozen config singleton.** `config.py` defines `@dataclass(frozen=True)
_Config`; import the module-level constants (`MUX_PORT`, `GPIB_ADDRESS`, …), not
`CONFIG.<attr>`. (Caveat: fields written without type annotations aren't
dataclass fields at all, so the frozen-ness is partly illusory — see AUDIT.)

## Where things live

```text
src/solarjv_analyzer/
├── main.py              # login → calibration → main window loop; owns the InstrumentManager
├── config.py            # frozen dataclass: ports, paths, constants
├── auth/                # SQLite users, Argon2id, session logging, SMTP OTP
│   ├── database.py      # init_db, register_user, authenticate_user
│   ├── session.py       # SessionManager singleton + per-session log file
│   ├── login_dialog.py  # login/registration UI (deliberately NOT themed)
│   ├── email_service.py # OTP generation/verification, rate limits, HTML mail
│   └── otp_widget.py    # 6-digit OTP input + modal
├── windows/
│   └── calibration_window.py   # the mandatory gate + checklist dialog
├── gui/
│   ├── jv_analyzer_window.py   # main window: layout, JV/SPO mode toggle, signals
│   ├── app_controller.py       # experiment queue, file merging, SPO lifecycle
│   ├── theme/                  # THE design system (see docs/ui.md)
│   ├── style.py                # legacy shim over theme/ — never edit colors here
│   ├── calibration_style.py    # likewise a shim
│   └── widgets/                # parameter_tab, instrument_tab, analysis_settings_tab,
│                               # analysis_panel, file_panel, combined_tab,
│                               # toggle_switch, channel_pinout
├── procedures/jv_procedure.py  # Keithley staircase sweep as a PyMeasure Procedure
├── instruments/
│   ├── instrument_manager.py   # connection lifecycle + session liveness (live hardware only)
│   ├── mux_controller.py       # serial hex protocol: AA010{pixel}000000BB
│   └── simulated/              # DEPRECATED fallbacks — do not expand
├── analysis/analysis.py        # compute_jv_metrics — pure computation (see docs/metrics.md)
├── spo/                        # optional module
│   ├── spo_procedure.py        # inherits JVProcedure; SpoWorker(QThread)
│   ├── spo_analysis.py         # compute_spo_metrics
│   ├── spo_report.py           # crash-safe CSV (fsync per row) + final report
│   └── spo_widget.py           # SPO tab + live power-vs-time plot
└── utils/
    ├── directory_manager.py    # the path singleton
    └── database.py             # LEGACY plaintext auth — do NOT use or extend
```

| Task | File |
|------|------|
| Sweep parameters | `gui/widgets/parameter_tab.py` |
| Instrument settings / NPLC preview | `gui/widgets/instrument_tab.py` |
| Metrics display | `gui/widgets/analysis_panel.py` |
| Experiment queue logic | `gui/app_controller.py` |
| Hardware connection | `instruments/instrument_manager.py` |
| J‑V sweep procedure | `procedures/jv_procedure.py` |
| SPO hold / UI | `spo/spo_procedure.py`, `spo/spo_widget.py` |
| SPO report format | `spo/spo_report.py` |
| Output paths | `utils/directory_manager.py` |
| Metric math | `analysis/analysis.py` |
| Colors / fonts / QSS | `gui/theme/tokens.py`, `theme/fonts.py`, `theme/stylesheet.py` |
| Auth / OTP | `auth/database.py`, `auth/email_service.py` |

## Subsystem notes

### Calibration gate
Mandatory before the main window. Measures the reference cell and compares Isc
against target within tolerance; report written to `.../Calibration/` with a
`PASS`/`FAIL` field. Can be skipped (with a warning) for emergency data
recovery.

- **Keithley-only**: the reference cell is wired straight to the SourceMeter.
  No MUX connection, no channel selection (`mux=None`, nominal channel 1). Do
  not re-add MUX gating to calibration readiness — the main window connects the
  MUX when a measurement needs it.
- Hardware connect is retried on checklist confirmation and via the "Hardware
  Disconnected — Retry" run-button state. Keep both paths: the first connect
  happens before the user has powered the hardware on.
- `CalibrationChecklistDialog` must be confirmed before hardware is enabled.
- Compliance-clamped points (|I| ≥ 99 % of compliance) are the instrument's
  ceiling, not the device curve. They are **suppressed entirely** — never
  plotted, never written, excluded from metrics
  (`JVProcedure.filter_compliance_points`, applied per branch *after* the
  fwd/rev index split; the full internal arrays keep them for sweep
  validation). Only the count survives: `compliance_clipped_points`, the user
  warning, and a "Compliance-Clipped Points (excluded)" report row.
- `closeEvent` is both the ✕ path and the hand-off path — see the trap in
  [`docs/testing.md`](testing.md).

### SPO module
- `SpoProcedure` inherits `JVProcedure` — shares `_write()`, `_query()`,
  `_safety_abort()`, MUX handling.
- Quick JV (to find Vmpp) runs **in memory only** via `SpoWorker`, not the
  PyMeasure Manager.
- The raw CSV is flushed with `fsync` after every row — crash-safe by design.
- Live metrics are recomputed every **5 samples**, not every sample.
- The inter-sample sleep is **abort-aware** (0.1 s chunks checking
  `should_stop()`). One `time.sleep(interval)` would make abort/`:OUTP OFF`
  wait up to a full sampling interval.
- A garbled sample read is retried once before failing the hold.
- **Teardown**: always `worker.abort()` then `worker.wait(timeout)` *before*
  disconnecting instruments (closeEvent, logout, abort_combined do this).
- **Combined JV+SPO**: SPO-phase parameters are snapshotted at run start
  (`_combined_spo_params`) — never re-read live widgets mid-run.
  `abort_combined` must NOT call `_reset_combined_state()` synchronously;
  cleanup belongs to `on_abort_returned` (JV phase) and
  `_on_combined_spo_finished` (SPO phase), which need `_combined_mode` still set.
- SPO uses its own block tags, distinct from the JV ones.

### SPO writes a journal and a report — only the report survives
`SpoReport` opens the raw CSV at run start and flushes + `fsync`s every sample,
so a crash mid-hold never loses what was already measured. `finalize()` then
writes the formatted report (parameters + metrics + the full series in mW) and
**deletes the raw journal** — but only after re-reading the report and counting
its rows against `self._rows`. A report is written with a single
`open(..., "w")`; a failure part way through it (disk full, or the cp1252 fault
in audit A4) leaves a plausible-looking short file, and deleting the journal on
the strength of that would be unrecoverable.

This is the same shape as the JV path, which removes its per-channel `_temp`
CSVs once the merged report is written. One run, one file.

What is left behind is exactly right. The journal only persists when
`finalize()` never ran — the application died mid-hold — which is the one case
where it is the sole record. The store sweeper publishes files that have been
quiet for `STORE_QUIET_SECONDS` (15 s) and the journal's mtime updates on every
sample, so it can never be published mid-run; a journal orphaned by a crash
goes quiet and does reach the S drive. An undeletable journal (write-once
share, or a handle still open on Windows) is a warning, not a failed run.

There is **no "Save Report" button.** The report is on disk and published
before the SPO card is updated; a button that copied the finished file
somewhere else implied it was not saved unless pressed, and was the only place
a measurement file was written with a plain file copy — which bypasses the
publisher's exclusive-create guarantee if aimed at the store. Tests:
`tests/spo/test_spo_raw_discard.py`.

### Output routing — each mode files into its own folder
`DirectoryManager.mode` decides the folder (`Calibration` / `Main` / `SPO`) and
every producer resolves through it. Two paths are easy to get wrong:

- `file_params['directory']` comes from the file panel, which always shows the
  **Main** folder. Anything derived from it lands in Main — correct for JV
  reports, wrong for everything else.
- A **combined JV+SPO** run passes an explicit `csv_path` to `SpoProcedure`,
  which skips the mode resolution the procedure does for itself. So
  `start_combined_run` resolves the SPO folder explicitly, with the singleton's
  mode restored in a `finally` — without it, an exception strands the
  process-wide `DirectoryManager` in `SPO` mode and every later JV run misfiles.

Regression test: `tests/gui/test_output_routing.py`.

### App controller
- **`analysis_shown` flag**: browser items are tagged after their metrics are
  pushed, to prevent duplicate analysis-panel updates. Cleared on panel resets
  (e.g. single → dual sweep transition).
- **Channel colors**: `AppController.CHANNEL_COLORS`, 6 fixed colors for
  channels 1–6. Forward = solid, reverse = dashed semi-transparent, same color.
- **File merge**: single-file mode writes temp per-channel CSVs during the run,
  then merges into one self-documenting report with block tags. Multi-file mode
  merges forward+reverse per channel.
- **`check_errors_between_points`** is always `False` in production — latency
  without benefit for solar cells.

### Run lifecycle — Queue / Abort / Resume
One rule: **an aborted sweep is discarded, never re-run.** Resume means "carry
on with what is still queued", which is what `ExperimentQueue.next()` already
does (it only returns experiments still marked `QUEUED`). Aborting the only
sweep therefore has nothing to resume, and the window simply returns to idle so
the operator can queue a fresh run — the same outcome as a longer queue, not a
special case.

Three things every path out of a run must do, because getting any of them wrong
is silent:

1. **Re-arm the manager** (`_rearm_manager()`). `Manager.abort()` clears
   pymeasure's `_start_on_add` and `_is_continuous`, and `Manager.resume()` is
   the only thing in pymeasure that restores them. Until this was called from
   every terminal state, one abort left `Manager.queue()` appending rows and
   returning without calling `next()` — the browser filled with QUEUED sweeps
   and nothing ran, for the rest of the session, from the JV tab *and* the
   combined tab.
2. **Finalize the run** (`_finalize_run()`): merge, write the report, release
   the instruments. This is reachable from `on_finished()` and from
   `on_abort_returned()`. Aborting the last sweep of a queue used to skip it
   entirely, so sweeps that had already completed were left as `_temp` files —
   which the store deliberately never publishes, and which the next
   `queue_experiment()` forgets.
3. **Discard the aborted sweep's partial file** (`_discard_aborted_sweep()`,
   driven by the experiment that `abort_returned` carries). A truncated sweep
   parses like any other but has no `[[ANALYSIS]]` block, so it used to be
   merged into the report as that channel's measurement, without metrics and
   without any marker. The browser row stays, marked Aborted, so the operator
   can still see what was stopped.

The button's label and its `clicked` connection are set together in
`_set_run_control()`. They were previously assigned in five places and drifted:
after an abort that emptied the queue the button read "Abort" while clicking it
ran `resume_experiment`. `abort_experiment()` also refuses to call
`Manager.abort()` when nothing is running — that raises, and an unhandled
exception in a Qt slot is not survivable in general (audit A8).

### Which UI fields reach the measurement
Most do. The four that need stating:

- **GPIB Address** (Instrument tab) is a manual backup. Typing a VISA resource
  and pressing Enter calls `port_resolver.set_session_keithley_resource()`,
  which beats both detection and `config.py`, then reconnects. It is
  **session-only** — never written to `~/.solarjv/config.json`, because an
  address typed to get through one afternoon must not silently become the
  permanent answer on every future launch. Refused while a run is in progress.
- **Line Frequency** is fixed at **50 Hz** (European mains) and is the single
  source of truth for the NPLC timing model on every path. A `:SYST:LFR?` query
  used to sit in `startup()`'s fallback-connect branch, which only runs when no
  instrument was handed in — so it normally never fired, and on the rare path
  where it did the sweep used different timing. On 60 Hz mains this constant
  must change: NPLC is `time_per_point * line_frequency`, so a wrong value
  makes the sweep run at a rate other than the one recorded in the report.
- **4-Probe Spacing** is **provenance only**. It reaches the report's
  `[[ EXPERIMENTAL PARAMETERS ]]` block (pymeasure writes every declared
  Parameter into the file header) but no metric: Rsq is derived from the J-V
  curve as `rho = Rsh*A/t`, `Rsq = rho/t * lateral_factor`. There is no
  4-point-probe measurement to put a spacing into. The tooltip says so.
- **"Enable detailed validation (debug only)"** (Analysis tab) does two things:
  raises the root logger to DEBUG, and passes `enable_validation` to the
  procedure for a per-sweep voltage-error line. The Log page's filter starts at
  DEBUG so that trace is actually visible; nothing below INFO is emitted unless
  the switch is on.

Tests: `tests/gui/test_parameter_wiring.py`.

### Which port is each instrument on
`config.py` ships **inside the packaged app**, so after a Briefcase build there
is no file a lab user can edit — and COM numbers change on their own. So
`instruments/port_resolver.py` decides both ports at startup (`main.py` calls
`resolve_at_startup()` before any window exists) and remembers what it found in
`~/.solarjv/config.json` under a `hardware` key.

The two instruments are separable because they use **different adapter chips**:
CH340 (VID `0x1A86`) for the MUX, Prolific PL2303 (VID `0x067B`) for the
Keithley. Signatures match on vendor ID **or** a description keyword, because
the PID varies across chip revisions while those do not.

Both instruments are identified **from the USB descriptor**; `*IDN?` is asked
only of PL2303 candidates, and only ever at startup:

- A **lone PL2303 adapter** is the Keithley by elimination. It is still asked
  `*IDN?`, because the rack is powered on before the app is launched and the
  reply belongs in the log — but **the reply verifies, it does not gate**. A
  silent adapter is used anyway, with a warning that the instrument may be off
  or set to a different baud rate. Refusing to resolve here would be worse than
  a wrong guess there is no evidence for.
- **Two or more PL2303 adapters** are told apart by `*IDN?`, looking for
  `KEITHLEY`/`2400`.
- The **MUX is always inferred**. Its protocol writes a frame and never reads a
  reply (audit S7), so there is nothing to ask it. Two CH340s and nothing
  remembered means resolution **refuses** rather than guessing: driving the
  wrong board would switch relays on unknown hardware.
- **Never probe the MUX port, or any port already open**, with `*IDN?` — the
  MUX expects binary hex frames and could read the query as one, and opening a
  busy port fails in a way that reads as "not the Keithley" and could move a
  live connection elsewhere.
- `refresh_if_unconfirmed()` **never probes**: it runs inside `connect_*()`, on
  the GUI thread, and enumeration alone is a registry read. A later refresh can
  only *upgrade* a resolution, never replace a known-good port with nothing.

`connect_keithley()`/`connect_mux()` call `refresh_if_unconfirmed()` first, so
an adapter that was not plugged in at startup is picked up on the next connect
attempt rather than being stuck with a stale config value. Once both are
confirmed that call does nothing.

**Both instruments are connected as the app opens.** `main.py` starts a daemon
thread (`startup-connect`) right after the `InstrumentManager` is built, which
calls `connect_keithley()` and `connect_mux()`; it is joined before the
calibration window is constructed, so nothing races the manager, and the login
dialog appears instantly meanwhile. Neither failure is fatal — the "Hardware
Disconnected — Retry" path and the connect at run start both still apply. The
checklist is a reminder for the operator, not a gate the software waits on: the
rack is already powered on when the app is launched, which is why the status
lights are expected to be green before anything is clicked.

`tools/port_probe.py` shows every port and what resolution would decide, with
`--probe` to include the `*IDN?` confirmation.

### Hardware status lights
The two dots in the main window answer one question — is each instrument
connected and usable *right now* — and they must be trustworthy **before** a
run, not after. `JVAnalyzerWindow.start_hardware_monitor()` polls every few
seconds on a worker thread (`instruments/port_status.py`), opening the MUX if
its adapter is present but not yet open, and repaints the lights.

**The dot is the whole interface.** Green means connected, red means not, and
that is all an operator needs before starting an experiment. Ports, adapter
names and the reason an open failed go to the **session log**, not the screen,
and only when the state actually changes — at one check every few seconds,
logging every tick would bury the run's own messages. The lights are indicators,
never controls: nothing about them should invite a click.

Two things are easy to get wrong here:

- **`is_open` and a VISA session handle both lie.** They stay valid after the
  USB adapter is pulled; only enumeration (`serial.tools.list_ports`) knows. So
  "connected" means *we hold an open handle AND the port is still enumerated*.
- **The monitor must be inert during a measurement.** While
  `is_busy`/`spo_running`/`_combined_mode` is set it only looks — never opens,
  closes or probes an instrument. `_measurement_running()` errs towards "busy".

Both instruments arrive as USB-to-serial adapters: the MUX through a CH340, the
Keithley through a Prolific PL2303 into its native RS-232 port, which is why
`GPIB_ADDRESS` is an `ASRL<n>::INSTR` resource rather than GPIB. Regression
test: `tests/gui/test_mux_status_light.py`.

### Auth & OTP
- Primary auth is `auth/database.py`: Argon2id password hashing via
  `argon2-cffi`. `utils/database.py` is a **legacy plaintext** module — do not
  use or extend it (and it still ships; see AUDIT C6).
- Registration includes email OTP verification (`auth/email_service.py`,
  `auth/otp_widget.py`).
- **SMTP credentials resolve through a cascade**:
  `S:\solarjv_email_config.json` (Windows lab deployment) → environment
  variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`,
  `SMTP_FROM_ADDR`, `SMTP_FROM_NAME`) → `~/.solarjv/email_config.json`.
- OTPs are held **in memory only** (never persisted) with rate limits: max 3
  requests per 10-minute window, 60 s resend cooldown, 3 failed attempts, 10-
  minute expiry.

### Session management
- Each login writes a timestamped log to `~/.local/share/SolarJV/logs/`
  (macOS/Linux) or `%LOCALAPPDATA%/SolarJV/logs/` (Windows).
- `SessionManager` is a class-level singleton; `get_current_user()` checks login
  state. `end_session()` removes the root-logger file handler **and clears
  `current_user`** — `main.py` reads that as "log in again", so only call it on
  a real logout or exit.
- Logout: instrument disconnect → session end → back to the login loop.

### CSV block tags — do not modify
Parsed by `AppController.load_files()` and `SpoReport.finalize()`:
`[[ EXPERIMENTAL PARAMETERS ]]`, `[[ ANALYSIS SUMMARY ]]`,
`[[ MEASUREMENT DATA ]]`, `[[ SPO METRICS ]]`, `[[ TIME SERIES DATA ]]`.
