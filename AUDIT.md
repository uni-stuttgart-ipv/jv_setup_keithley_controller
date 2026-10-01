# Audit — open findings

Working list of known, unfixed defects. **Grep this file for the area you are
about to touch; don't read the archive unless a row here is the bug you're
chasing.** Full reasoning, verified numbers and impact analysis:
[`docs/audit-2026-08.md`](docs/audit-2026-08.md).

Source: full four-reviewer audit, 2026-08-07. Line numbers below are current as
of the last edit to each row; treat them as a starting point, not gospel.

## Open — hardware & safety

| ID | What breaks | Where | Fix direction |
|----|-------------|-------|---------------|
| C3 | The whole sweep is one blocking `:READ?` on the worker thread, so `should_stop()` is unreachable: **Abort only sets a flag — the Keithley stays energized until the sweep ends.** Meanwhile `resume_experiment` and the `closeEvent`s write `:OUTP OFF` from the GUI thread on the same non-thread-safe session. | `procedures/jv_procedure.py` (sweep read), `gui/app_controller.py` (resume) | `:INIT` + status polling (`*OPC`/MAV) with short timeouts, `should_stop()` between polls; one lock around all VISA access |
| C4-partial | Calibration `_on_proceed`/`_on_skip` abort the sweep but never **join** the worker, so a `:READ?` can still be in flight when the main window starts its own run. | `windows/calibration_window.py` | Bounded wait for `manager.is_running()` to clear before completing the hand-off |
| S5 | Off-grid sweep endpoint: the exact stop voltage is appended but the instrument only sources `start + n·step`, so the buffer count disagrees → the last point is recorded **at the start voltage**. | `procedures/jv_procedure.py` `_generate_voltage_sequence` | Snap the endpoint to the grid, or set the count from the instrument's own sweep length |
| S6 | `:SYST:ERR?` responses are only logged — a mis-configured instrument sweeps anyway and produces wrong data silently. | `procedures/jv_procedure.py` `_check_errors` | Raise on non-`0,` responses at configure time |
| S7 | MUX writes the frame, sleeps 0.5 s, **never reads an ACK**; no channel-range validation (ch 0 / >6 send undefined frames or crash). A missed relay attributes a curve to the wrong pixel, undetectably. | `instruments/mux_controller.py` | Read/verify the ACK; validate `1 <= channel <= CHANNEL_COUNT` |
| S8 | Compliance clipping is only detected by magnitude. STAT element not requested, `:SENS:CURR:PROT:TRIP?` never queried — with a fixed measure range the real ceiling is the range, not the configured limit. | `procedures/jv_procedure.py` | Request the STAT buffer element / query TRIP (the ≥99 % suppression already shipped) |

## Open — robustness & data integrity

| ID | What breaks | Where | Fix direction |
|----|-------------|-------|---------------|
| G1 | **Intermittent segfault while a second `JVAnalyzerWindow` is constructed.** No traceback; faulthandler points at `combined_tab.py` `findChild`, `spo_widget` card building or `PlotWidget` construction — three places that walk/extend a widget tree, i.e. where a stale pointer is *touched*, not where it is created. A PyQt wrapper and its C++ object get out of sync and some registry keeps a pointer into freed memory; reading it usually works, occasionally does not. Reaches the real app only via `main.py`'s relogin loop (logout → login builds a new window); never mid-measurement, so nothing already written is at risk. ~1 in 4 full-suite runs with fixed ordering. Two instances of the pattern have been found and fixed (pymeasure `LogWidget._blink_qtimer`, pyqtgraph `ViewBox.AllViews`) — neither was this one. | `gui/`, PyQt5/pyqtgraph/pymeasure interop | Run under gdb/faulthandler with `PYTHONMALLOC=debug` to catch the first invalid access rather than the crash; audit every parentless Qt object held in a module or class attribute |
| A1 | `queue_experiment` sets `is_busy=True` **then** parses free-text params with raw `float()`; one typo raises out of the slot and no run can start again this session. | `gui/app_controller.py` | Validate before mutating state; add `QDoubleValidator` |
| A2 | Temp per-channel CSVs are `os.remove`d **before** the merged report is written — if that write fails the measurement is unrecoverable. | `gui/app_controller.py` | Write first, `os.replace`, delete last |
| A4 | Report writing uses the platform default encoding (**cp1252 on Windows**): µ/°/umlauts in Notes → `UnicodeEncodeError` → merge fails → with A2, data loss. `spo_report.py` is correct; only `app_controller.py` is wrong. | `gui/app_controller.py` | `encoding="utf-8"` everywhere + atomic write |
| A5 | Filename gate checks the stripped text but *uses* the raw text: `..\`, `:`, `*`, trailing dots pass and fail at write time. | `gui/widgets/file_panel.py` | Validate and use the same sanitized string |
| A6 | The L1 "JV+SPO ⇄ Advanced" toggle has no busy guard, so the operator can hide the Abort button mid-run with the source on. | `gui/jv_analyzer_window.py` | Same busy guard the Advanced toggle already has |
| A7 | `start_session` never raises the root logger level (default WARNING drops INFO) and the main window blanket-removes root handlers — **the per-session audit log is effectively empty.** | `auth/session.py`, `gui/jv_analyzer_window.py` | Set the level in `start_session`; remove only own handlers |
| A8 | No `sys.excepthook`; PyQt5 turns any unhandled slot exception into `abort()` — the process dies with hardware possibly energized. (pyqtgraph installs one of its own as a side effect, which is why this has not been seen in practice — do not rely on it.) | `src/solarjv_analyzer/main.py` | Install an excepthook that de-energizes, logs, and shows a dialog |
| C5 | `reset_password` needs only username+email (OTP verification just enables a text field); OTPs come from non-crypto `random`. | `auth/database.py`, `auth/email_service.py` | `secrets` for OTPs; `verify_otp` returns a short-lived token `reset_password` requires |
| C6 | `src/solarjv_analyzer/users.db` (plaintext `admin/admin`) is committed and ships in every installer; `utils/database.py` is one import from re-enabling plaintext auth. | repo root of the package | Delete both; add `*.db` to `.gitignore` |
| A9 | No login rate limiting/lockout; 6-char minimum password; `starttls()` with no SSLContext (`CERT_NONE`); shared plaintext SMTP password on `S:\`. | `auth/database.py`, `auth/email_service.py` | Lockout + rehash check + verified TLS context |

**Blocking for the S-drive work** (a write-once store cannot be cleaned up
after a bad write): **A4** and **A5** must be fixed first, and **A2**'s
ordering rule is the same one the publisher follows — see
[`docs/plans/s-drive-publishing.md`](docs/plans/s-drive-publishing.md) §7.

Medium/low findings (NPLC off-by-one, autorange latency, dark gap between
branches, `pre_sweep_delay` applied twice, bare `except` on MUX deselect, email
case handling, `config.py` fields without annotations, date rollover mid-run,
block-tag parser vs `[[` in Notes, …) are listed in
[§5 of the archive](docs/audit-2026-08.md#5-medium--low-findings-condensed).

## Fixed (do not re-open)

Every fix below was made test-first, with expected values derived analytically
from the generating physical model — never transcribed from the implementation.

| Area | Findings | Regression tests |
|------|----------|------------------|
| Metric math | S1 SPO drift sign (generated power `P_gen = −(V·I)`), S2 signed Vmpp, S3 NaN instead of extrapolated Voc/Isc, S4 branch mixing, quintic MPP → anchored local parabola, voltage-span Rs/Rsh window, Voc regression direction, FF noise guard | `tests/analysis/test_scientific_correctness.py`, `tests/spo/test_spo_scientific_correctness.py` |
| Combined JV+SPO | C1 `QtWidgets` NameError, A3 abort ordering, single-sweep-mode key, param snapshotting, worker joins, abort-aware SPO sleep, one sample retry, `try/finally` on the SPO mode switch | `tests/spo/`, `tests/gui/` |
| Shared widgets | Duplicate bottom section left one view permanently blank | `tests/gui/test_shared_bottom_section.py` |
| Calibration | Hardware pill never updated / no retry path; compliance-clamped points suppressed from plot, files and metrics | `tests/procedures/test_compliance_flagging.py` |
| C2 | ✕-closing the calibration window left the Keithley energized | `tests/gui/test_calibration_close_event.py` |
| C2a | The **hand-off** to the main window also closes the calibration window, so that `closeEvent` closed the VISA session the main window had just been handed (green light, `InvalidSession` on the first sweep) and ended the logging session. Fixed by one process-wide `InstrumentManager` + a `_handed_off` guard + session-liveness checks. | `tests/gui/test_calibration_handoff.py`, `tests/instruments/test_instrument_manager_session.py` |
| H6-leak | `get_keithley()` leaked the VISA session on a failed init, bricking reconnection on exclusive serial | covered in `test_instrument_manager_session.py` |
| Log view | pymeasure's `LogWidget` only works inside a `QTabWidget` (`_blinking_start` calls `tabBar()` on its grandparent) and keeps its blink state in class attributes, so Advanced had a log page and JV+SPO had none. Replaced everywhere by `gui/widgets/log_panel.py`: three panels, one shared root handler, shared history, severity by colour **and** tag. (Its parentless class-level `QTimer` was *a* dangling-pointer source — see G1 — but removing it did not stop the crash.) | `tests/gui/test_log_panel.py` |
| Control affordances | n-i-p/p-i-n were two shades of one teal (architecture flips every metric's sign convention); Advanced→SPO's "Quick JV" shared the generic `ModeButton` style whose unselected state is indistinguishable from disabled. | `tests/gui/test_control_affordances.py` |
| Run lifecycle | `Manager.abort()` clears pymeasure's `_start_on_add`/`_is_continuous` and only `Manager.resume()` restores them, so any abort not followed by a Resume click left the manager unable to start anything for the rest of the session — the next run queued rows that never executed. Aborting the last sweep also skipped all post-processing, leaving completed sweeps as orphaned `_temp` files; the aborted sweep's own partial curve was merged into the report as a real measurement; a reverse-only channel matched no branch in `_process_multi_files`; the button's label and its `clicked` connection were set in five places and drifted apart; and `abort_experiment` let `Manager.abort()`'s "no experiment is running" exception escape a Qt slot. | `tests/gui/test_abort_resume_scenarios.py`, `tests/gui/test_abort_resume_lifecycle.py` |
| Output routing | A combined JV+SPO run filed its SPO raw CSV and report under **Main**: `start_combined_run` built both paths from the file panel's directory, which is always the Main folder. Standalone SPO was unaffected (`SpoProcedure` resolves its own folder), but combined mode passes an explicit `csv_path` that bypasses it. | `tests/gui/test_output_routing.py` |
