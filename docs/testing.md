# Testing & debugging notes

## What to run

See the table in [`CLAUDE.md`](../CLAUDE.md#verify-what-you-changed). Short
version: run the tests for the area you touched; run the whole suite when the
change spans areas or before you hand work over.

Timings (offscreen, for calibrating your own patience):
whole suite ~6–16 s · non-GUI ~4 s · `tests/gui` ~4 s ·
`render_preview.py` a few seconds.

## Fixtures every window test needs

Tests that build full windows **must**:

- reset the singleton: `DirectoryManager._instance = None` before and after, and
  `monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))`;
- snapshot and restore the root-logger handlers. Window log handlers outlive
  the window, and a dead C++ Emitter crashes every later log call — in an
  unrelated test.

`tests/conftest.py` already revives pymeasure's parentless class-level
`LogWidget._blink_qtimer` before every test, so any GUI test can build windows
in any order.

Copy the fixtures from `tests/gui/test_calibration_handoff.py` or
`test_shared_bottom_section.py` rather than writing new ones.

## Traps that have bitten

- **`np.arange` with float steps overshoots its endpoint** — build fixture grids
  from integers. Duplicate-averaging in analysis merges only bit-exact
  voltages.
- **Controller code binds to widget attribute NAMES**, so a rename breaks
  silently at runtime, not at import. Read the whole function *and its callers*
  before renaming anything on a widget, and preserve signal signatures.
- **`CalibrationWindow.close()` is two paths in one.** `_on_proceed`/`_on_skip`
  emit `calibration_passed` (delivered **synchronously** — `main.py`'s slot
  builds the main window and closes this one) and then call `close()` again. So
  `closeEvent` fires *inside* the hand-off and runs more than once. Only the
  `_handed_off` flag (set **before** the emit) stops it from de-energizing the
  Keithley, closing the VISA session the main window was just handed, and
  calling `SessionManager.end_session()` — which strips the root-logger file
  handler (the run goes unlogged) and clears `current_user` (closing the main
  window bounces back to the login dialog). Regression test:
  `tests/gui/test_calibration_handoff.py`.
- **A filename predicate is not a guard.** The store sweeper once decided
  "is this file finished?" from its name alone, missed a temp file whose
  extension was empty, and deleted it while pymeasure still had it open — which
  appears as `KeyError: 'Voltage (V)'` from pandas reading a vanished file. It
  now renames a file before touching it: on Windows a rename that succeeds
  proves no other process holds it open. Prove ownership, don't infer it.
- **Never call PyVISA/PySerial on the GUI thread.** Always `worker.abort()` then
  `worker.wait(timeout)` before disconnecting instruments.

## Debugging playbook

1. **Triage first.** If the symptom and location are obvious from reading the
   widget and its layout (a clip, a missing call, a wrong binding), *edit
   first* — the test + render loop **is** the reproduction. Write a
   numeric/offscreen reproduction before editing only when the bug is
   intermittent, cross-file, or the cause isn't visible in the code.
2. **Read the whole function and its callers** (see the attribute-name trap).
3. **Grep [`AUDIT.md`](../AUDIT.md)** for the area — the bug may already be
   diagnosed there, with a fix direction.
4. **After fixing, prove it with the same reproduction.**

**Anti-probing guardrail:** if you've run more than 2 diagnostic probes and
still don't know the root cause, stop probing and read the code. Diagnosis
sprawl — not the test suite — is the time sink. A visual-clip bug is usually one
line in the widget plus its layout.
