# Spec — hardware status lights & dynamic port detection

Two changes that share one question: *which port is each instrument on, and is
it there right now?*

- **Part A — status lights.** Built and in the repo (267 tests green). Written
  out here so the behaviour can be reviewed and adjusted before it is accepted.
- **Part B — dynamic port detection.** **Built** (2026-09-17):
  `instruments/port_resolver.py`, `tools/port_probe.py`, 24 tests. Resolution
  runs automatically at startup — there is no operator-initiated Detect step,
  per the decision to make it work "without fail" on launch.
- **Part C — how they interact.** Part B changes what Part A checks against;
  the rules that join them are the easiest thing to get wrong.

Notation: `IF … THEN …`, with `AND` for conditions that must all hold and `OR`
for alternatives. Rules within a section are evaluated **top to bottom, first
match wins**, unless the section says otherwise.

---

## Part A — the status lights (as built)

### A0. What the lights mean

> **IF** the app holds an open handle to the instrument
> **AND** the instrument's COM port is currently enumerated by Windows
> **THEN** the light is **green**
> **OTHERWISE** it is **red**.

Both halves are needed. `pyserial`'s `is_open` and a PyVISA session handle both
stay valid after the USB adapter is physically unplugged — they only find out
on the next failed I/O — so a light driven by the handle alone would sit green
with the cable in your hand.

> **IF** the instrument's VISA resource is **not** an `ASRL…` resource
> (a GPIB or USB-TMC Keithley)
> **THEN** the presence half is skipped and the light reflects the handle only
> — there is no COM port to look for, and absence must not be inferred from it.

The operator sees a coloured dot and a one-line tooltip (`MUX: Connected`).
Nothing else — no port names, no adapter names, no reasons. Those go to the
session log.

### A1. When the check runs

| # | Condition | Result |
|---|---|---|
| A1.1 | **IF** the main window has opened | **THEN** start the monitor **AND** run one check immediately |
| A1.2 | **IF** the poll interval (3 s) elapses | **THEN** run a check |
| A1.3 | **IF** a check is already running | **THEN** skip this tick (never queue two) |
| A1.4 | **IF** the window is closing | **THEN** stop the timer **AND** wait for the worker to finish |

Every check runs on a worker thread. Enumerating ports is a pure read, but
opening one is not, and a ghost COM port can block for seconds.

### A2. Whether the check may *act*

> **IF** `is_busy` **OR** `spo_running` **OR** `_combined_mode` is set
> **THEN** the check may only **look** — it must not open, close or probe
> anything
> **AND IF** reading those flags raises for any reason
> **THEN** assume a measurement **is** running (fail safe).

### A3. Opening the MUX

| # | Condition | Result |
|---|---|---|
| A3.1 | **IF** may-act **AND** the MUX port is enumerated **AND** no open handle | **THEN** open it |
| A3.2 | **IF** the port is **not** enumerated | **THEN** do not attempt to open — cheaper than a failed open with its timeout, and it stops a missing instrument being hammered every 3 s |
| A3.3 | **IF** the open raises | **THEN** record the reason, leave the light red, carry on — the window must stay usable |
| A3.4 | **IF** a handle is already open | **THEN** do nothing |

The Keithley is **never** auto-opened by the monitor — see open question Q1.

### A4. Reporting

> **IF** the (keithley_connected, mux_connected) pair differs from the previous
> check **THEN** write one INFO line to the session log with both ports, both
> adapter descriptions and any open error
> **OTHERWISE** log nothing — at one check every 3 s, logging every tick would
> bury the run's own messages.

### A5. What the lights never do

- Never a control: no click handler, no pointing-hand cursor.
- Never abort or interfere with a running measurement — they report only.
- Never show a port name, adapter name or error on screen.

---

## Part B — dynamic port detection (proposed)

### B0. Why

`MUX_PORT` and `GPIB_ADDRESS` live in `config.py`, which is **inside the
packaged app**. After a Briefcase build there is no file to edit, and COM
numbers change on their own — a different USB socket, a driver update, another
device claiming the number first.

Confirmed on the lab PC: the two instruments use **different adapter chips**,
so they can be told apart by chip alone.

| Instrument | Chip | Windows description | Identified by |
|---|---|---|---|
| MUX | CH340 | `USB-SERIAL CH340` | USB identity only — the protocol has no reply (audit S7) |
| Keithley | Prolific PL2303GT | `Prolific PL2303GT USB Serial COM Port` | USB identity **AND** `*IDN?` |

### B1. Where settings live

`~/.solarjv/config.json` — the per-user file `DirectoryManager` and the store
already use. Outside the app bundle, so it survives reinstalls and can be
hand-edited in a pinch. Proposed keys: an explicit `mux_port` /
`keithley_resource`, and a remembered `mux_identity` / `keithley_identity`
(`{vid, pid, serial, location, description}`).

### B2. Resolving the MUX port — first match wins

| # | Condition | Result |
|---|---|---|
| B2.1 | **IF** an explicit `mux_port` is set **AND** it is enumerated | **THEN** use it |
| B2.2 | **IF** an explicit `mux_port` is set **AND** it is **not** enumerated | **THEN** unresolved — **do not** silently fall through to a different device; the operator chose this one |
| B2.3 | **IF** a remembered identity matches **exactly one** enumerated port | **THEN** use that port's current COM number **AND** refresh the cached number |
| B2.4 | **IF** no remembered identity **AND** exactly one port matches the CH340 signature | **THEN** use it **AND** remember its identity |
| B2.5 | **IF** more than one port matches the CH340 signature **AND** none is remembered | **THEN** **unresolved — ambiguous**. Do not guess; ask |
| B2.6 | **IF** nothing above matched **AND** `config.MUX_PORT` is enumerated | **THEN** use it (compatibility fallback) |
| B2.7 | **OTHERWISE** | **THEN** unresolved — report every port that *was* seen, not "could not connect" |

### B3. Resolving the Keithley resource — first match wins

| # | Condition | Result |
|---|---|---|
| B3.1 | **IF** an explicit `keithley_resource` is set | **THEN** use it **AND** do not probe |
| B3.2 | **IF** `config.GPIB_ADDRESS` is a non-`ASRL` resource (GPIB / USB-TMC) | **THEN** use it as-is **AND** do not probe — there is no COM port to search |
| B3.3 | **IF** a remembered identity matches exactly one enumerated port | **THEN** use `ASRL<n>::INSTR` for it |
| B3.4 | **IF** probing is permitted (B4) **AND** ports match the PL2303 signature | **THEN** probe those **first**, in order; the first answering `KEITHLEY…2400` wins **AND** is remembered |
| B3.5 | **IF** no PL2303 candidate answered **AND** probing is permitted | **THEN** probe the remaining eligible ports (B4) |
| B3.6 | **IF** `config.GPIB_ADDRESS` names an enumerated port | **THEN** use it (compatibility fallback) |
| B3.7 | **OTHERWISE** | **THEN** unresolved |

### B4. Probing safety — every condition must hold

> **IF** the port is enumerated
> **AND** it is not already open by us
> **AND** it is not the port resolved (or remembered) as the **MUX**
> **AND** no measurement is running
> **AND** the operator asked for detection, **OR** nothing is remembered or
> configured yet
> **THEN** the port may be probed — with `*IDN?` **only**, ~300 ms timeout, one
> retry, on a worker thread
> **OTHERWISE** it must not be touched.

Rationale, in order of importance: never send SCPI to the MUX (it speaks a
binary hex protocol and `*IDN?` could be read as a frame); never disturb a
colleague's instrument on the same bench; never probe mid-measurement; and
`*IDN?` is read-only — never `*RST`, never an output command, to a device we
have not identified.

> **IF** a probe returns something that is not a Keithley 2400
> **THEN** record it as "not the Keithley", move on, and do not probe it again
> this session.

### B5. When detection runs

| # | Condition | Result |
|---|---|---|
| B5.1 | **IF** settings resolve **AND** the ports are enumerated | **THEN** no probing at all — the fast path, one enumeration read |
| B5.2 | **IF** nothing is remembered or configured (first run on this machine) | **THEN** run detection once at startup |
| B5.3 | **IF** the operator presses **Detect** | **THEN** run detection regardless of what is remembered |
| B5.4 | **IF** resolution fails at startup | **THEN** show red, log the detail, **AND** make the Instrument tab's port settings reachable — never a silent failure, never a retry loop |

### B6. The settings UI

Extends the existing Instrument tab (which already shows the GPIB address as an
editable field) rather than adding a dialog. It must also be reachable from the
**calibration window**: calibration needs the Keithley before the main window
exists, so an operator stuck at the gate must be able to fix ports there.

Shows every enumerated port with its `VID:PID`, serial, description and what we
think it is; a **Detect** button; a **Test** button per instrument (Keithley:
show the `*IDN?` reply; MUX: report that the port opened — a real test needs
the firmware ACK that does not exist); and **Save**.

---

## Part C — how the two interact

Today the monitor checks against `config.MUX_PORT` and `config.GPIB_ADDRESS`.
Once ports are resolved dynamically it must check against the **resolved**
values, or the lights will be wrong exactly when the feature is doing its job.

| # | Condition | Result |
|---|---|---|
| C1 | **IF** ports are resolved | **THEN** the monitor uses the resolved port/resource, never `config.py` directly |
| C2 | **IF** a remembered identity now appears at a **different** COM number (replugged into another socket) **AND** no measurement is running | **THEN** re-resolve, close the stale handle, reopen on the new port — the light goes green again by itself |
| C3 | **IF** that happens **while a measurement is running** | **THEN** only report red; never close or reopen mid-run |
| C4 | **IF** resolution is ambiguous (B2.5) | **THEN** the light is red **AND** the log says "ambiguous", not "not connected" — they need different fixes |
| C5 | **IF** resolution has never succeeded | **THEN** the light is red **AND** the monitor does not probe on its own (B4); detection is operator-initiated |
| C6 | **IF** the adapter disappears mid-measurement | **THEN** the light goes red, **AND** the monitor does nothing else — the run fails on its own next I/O and that is the measurement code's business, not the indicator's |

---

## Review findings (2026-09-19)

A re-read of the first implementation found five defects, all now fixed and
covered by tests:

| # | Defect | Why it mattered |
|---|---|---|
| 1 | `refresh_if_unconfirmed()` probed serial ports, and is called from `connect_*()` — i.e. the GUI thread, including `CalibrationWindow.__init__` | Breaks the project's rule against serial I/O on the GUI thread; a ghost COM port can block the open for seconds, freezing the window |
| 2 | Probing could open a port the app already held | The open fails because the port is busy, which reads as "not the Keithley" — and could move a live connection to a different port |
| 3 | A refresh could *downgrade* a good resolution and persist the worse value | Unplug the MUX mid-session and a known-good port was replaced with nothing |
| 4 | `"usb-serial"` in the MUX keyword list | Far too broad; an unrelated adapter makes the MUX look *ambiguous*, which refuses to resolve — worse than the mis-identification the check exists to prevent |
| 5 | The Instrument tab displayed `config.GPIB_ADDRESS` | Showed an address the app was not using, in a feature whose whole point is that config.py is not the source |

The fix for (1) and (2) confined probing to startup: `refresh_if_unconfirmed()`
never opens a port.

**Correction (same day).** The first version of this fix also stopped probing a
*lone* PL2303 adapter, justified by the instrument being switched off at launch.
That justification was wrong: the rack is powered on before the app starts, and
the calibration checklist is a reminder for the operator rather than something
the software waits on. A lone adapter is therefore probed again — the reply is
logged as confirmation, and a silent adapter still resolves, with a warning.
Following from the same correction, `main.py` now connects **both** instruments
on a background thread as the app opens, instead of waiting for the calibration
window (Keithley) and the first run (MUX).

## Open questions — please decide

**Q1–Q3 are moot** — they concern the status lights, which are staying as
built.

**Q1. Should the monitor auto-reconnect the *Keithley*?** Today it only
auto-opens the MUX; the Keithley is opened by the calibration gate and, if it
drops, the light goes red and stays red until the next run attempt. Silently
reopening a SourceMeter is riskier than reopening a MUX — it may have its
output on, and reconnecting hides a fault the operator should probably see.
My recommendation: **leave it manual**, and make a red Keithley light stop a run
from starting (which the pre-flight check already does).

**Q2. Poll interval.** 3 s costs one registry read per tick. Is that the right
trade between "the light is current" and "something is happening in the
background"? 5 s would be quieter.

**Q3. Should an ambiguous or unresolved port block a run?** Currently a run
attempt will fail at connect time with a message. The alternative is to disable
the run buttons while either light is red. That is safer but more intrusive —
and it would have to be careful not to trap an operator whose light is briefly
red for an unrelated reason.

**Q4. Remember by serial number or by socket?** If the adapters expose unique
serial numbers, identity should pin to the *cable*. If they do not (common for
CH340), the fallback is `location` — the USB hub+port — which is stable only
while the same socket is used. The `list_ports` one-liner from the install guide
answers this; worth running before B is built.

---

## Build order, once approved

1. `tools/port_probe.py` — list every port with full identity, optional
   `*IDN?`. Run on the lab PC; confirms Q4 and the real `VID:PID` values.
2. `instruments/port_resolver.py` — B1–B3 as pure functions over a mocked
   `comports()`, fully unit-tested. Nothing wired up.
3. Wire the resolver into `InstrumentManager`, with `config.py` as the
   documented fallback (B2.6 / B3.6). Behaviour unchanged when no settings file
   exists.
4. Part C — point the status monitor at the resolver.
5. The settings UI (B6).
6. Optional, firmware-dependent: a MUX identify/ACK command, which would make
   B2 a handshake instead of an inference and close audit S7.

Steps 1–3 change no measurement, analysis or instrument-control behaviour: they
only decide which port string is handed to the existing `connect_*()` calls.
