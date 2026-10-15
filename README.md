<p align="center">
  <img src="Login.png" alt="SolarJV Analyzer" width="320" />
</p>

<h1 align="center">SolarJV Analyzer</h1>

<p align="center">
  <strong>Automated J–V Characterisation, MPP Tracking & SPO Degradation Testing</strong><br>
  <em>University of Stuttgart · Institut für Photovoltaik (ipv)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-≥3.8-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/tests-468%20passed-brightgreen.svg" alt="Tests" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License" />
  <img src="https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey.svg" alt="Platform" />
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Data storage and attribution](#data-storage-and-attribution)
4. [System Architecture](#system-architecture)
5. [Hardware Requirements](#hardware-requirements)
6. [Installation](#installation)
7. [Running the Application](#running-the-application)
8. [Solar Cell Metrics — Calculation & Validation Guide](#solar-cell-metrics--calculation--validation-guide)
9. [Device Architecture: n‑i‑p vs p‑i‑n](#device-architecture-n‑i‑p-vs-p‑i‑n)
10. [Analysis-Settings Field Wiring (audit note)](#analysis-settings-field-wiring-audit-note)
11. [SPO — Set‑Point Operation](#spo--set‑point-operation)
12. [MPP — Maximum Power Point Tracking](#mpp--maximum-power-point-tracking)
13. [The shared Hold Voltage card](#the-shared-hold-voltage-card)
14. [Commissioning a lab PC](#commissioning-a-lab-pc)
15. [Testing](#testing)
16. [Building Standalone Applications](#building-standalone-applications)
17. [Project Structure](#project-structure)
18. [Configuration](#configuration)
19. [Data Format](#data-format)
20. [Development Guide](#development-guide)
21. [License](#license)

---

## Overview

**SolarJV Analyzer** is a Python-based, PyQt5 graphical application for automated, multi-channel **J–V (current–voltage) sweep characterisation**, **MPP (maximum power point) tracking** and **SPO (Set‑Point Operation) long-term stability testing** of solar cells. It provides a complete experimental pipeline: from instrument control to real-time visualisation, on-the-fly solar cell metric extraction, and structured data export.

The Advanced tab offers three measurement modes — **JV Sweep**, **MPP** and **SPO** — sharing one instrument stack, one output pipeline and one Hold Voltage card.

The software interfaces directly with a **Keithley 2400 SourceMeter** and a **6‑channel multiplexer** to measure up to six solar cells sequentially in a single experimental run. It supports **dual-direction sweeps** (forward + reverse) for hysteresis analysis and **device architecture selection** (n‑i‑p / p‑i‑n) with quadrant-agnostic metric computation.

Designed and maintained at the **Institute for Photovoltaics (ipv), University of Stuttgart**, this tool replaces manual lab-book workflows with a reproducible, auditable, and user-friendly measurement platform.

---

## Key Features

### Measurement & Instrument Control

| Feature | Description |
|---------|-------------|
| **Multi-Channel J–V Sweeps** | Hardware-controlled staircase sweeps on up to 6 channels via the Keithley 2400's built-in trigger model and buffer |
| **Dual-Direction Sweeps** | Per-channel Forward + Reverse sweeps with automatic hysteresis-aware metric extraction |
| **SPO Stability Testing** | Constant-voltage hold with configurable duration, sampling interval, and pre-conditioning time |
| **Real-Time Plotting** | Live I–V and Power–vs–Time curves via pyqtgraph with channel-coloured traces |
| **Instrument Status** | Red/green connectivity lights for Keithley and MUX |

### Analysis & Computation

| Feature | Description |
|---------|-------------|
| **14 Solar Cell Metrics** | EFF, FF, Voc, Jsc, Vmpp, Jmpp, Pmpp, Isc, Rsh, Rs, Rho_shunt, Rsq, Area, Incident Power |
| **Quadrant-Agnostic** | Automatically handles both n‑i‑p (Q4: V>0, I<0) and p‑i‑n (Q2: V<0, I>0) architectures. Metrics report positive magnitudes except **Vmpp, which keeps its sign** (negative for p‑i‑n) so the SPO hold polarity is correct |
| **Anchored Local-Parabola MPP** | Global measured optimum anchors an adaptive-window local quadratic fit with iterative recentering — tracks sharp MPP knees without global-polynomial underfit, averages noise on flat maxima |
| **Curvature-Aware Voc/Isc** | Local regression at the zero crossing with residual-gated quadratic refinement; **no crossing → NaN**, never an extrapolated (fabricated) value |
| **Rs / Rsh Extraction** | Voltage-span-selected local slope regression (±max(2·step, 25 mV)) at V=Voc (Rs) and V=0 (Rsh) |
| **SPO Metrics** | Mean power, standard deviation, drift percentage, energy yield (trapezoidal integration), peak/trough power |
| **MPP Tracking** | Perturb-and-observe tracker with bounded excursion; reports drift, probe count, direction flips and clamp hits |

### Data Management & UI

| Feature | Description |
|---------|-------------|
| **Structured CSV Export** | Experiment parameters, analysis summary, and measurement data in a single self-documenting file with `[[ EXPERIMENTAL PARAMETERS ]]`, `[[ ANALYSIS SUMMARY ]]`, and `[[ MEASUREMENT DATA ]]` block tags |
| **File Loading & Restoration** | Previously saved measurement files are fully restored — plot curves, browser tree, analysis matrix, and channel data |
| **Crash-Safe SPO / MPP Logging** | Every sample is flushed to disk immediately with `fsync` — a crash never loses previously recorded data, and an interrupted MPP run's journal is promoted to a real report |
| **Atomic Report Store** | Reports reach the shared store only when complete: write `.part` → fsync → close → rename, reconciled by a file watcher plus a periodic sweep |
| **Shared Hold Voltage Card** | One widget serves MPP and SPO — Manual or Quick JV, every field with a unit selector that converts rather than relabels |
| **User Authentication** | SQLite-backed login system with Argon2id password hashing, registration, password reset, and timestamped session logs |
| **Directory Management** | Automatic `Base/Username/Date/{Calibration\|JV\|MPP\|SPO}/` folder structure, where `Username` is the **application login** — see [Data storage and attribution](#data-storage-and-attribution) |
| **Filename Validation** | Execution buttons are gated on a non-empty filename prefix — prevents orphaned data files |

### Modern UI

| Feature | Description |
|---------|-------------|
| **Dark-Themed Login Screen** | Premium split-screen design with full-bleed lab illustration and frosted-glass form card |
| **Full-Screen Operation** | Calibration and Main windows open maximised |
| **Centred Pill Tab Bar** | Bottom dashboard with mint-green pill-style tab selectors for Experiment Queue and Channel Analysis |
| **Persistent Action Bar** | Show all / Hide all / Clear all / Open controls visible in both views |
| **Channel Colour Badges** | Vibrant channel indicators in the action bar matching plot curve colours |
| **Architecture Badge** | n‑i‑p / p‑i‑n indicator in the action bar |
| **Dynamic Selection Highlight** | Row selection in the analysis matrix highlights in the selected channel's exact plot colour |

---


## Data storage and attribution

Every measurement is filed under the name the operator **signed into the
application** with:

```
S:\Data\JV\<app login>\<yyyy-mm-dd>\<Calibration|JV|MPP|SPO>\
```

The same name is used for the local working folders, so a given person's data
appears under one name everywhere.

### Why the application login, and not the Windows account

The folder name used to come from the Windows process token, which nobody can
spoof. Every lab machine now runs under a **single shared Windows account**, so
that name is identical for everyone and would file all operators' data into one
folder with no attribution at all. The application login is what distinguishes
people, so it is what names the folder.

Two consequences follow from that choice:

* **Nothing publishes before login.** The store holds finished files and
  session logs in staging until somebody signs in — an unattributed folder is
  worse than a delayed one. This is normal at startup, not an error.
* **The identity is not cached.** One process can serve several operators via
  the re-login loop, so the name is set at login and cleared at logout. Logging
  out triggers a final publish first, so a departing operator's files never
  land in the next person's folder.

### What this does and does not guarantee

Attribution is now only as strong as the password protecting an account.
Account creation is self-service from the login screen and only checks that the
username is non-empty and unique, so the store records *who signed in*, not a
fact verified by the operating system. That is appropriate for a shared lab
instrument; it is not an audit trail.

### How a finished report reaches the store

Every report is written through one helper, `store/finalize.py`:

```
write to <final>.part  →  flush + fsync  →  close  →  rename to <final>
```

The **close** is the completion guarantee — a `.part` file is never a
finished report, so a crash can leave a partial file but never a
partial-looking real one. A `QFileSystemWatcher` on the staging folders
triggers a sweep within ~300 ms of the rename, with a periodic sweep behind
it as the correctness net (Windows coalesces and can drop directory
notifications outright). A finished report now reaches the store in well
under a second; it used to take about 25.

Two suffixes exist, with deliberately **opposite** cleanup rules:

| Suffix | What it is | When stale |
|--------|------------|------------|
| `.part` | A report being written | **Deleted** — a half-written report is worthless |
| `.journal` | The live raw time series of a hold | **Promoted** — a crashed run's journal is the only copy of its data |

### Known limitation

Logging out triggers a final publish and waits for it. A file still being
written at that exact moment is published on a later tick, under whoever is
signed in then. In practice a run finishes long before anyone logs out, but
if you abort a measurement and immediately log out, check that the file
landed in the right folder.

## System Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│   Login      │────▶│   Calibration   │────▶│   Main Analyzer      │
│   Dialog     │     │   Window        │     │   Window             │
│  (auth.db)   │     │  (reference     │     │  (J-V Sweeps + SPO)  │
│              │     │   cell check)   │     │                      │
└──────────────┘     └─────────────────┘     └──────────────────────┘
                                                       │
                          ┌────────────────────────────┼────────────────────────────┐
                          │                            │                            │
                   ┌──────▼──────┐            ┌───────▼───────┐           ┌────────▼────────┐
                   │  JV Sweep   │            │  SPO Testing   │           │   File Loading  │
                   │  (PyMeasure │            │  (QThread      │           │   & Analysis    │
                   │   Manager)  │            │   Worker)      │           │   Restoration   │
                   └──────┬──────┘            └───────┬───────┘           └────────┬────────┘
                          │                            │                            │
                   ┌──────▼──────┐            ┌───────▼───────┐           ┌────────▼────────┐
                   │ JVProcedure │            │ SpoProcedure   │           │  AppController  │
                   │ (Keithley   │            │ (inherits      │           │  (parse, merge, │
                   │  staircase  │            │  JVProcedure)  │           │   format, load) │
                   │  sweep)     │            │                │           │                 │
                   └──────┬──────┘            └───────┬───────┘           └────────┬────────┘
                          │                            │                            │
                   ┌──────▼────────────────────────────▼───────────────────────────▼────────┐
                   │                         compute_jv_metrics / compute_spo_metrics        │
                   │                        (Pure computation — no hardware dependency)       │
                   └─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | PyQt5 + pyqtgraph |
| **Measurement Engine** | PyMeasure (Procedure / Manager / Worker) |
| **Instrument Control** | PyVISA (Keithley 2400), PySerial (MUX hex protocol) |
| **Numerical Analysis** | NumPy, SciPy (linregress, lstsq, trapezoid) |
| **Data Management** | Pandas (CSV parsing, DataFrame multi-index) |
| **Authentication** | SQLite + Argon2id password hashing |
| **Packaging** | Hatchling (build), Briefcase (native macOS .dmg / Windows .msi) |
| **Testing** | pytest (110 tests, 13 test files) |

### Thread Model

```
Main GUI Thread (PyQt5 event loop)
    │
    ├── JV Sweeps → PyMeasure Manager → Worker threads (blocking I/O safe)
    │
    ├── SPO Runs  → SpoWorker (QThread subclass, custom lifecycle)
    │
    └── All data emitted back via Qt signals (pyqtSignal) — thread-safe
```

**Critical Rule**: PyVISA and PySerial calls MUST never execute on the main GUI thread. All hardware communication runs in background workers.

---

## Hardware Requirements

| Instrument | Model | Interface | Protocol |
|-----------|-------|-----------|----------|
| **SourceMeter** | Keithley 2400 | USB / GPIB | VISA (SCPI commands) |
| **Multiplexer** | 6-channel custom MUX | Serial (RS-232) | Hex protocol (`AA010{pixel}000000BB`) |

### Calibration Setup

Before each measurement session, the calibration gate verifies:
1. Chiller, Sun Simulator, and Keithley powered ON
2. Wavelabs "AM1.5G" recipe loaded
3. Silicon reference cell (RERA) placed correctly
4. Lamp height verified at 46.8 cm

A fast J‑V sweep on the reference cell must pass Isc tolerance (±5% of target) before the main application unlocks.

---

## Installation

> **Setting up a lab PC from scratch (no Python, no drivers)?** Follow
> [`docs/install-windows.md`](docs/install-windows.md) instead — it covers the
> Python install, the VISA backend, USB-serial drivers, COM-port configuration
> and first login, step by step.

### Prerequisites

- **Python** ≥ 3.8 (development), exactly 3.9 (Briefcase packaging).
  Python ≥ 3.11 if you use the pinned `requirements-windows.txt`.
- **NI-VISA** or **PyVISA-py** backend for Keithley communication
- **Serial port** access for MUX communication

### Development Installation

```bash
git clone https://github.com/uni-stuttgart-ipv/jv_setup_keithley_controller.git
cd jv_setup_keithley_controller

python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

For a reproducible install pinned to the tested versions (recommended on lab
machines, where "it works on the other PC" needs to stay true):

```bash
pip install -r requirements-windows.txt
pip install -e . --no-deps
```

### Dependencies

```
pymeasure          # Measurement procedure framework
PyQt5              # Graphical user interface
pyqtgraph          # Real-time plotting
pyvisa             # Keithley VISA communication
pyserial           # MUX serial communication
numpy, scipy       # Numerical analysis
pandas             # CSV parsing & data management
matplotlib         # (optional) additional plotting
reportlab          # PDF report generation
argon2-cffi        # Password hashing
keyring            # OS credential vault (DPAPI on Windows) for the SMTP password
```

---

## Running the Application

### Development Mode

```bash
# Via module
python -m solarjv_analyzer.main

# Via hatch
hatch run start
```

### Application Flow

1. **Login** — Authenticate with University ID and password (Ctrl+Shift+A reveals Register / Forgot Password)
2. **Calibration** — Mandatory reference cell check. Can be skipped only for emergency data recovery
3. **Main Window** — Configure sweep parameters, select channels, enter filename, run measurements
4. **Advanced modes** — Switch the main window to **JV + SPO**, **SPO** or
   **MPP** for stability testing; the Hold Voltage card can locate the hold
   point with a Quick JV first
5. **Logout** — Disconnects instruments, ends session, returns to login

---

## Solar Cell Metrics — Calculation & Validation Guide

This section documents, for every reported metric, the exact definition, the
extraction algorithm implemented in `analysis/analysis.py`, the unit
conversions, and the edge-case policy — so the computation can be validated
against first principles without reading the source.

### Preprocessing (applies to every metric)

1. NaN samples are removed; empty or all-NaN input raises `ValueError`.
2. Data is sorted by voltage. Exact-duplicate voltages (a merged
   forward+reverse loop passed as one array) are **averaged**, and every
   metric below is computed from that single averaged curve. This guarantees
   internal consistency: FF = Pmax/(Voc·Isc) is only physically meaningful
   when all three values describe the *same* curve.
3. The function is quadrant-agnostic: it accepts n‑i‑p data in Q4
   (V > 0, I < 0) and p‑i‑n data in Q2 (V < 0, I > 0) without any flag,
   because the generating region is identified by the sign of P = V·I, not
   by an assumed polarity.

### Per-metric algorithms

| Metric | Symbol | Unit | Definition & algorithm |
|--------|--------|------|------------------------|
| **Open-Circuit Voltage** | Voc | mV | Voltage where I = 0. The first sign change of I along the sorted curve is bracketed; a local window (≤ 15 points) around the crossing is fitted with `I = m·V + c` (current is the dependent variable — the instrument sources V and measures I, so the noise lives in I) and Voc = −c/m. A local **quadratic** refinement replaces the linear root only when it reduces the fit residual by ≥ 2× (it captures the exponential knee's curvature, worth ~6 mV; on truly linear data the quadratic is numerically degenerate and is rejected). **If the sweep never crosses I = 0 (dark or truncated sweep), Voc = NaN — extrapolation would fabricate a value.** Displayed as a magnitude. |
| **Short-Circuit Current** | Isc | A | Current where V = 0, by the same local-window linear fit evaluated at V = 0. **No V = 0 crossing → NaN.** Displayed as a magnitude. |
| **Short-Circuit Current Density** | Jsc | mA/cm² | \|Isc [A]\| / Area [cm²] × 1000. |
| **Maximum Power Point** | Vmpp, Pmpp | mV, mW | P(V) = V·I is evaluated on the averaged curve; the generating region is P < 0. The measured global optimum (argmin of signed P — global by definition) anchors a **local parabola fit**: the window starts at ±max(6·ΔV, 50 mV) and doubles until the curvature is statistically resolved (quadratic residual < 0.7 × linear residual), then the fit is re-centred on its own vertex (≤ 3 iterations). Vmpp is the converged vertex, Pmpp the parabola value there. This tracks sharp MPP knees without the systematic underfit of a global high-order polynomial, and averages noise on flat maxima. **Vmpp is reported SIGNED** — negative for p‑i‑n (Q2) devices — because it is consumed as the SPO hold voltage and the polarity matters physically. |
| **Max-Power Current Density** | Jmpp | mA/cm² | Impp = Pmpp/Vmpp (exactly consistent with the reported pair), then \|Impp\|/Area × 1000. |
| **Sanity clamp** | — | — | If \|Vmpp\| > \|Voc\| (fit inconsistency), Vmpp is clamped to Voc and the operating point is **re-evaluated on the measured curve** — the reported triple is always a real point of the J-V characteristic, never a product of two independently clamped magnitudes. Skipped when Voc is NaN. |
| **Fill Factor** | FF | % | \|Pmax\| / \|Voc × Isc\| × 100, computed in SI units (V, A, W) before any display conversion. Guarded: requires \|Isc\| > noise floor (default 1 nA) and \|Voc\| > 1 µV, otherwise FF = 0 — a near-dead pixel must not report an astronomically large FF. NaN Voc/Isc also yields FF = 0. |
| **Efficiency** | η (EFF) | % | \|Pmax [W]\| / (Pin [mW/cm²] × Area [cm²] × 10⁻³ [W/mW]) × 100. |
| **Shunt Resistance** | Rsh | Ω | Local \|dV/dI\| at V = 0. The fit window is selected by **voltage span** — points within ±max(2·ΔV, 25 mV) of the target (≥ 3 points; else the 5 nearest) — then `I = g·V + b` by least squares and Rsh = 1/\|g\|. Span-based selection is essential: a fixed 20-*point* window at 50 mV steps reaches into the exponential diode knee and destroys the estimate (verified: a true 2000 Ω shunt reported as 31 Ω). \|g\| < 10⁻¹² S → ∞. |
| **Series Resistance** | Rs | Ω | Same estimator evaluated at V = Voc. Skipped (∞) when Voc is NaN. |
| **Shunt Resistivity** | ρ (Rho_shunt) | Ω·cm | Bulk shunt resistivity derived from the shunt resistance and the 4-probe geometry: ρ = Rsh × A / t, where A is the device area and t the sample thickness (R = ρ·t/A inverted). Reported as ∞ when Rsh is ∞ or no thickness is supplied. |
| **Sheet Resistance** | Rsq | Ω/sq | In-plane sheet resistance ρ / t scaled by the **4-probe lateral factor** (the finite-lateral-extent correction): Rsq = ρ / t × lateral_factor. Reported as ∞ when ρ is ∞. |
| **Contact gate** | — | — | `contact_threshold` is a minimum \|Isc\|. When \|Isc\| falls below it, the cell is judged a failed contact or dark and **all derived performance metrics are reported as NaN** (unknown) with a warning, rather than as if the cell were healthy. Disabled when the threshold is 0. |
| **Device Area** | A | cm² | User-supplied. |
| **Incident Power** | Pin | mW/cm² | User-supplied (100 = 1 sun, AM1.5G). |

### Unit conversions (exact factors)

Internal computation uses SI (V, A, W). Display conversions: Voc, Vmpp ×10³
(V→mV); Pmpp ×10³ (W→mW); Jsc, Jmpp = A/cm² ×10³ (→mA/cm²); Pin input is
mW/cm² and is converted by ×A×10⁻³ to W for the efficiency denominator.

### Validation methodology

The metric pipeline is validated in
`tests/analysis/test_scientific_correctness.py` against **analytic ground
truth**, never against the implementation's own output. Synthetic curves are
generated from the single-diode model
I(V) = −(Iph − I₀(e^(V/nVt) − 1) − V/Rsh) with Vt = 25.7 mV; the true Voc is
the machine-precision root of I(V) = 0 (Brent's method), the true MPP is the
optimum of P(V) on a 400 001-point grid, and the true local resistances come
from the analytic derivative dI/dV. Verified accuracies on a realistic 5 mV
sweep of a 20 mA, n = 1.5 cell: Voc within 0.3 mV, Vmpp within 7 mV, Pmpp
within 0.1% (0.6% with 1% current noise), Rsh exact to 4 digits at 50 mV
steps, Rs within 10% of the analytic point value. Additional tests pin the
NaN policy (truncated/dark sweeps), the signed-Vmpp contract (Q2 vs Q4), the
merged-loop consistency rule (FF of the averaged curve, exactly 25% for any
linear cell), and the physical-consistency invariant that the reported
(Vmpp, Jmpp, Pmpp) triple lies on the measured curve.

---

## Device Architecture: n‑i‑p vs p‑i‑n

The application supports both common solar cell architectures via a toggle switch in the J‑V Sweep parameter panel:

| Architecture | Quadrant | Voltage | Current | Example |
|-------------|----------|---------|---------|---------|
| **n‑i‑p** | Q4 | V > 0 | I < 0 | Standard perovskite, CIGS, CdTe |
| **p‑i‑n** | Q2 | V < 0 | I > 0 | Inverted perovskite, organic PV |

### How It Works

1. The user selects the architecture via a toggle switch (`n‑i‑p` / `p‑i‑n`) in the Parameters tab
2. The selection is stored as `Device Architecture` in the `[[ EXPERIMENTAL PARAMETERS ]]` block of every saved CSV
3. On file load, the architecture is automatically parsed and displayed as a badge in the persistent action bar
4. Legacy files without the `Device Architecture` key default to `n‑i‑p`

**Important**: `compute_jv_metrics()` is **architecture-agnostic** — it extracts positive magnitudes from both Q2 and Q4 without any sign-convention flag. The architecture metadata does not affect the mathematics. It now also drives a **polarity cross-check**: if the measured signed Vmpp contradicts the declared architecture (e.g. a positive Vmpp for a `p‑i‑n` device), the backend logs a warning and flags `polarity_ok = False` — surfacing a miswired or mislabelled cell without altering any metric.

---

## Analysis-Settings Field Wiring (audit note)

The Analysis Settings tab contains four "4-probe" fields whose real effect was
audited against the backend. The result:

| Field | Effect on measurement / core metrics | Notes |
|-------|--------------------------------------|-------|
| **Contact Threshold** | Yes (gating) | Minimum \|Isc\|. Below it, the cell is treated as a failed contact / dark cell and all derived metrics are suppressed to NaN. |
| **Sample Thickness** | Derived only | Feeds `Rho_shunt = Rsh·A/t` and `Rsq = ρ/t`. Does **not** affect Voc/Jsc/FF/EFF/Rs/Rsh or the sweep itself. |
| **4-Probe Lateral Factor** | Derived only | Scales the reported sheet resistance `Rsq = ρ/t × lateral_factor`. Does **not** affect the measurement or core metrics. |
| **4-Probe Spacing** | **None** | Belongs to a standalone 4-point-probe instrument (collinear-probe geometry). This analyzer never performs a 4-point-probe V/I read — it derives resistance from the J-V curve via the Keithley — so this value is recorded as metadata only and has no effect on any calculation. |

> **Why the probe spacing is inert:** this instrument measures resistance from
> the current–voltage characteristic of the cell itself (Keithley SourceMeter),
> not from a separate four-point-probe measurement. The `4-Probe Spacing`
> field describes the physical geometry of a 4-point-probe instrument that is
> not part of this measurement chain. It is retained as metadata for lab
> record-keeping; the same applies in spirit to the "4-Probe Lateral Factor",
> which only rescales the optional derived sheet-resistance figure.

**Sense-mode default:** the instrument tab defaults to **4-wire** (Kelvin)
remote sense (`:SYST:RSEN ON`), the correct choice for solar-cell J-V because
it removes cable and contact resistance from the series-resistance (Rs)
reading. Use 2-wire only when the Keithley's SENSE HI/LO leads are not
connected at the cell.

---

## SPO — Set‑Point Operation

SPO (Set‑Point Operation) is a passive stability test: the Keithley sources a fixed voltage (typically Vmpp found by a quick J‑V sweep) and the current is sampled at a regular interval for a configurable hold duration.

### SPO Workflow

1. **Quick JV** — A fast single sweep locates Vmpp and auto-fills the hold voltage
   (the [shared Hold Voltage card](#the-shared-hold-voltage-card))
2. **Configuration** — Set hold duration, sampling interval, and pre-conditioning time
3. **Pre-Conditioning** — The cell stabilises at the hold voltage before logging begins
4. **Data Collection** — Current and power are sampled every N seconds, written to a crash-safe CSV
5. **Report Generation** — On completion (or abort), a formatted report is generated with all SPO metrics

### SPO Metrics — Calculation & Validation

**Sign convention (fundamental):** all SPO power metrics are computed on the
**generated power** P_gen = −(V·I), which is positive whenever the cell
delivers power — in Q4 (V > 0, I < 0) and equally in Q2 (V < 0, I > 0).
This makes the metrics physically readable: a degrading cell shows
**negative** drift, "Max Power" is the best sample, and energy is the net
energy delivered. (Computing on raw signed V·I — which is negative while
generating — inverted the drift sign: a cell decaying from 9 mW to 6 mW
reported +33% "drift".)

| Metric | Definition |
|--------|-------------|
| Mean Power (mW) | Mean of P_gen over the hold — positive for a healthy cell |
| Std Power (mW) | Standard deviation of P_gen — stability indicator |
| Initial / Final Power (mW) | Mean P_gen of the first / last 5 samples (edge averages suppress single-sample noise) |
| Drift (%) | (P_final − P_initial) / \|P_initial\| × 100 on the edge averages. **Negative = degradation.** If \|P_initial\| < 1 nW the drift is undefined and reported as NaN — never as a fake number |
| Max / Min Power (mW) | Best and worst P_gen sample during the hold |
| Total Energy (J) | Trapezoidal integration of **signed** P_gen over time: net generated energy in Joules (W·s); intervals where the cell consumes power subtract, as physics requires |
| Sample Count | Number of valid (non-NaN) samples |

Input handling: time/current/voltage arrays must have equal lengths
(`ValueError` otherwise); NaN samples are dropped; out-of-order timestamps
are sorted before integration (a trapezoid over non-monotonic time silently
produces cancelling segments).

Validation (`tests/spo/test_spo_scientific_correctness.py`): expected values
are hand-derived from constructed time series — e.g. a linear 15→10 mW decay
gives exact edge averages (14.90 / 10.10 mW) and drift −32.21%; constant
10 mW × 100 s integrates to exactly 1.000 J; a generate-then-consume series
verifies net (not absolute) energy; Q2 and Q4 holds of the same cell must
produce identical metrics.

### Crash Safety

Every SPO sample is written to a raw CSV file and flushed to disk with `os.fsync()` immediately after measurement. If the application or OS crashes mid-run, all data collected up to that point is safely preserved.

---

## MPP — Maximum Power Point Tracking

MPP is an **active** stability test, where SPO is a passive one. Instead of
holding a fixed voltage, the tracker perturbs the operating point and keeps
whichever direction produced more power — so it follows the maximum as the
cell ages, soaks or warms.

### The algorithm

Perturb-and-observe:

1. **Baseline** — hold the present voltage for `Probe Interval`, recording a
   sample every `Data Interval`
2. **Probe** — step one `Probe Step` in the current direction and record
   `Probe Points` samples
3. **Decide** — compare the sums of the last *N* powers from each phase; keep
   the new voltage if it won, otherwise reverse direction
4. Repeat until `Run Time` elapses

A `Max Excursion` window around the start voltage bounds the search. Hitting
it is **counted and reported** (`clamp_hits`), never silently obeyed.

### Choosing the intervals — this matters more than it looks

One cycle takes `Probe Interval + Probe Points × Data Interval` and moves the
operating point by exactly **one** `Probe Step`. With the defaults (60 s
interval, 3 points, 10 s data, 10 mV step) a cycle is ~90 s and a 5-minute run
travels **about 30 mV in total**. A start voltage 100 mV away will never reach
the maximum, and the report will still look entirely plausible.

Two practical consequences, both learned on hardware:

* **For a short run, shorten the probe interval.** 10 s interval with 1 s data
  gives a 13 s cycle — roughly 23 cycles in 5 minutes.
* **Keep `Probe Interval` several times `Data Interval`.** Each baseline phase
  must collect several samples; if the two are equal, the comparison degrades
  to one sample against one and noise decides the direction.

On a soaking perovskite, power can drift upward faster than a 10 mV step
changes it. The tracker then follows the drift rather than the P(V) curve —
visible as the voltage walking steadily in one direction while power rises
monotonically. Pre-condition for longer, or probe less often, if you see that.

### Sign convention

Identical to SPO. Powers are reported as **magnitudes** (like `Pmpp`) and
voltages **signed** (like `Vmpp`). The live plot shows generated power
−(V·I), positive while the cell generates in Q2 and Q4 alike — so a cell that
is *consuming* plots below zero, which is exactly what you want to see.

### Output

A journal (`.journal`) is written and fsynced per sample, and promoted to a
real report if the run crashes. On completion the formatted report carries
the parameters, an analysis summary (`mean_power_mw`, `drift_percent`,
`efficiency_percent`, `probe_count`, `direction_flips`, `clamp_hits`, …) and
the full time series with a `Phase` column marking `baseline` / `probe`.

`Open MPP Report…` reads one back into the live view.

> **Note:** `run_time_s` in the summary is the **last sample's timestamp**,
> which is up to one sample interval short of the true elapsed run time.

---

## The shared Hold Voltage card

MPP and SPO use **one** widget (`gui/widgets/vmpp_card.py`) for setting the
hold / start voltage. It has two modes:

**Manual** — just the voltage box and its unit selector.

**Quick JV** — a self-contained mini sweep, with seven fields in this order:

| Field | Units |
|-------|-------|
| Start | V / mV |
| Stop | V / mV |
| Step Size | V / mV |
| Sweep Rate | V/s / mV/s |
| Compliance | A / mA / µA |
| Device Area | cm² / mm² / m² |
| Device Architecture | n‑i‑p / p‑i‑n toggle |

These open with the **same defaults as the J–V Sweep tab** (1.2 V → −0.2 V,
−10 mV, 100 mV/s, 180 mA, 0.089 cm²), because it is the same measurement and
a Quick JV result should be comparable with a full sweep of the same cell.

Flipping the architecture **mirrors the sweep window** (p‑i‑n generates in
quadrant 2, so −1.2 V → +0.2 V), asking first if you have typed your own
values. Changing a unit converts the number, so the physical quantity is
preserved — switching Compliance from mA to A cannot silently mean 180 A.

**Where the run reads its device settings from depends on the mode:** Quick JV
selected → the card's own fields; Manual selected → the J–V Parameters tab,
because in Manual the card shows nothing to read. Incident power always comes
from the Analysis tab.

---

## Commissioning a lab PC

The SMTP password is **never** in the source tree. It lives in the operating
system's credential store — Windows Credential Manager via DPAPI, encrypted
under a key derived from that machine and account. The app fails loudly when
it is missing rather than falling back to a built-in value.

```powershell
copy tools\smtp_local_config.example.py tools\smtp_local_config.py
   # fill in SMTP_USER, SMTP_FROM_ADDR, SMTP_PASSWORD
python tools\setup_smtp_credential.py --test
```

`tools/smtp_local_config.py` is in `.gitignore`, so `git add .` cannot pick it
up. The setup script refuses to run if a password constant is added to the
*tracked* script, and writes the non-secret settings (username, host, port,
sender) to `~/.solarjv/email_config.json` — which is how the application finds
the vault entry, since it looks the password up **by username**.

Other flags: `--check` (is one stored?), `--delete`, `--diagnose` (prints
exactly what the application sees, never the password).

**Order matters on a fresh machine.** A new install has an empty `auth.db`
with no users, and registration needs an OTP email — so the credential has to
be in place before the first account can be created.

> Run the script from a normal prompt **as the account that runs the app**.
> DPAPI ties the stored credential to that Windows profile, and a mismatch is
> silent: the app simply reports "not configured".

---

## Testing

The project has a comprehensive test suite: **468 tests across 45 files**, covering the analysis pipeline, SPO and MPP metrics, the report store, auth and e-mail, hardware protocol logic, and the GUI's wiring. Two dedicated *scientific-correctness* suites (`tests/analysis/test_scientific_correctness.py`, `tests/spo/test_spo_scientific_correctness.py`) verify every metric against analytic ground truth derived from the single-diode model — expected values are never transcribed from the implementation's own output.

Three kinds of test are worth knowing about before adding code:

* **Scientific correctness** — ground truth from the single-diode model, never
  from this code's own output.
* **Store contract tests** (`tests/store/`) assert what the publisher must
  *never* do — e.g. that no code path calls `os.remove` / `os.replace` /
  `os.rename` on a path under the store root. They parse the source with
  `ast`, so a regression is caught wherever it is written, not only on a path
  some test happens to execute.
* **Characterisation tests** (`tests/gui/test_spo_card_contract.py` and
  friends) pinned the SPO tab's observable behaviour *before* its Hold Voltage
  card was replaced by the shared widget, so the refactor could be shown to
  preserve behaviour.

GUI tests need an X server. On a headless box: `xvfb-run -a pytest tests/`.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/analysis/test_analysis.py -v

# Run a specific test
pytest tests/analysis/test_analysis.py::test_compute_jv_metrics_ideal_linear_cell -v
```

### Test Architecture

| Tier | Modules | Tests | Description |
|------|---------|-------|-------------|
| **Tier 1 — Pure Logic** | `analysis`, `spo_analysis`, `auth/database`, `config`, `utils/directory_manager` | 44 | No QApplication or hardware needed |
| **Tier 2 — Algorithms** | `jv_procedure`, `spo_report`, `mux_controller`, `app_controller`, `session` | 30 | Core logic with minimal mocking |
| **Tier 3 — UI Widgets** | `analysis_panel`, `parameter_tab`, `file_panel`, `toggle_switch` | 29 | Need QApplication, testable in isolation |
| **Tier 4 — Integration** | End-to-end workflows | 7 | Complex setup, deferred |

### Scientific Rigor

All tests adhere to **first-principles verification**:

- Expected values are **hardcoded** or **independently pre-calculated** from physics equations — never generated by calling the same function under test
- Synthetic I‑V curves satisfy fundamental semiconductor relationships: `Pmax < Voc × Isc`, `0 < FF < 1`
- SPO energy integration verified against hand-calculated values (e.g., constant 10 mW × 100 s = exactly 1.0 J)
- Architecture polarity tests confirm identical magnitude outputs for identical linear relationships in Q2 vs Q4

---

## Building Standalone Applications

### macOS

```bash
pip install briefcase
briefcase create macOS
briefcase build   macOS
briefcase package macOS
# Distribute the resulting .dmg in dist/macOS/
```

### Windows

```powershell
pip install briefcase
briefcase create windows
briefcase build   windows
briefcase package windows
# Distribute the resulting .msi in dist\Windows\
```

> **Note**: Briefcase packaging requires **Python 3.9** (see `.python-version`).

---

## Project Structure

```
jv_setup_keithley_controller/
├── src/solarjv_analyzer/
│   ├── main.py                         # Entry point: login → calibration → main
│   ├── config.py                       # Frozen dataclass singleton (ports, paths)
│   ├── analysis/
│   │   └── analysis.py                 # compute_jv_metrics (11 solar cell metrics)
│   ├── auth/
│   │   ├── database.py                 # SQLite user DB, Argon2id hashing
│   │   ├── email_service.py            # OTP e-mail; password from the OS vault
│   │   ├── otp_widget.py               # One-time-code entry
│   │   ├── session.py                  # SessionManager, per-session log files
│   │   └── login_dialog.py             # Dark-themed split-screen login UI
│   ├── gui/
│   │   ├── jv_analyzer_window.py       # Main window (layout, mode toggle, signals)
│   │   ├── app_controller.py           # Experiment queue, file I/O, SPO lifecycle
│   │   ├── style.py                    # Centralised colour palette, CSS constants
│   │   └── widgets/
│   │       ├── analysis_panel.py       # Full-width matrix table (per-channel metrics)
│   │       ├── parameter_tab.py        # Sweep config, architecture toggle, channels
│   │       ├── instrument_tab.py       # Sense mode, range, NPLC preview
│   │       ├── analysis_settings_tab.py
│   │       ├── file_panel.py           # Filename prefix, directory, validation
│   │       ├── log_panel.py            # Live run log
│   │       ├── vmpp_card.py            # SHARED Hold Voltage card (MPP + SPO)
│   │       ├── unit_field.py           # Number + unit selector, converts on change
│   │       ├── single_channel_selector.py
│   │       ├── combined_tab.py
│   │       ├── toggle_switch.py        # iOS-style animated toggle switch
│   │       └── channel_pinout.py
│   ├── instruments/
│   │   ├── instrument_manager.py       # Keithley + MUX connection lifecycle
│   │   └── mux_controller.py           # Serial hex protocol (AA010{pixel}…)
│   ├── procedures/
│   │   └── jv_procedure.py             # Keithley staircase sweep (PyMeasure)
│   ├── spo/
│   │   ├── spo_procedure.py            # SPO hold test (inherits JVProcedure)
│   │   ├── spo_analysis.py             # compute_spo_metrics (stability, drift, energy)
│   │   ├── spo_report.py               # Crash-safe CSV writer (fsync per row)
│   │   └── spo_widget.py               # SPO configuration + live P-vs-t plot
│   ├── mpp/
│   │   ├── mpp_procedure.py            # Perturb-and-observe tracker
│   │   ├── mpp_analysis.py             # compute_mpp_metrics (drift, flips, clamps)
│   │   ├── mpp_report.py               # Journal per sample + formatted report
│   │   └── mpp_widget.py               # MPP configuration + live P-vs-t plot
│   ├── store/
│   │   ├── paths.py                    # Store root, per-user/date/mode layout
│   │   ├── identity.py                 # Stable report identity
│   │   ├── publisher.py                # published_file(): .part → fsync → rename
│   │   ├── journal.py                  # Crash journals (.journal is promoted)
│   │   ├── finalize.py                 # Promote / discard on reconciliation
│   │   ├── sweeper.py                  # Periodic level-triggered sweep
│   │   └── sidecar.py                  # Per-window watcher attach/detach
│   ├── utils/
│   │   └── directory_manager.py        # Singleton: Base/Username/Date/Mode paths
│   └── windows/
│       └── calibration_window.py       # Mandatory pre-measurement calibration gate
├── tools/
│   ├── setup_smtp_credential.py        # Store the SMTP password in the OS vault
│   ├── smtp_local_config.example.py    # Template — copy, fill in, never committed
│   └── repro_window_crash.py           # Window build/destroy loop (crash hunting)
├── tests/                              # 468 tests, 45 files
│   ├── conftest.py                     # Shared fixtures (synthetic I-V curves)
│   ├── analysis/                       # Metrics + scientific correctness
│   ├── spo/                            # SPO metrics, report, raw-sample discard
│   ├── mpp/                            # Tracking algorithm
│   ├── store/                          # Publisher, sweeper, sidecar, AST contracts
│   ├── auth/                           # User DB + e-mail service
│   ├── gui/                            # Mode switching, wiring, card contracts
│   ├── instruments/ procedures/ utils/ config/ tools/
│   └── test_app_controller.py, test_session.py
├── pyproject.toml                     # Build config (Hatchling), Briefcase settings
├── Login.png                          # 3D lab illustration (~6 MB)
├── channel_pinout.png                 # MUX channel pinout reference
└── README.md
```

---

## Configuration

Edit `src/solarjv_analyzer/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `GPIB_ADDRESS` | `"ASRL3::INSTR"` | VISA resource string for Keithley 2400 |
| `MUX_PORT` | `"COM4"` | Serial port for 6-channel MUX |
| `RESULTS_ROOT` | `~/SolarJV_Data` | Base directory for all measurement data |
| `DATE_FORMAT` | `"%d-%m-%Y"` | Date format for data subdirectories |
| `CHANNEL_COUNT` | `6` | Number of MUX channels |
| `SIMULATION_MODE` | `False` | **Deprecated** — no simulated fallback in production |
| `STORE_ROOT` | `S:\Data\JV` | Protected share where finished reports are published |
| `STAGING_ROOT` | `%LOCALAPPDATA%\SolarJV\staging` | Working files; must allow deletion, because the store does not |
| `STORE_ENABLED` | `True` | Master switch — `SOLARJV_STORE_ENABLED=0` restores pre-store behaviour |
| `STORE_SWEEP_INTERVAL_MS` | `60000` | Safety-net sweep behind the file watcher |
| `STORE_QUIET_SECONDS` | `0.0` | Legacy guess-based completion test; 0 because `published_file()` gives a real completion event |

Two of these are overridable from the environment — `SOLARJV_STORE_ROOT` and
`SOLARJV_STORE_ENABLED` — which is how rehearsals and the test suite avoid
touching the real share.

---

## Data Format

### CSV Block Structure

All exported measurement files use a self-documenting block-tag format:

```csv
[[ EXPERIMENTAL PARAMETERS ]]
Parameter,Value,Unit
Start Voltage,1.2,V
...
Device Architecture,n-i-p,

[[ ANALYSIS SUMMARY ]]
Channel,EFF (%),FF (%),Voc (mV),...
1_Forward,22.57,75.12,680.50,...

[[ MEASUREMENT DATA ]]
channel,1,1,...
direction,Forward,Forward,...
value,V,J,...
,0.0000,-15.0000,...
```

SPO reports use an additional block:

```csv
[[ SPO METRICS ]]
Metric,Value,Unit
mean_power_mw,8.7,mW
drift_percent,1.2,%

[[ TIME SERIES DATA ]]
Time (s),Voltage (V),Current (A),Power (mW)
0.0000,0.6000,-0.01500,9.000
```

---

## Development Guide

> **Working on this code with an AI agent?** The authoritative, machine-facing
> rules live in [`CLAUDE.md`](CLAUDE.md) and [`docs/`](docs/)
> ([architecture](docs/architecture.md) · [metric contracts](docs/metrics.md) ·
> [UI & theme](docs/ui.md) · [testing & debugging](docs/testing.md)), with known
> open defects in [`AUDIT.md`](AUDIT.md). This section is the human summary;
> where the two disagree, `CLAUDE.md` and `docs/` are current.

### Adding a New Feature

1. **Identify the layer**: UI widget → `gui/widgets/`, procedure → `procedures/`, hardware → `instruments/`, analysis → `analysis/`, `spo/` or `mpp/`
2. **Follow existing patterns** — `spo/` and `mpp/` both mirror the JV module's structure (procedure / analysis / report / widget)
3. **Use `DirectoryManager` singleton** for all file paths — never hardcode paths
4. **Add logging** with `logger = logging.getLogger(__name__)` at module level
5. **Thread safety** — hardware calls MUST run in background workers; cross-thread data MUST use Qt signals
6. **Add tests** — pure logic in Tier 1, widgets in Tier 3, hardware-dependent in Tier 2
7. **Before refactoring a widget**, pin its behaviour with characterisation
   tests first, then prove the tests fail against a deliberately broken copy —
   a test that cannot fail is not evidence
8. **Never write to the store directly.** Use `store.finalize.published_file()`
   so the report appears under its final name only once it is closed on disk

### Code Style

- Type hints on all public functions
- Docstrings for all classes and public methods
- Colours and spacing come from `gui/theme/tokens.py` (`gui/style.py` is a legacy re-export shim) — no hardcoded hex values in widget code
- CSV block tags (`[[ EXPERIMENTAL PARAMETERS ]]`, etc.) must not be modified

### Key Architectural Patterns

- **Singleton `DirectoryManager`** — one instance across the entire application
- **SPO and MPP are optional** — both packages are imported behind `try/except ImportError` guards; the JV application runs without either
- **One shared card, two tabs** — MPP and SPO both embed `VmppCard`; it owns
  the Quick JV run, and `device_settings()` decides whether area/compliance
  come from the card or the J–V Parameters tab
- **Secrets live in the OS vault** — no credential is ever read from the source
  tree, and a missing one is reported, not substituted
- **PyMeasure Manager for JV, custom QThread for SPO** — different threading models for different experiment types
- **Hardware-controlled sweeps** — Keithley configured via SCPI trigger model, not software-point-by-point
- **Level-triggered reconciliation** — the file watcher is a latency
  optimisation; the periodic sweep is what makes publishing correct, so no
  dropped OS event can strand a report

---

## License

This project is licensed under the **MIT License**. See [LICENSE.txt](LICENSE.txt) for the full text.

---

<p align="center">
  <em>University of Stuttgart · Institute for Photovoltaics (ipv)</em><br>
  <em>Pfaffenwaldring 47 · 70569 Stuttgart · Germany</em>
</p>
