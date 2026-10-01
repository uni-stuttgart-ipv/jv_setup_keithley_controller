# Metric contracts — read before touching `analysis/` or `spo/`

These are the invariants that survived a physics audit. Each one exists because
breaking it produced wrong science in a real report. The professor-facing
formula spec is README § "Solar Cell Metrics — Calculation & Validation Guide";
keep it in sync with any algorithm change.

## Workflow (non-negotiable)

Any change to metric computation is **test-first**: add or adjust the test in
`tests/analysis/test_scientific_correctness.py` or
`tests/spo/test_spo_scientific_correctness.py` **before** the fix, and confirm
it fails against the old code. Every expected value must be derived
analytically from the generating physical model (single-diode equation, `brentq`
roots, dense-grid optima) — **never** transcribed from the implementation's own
output. "Based on actual test results" fixtures are how three sign/consistency
bugs survived 110 passing tests. The `physics-change` skill has the full
procedure.

Then run `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/analysis tests/spo -q`.

## JV metrics (`analysis/analysis.py`)

`compute_jv_metrics()` is **pure computation** — no hardware dependency, so
`tests/analysis/` runs without instruments.

- **Vmpp is SIGNED** (negative for p-i-n / Q2 devices). SPO uses it directly as
  the hold voltage — taking `abs()` reintroduces a wrong-polarity-hold bug that
  invalidates every p-i-n stability run. All *other* metrics are reported as
  magnitudes.
- **NaN policy.** Voc/Isc are `NaN` when the sweep never crosses I=0 / V=0 (dark
  or truncated sweeps). **Never extrapolate a zero crossing** — a linear fit
  extended past an exponential knee fabricates values (a truncated 914 mV cell
  once reported Voc = 4829 mV). FF/Rs degrade gracefully on NaN (0 / inf).
- **Single-curve consistency.** Data is sorted and exact-duplicate voltages
  averaged *first*; Voc, Isc, Pmax, FF, Rs, Rsh all describe that one averaged
  curve. Never mix branch metrics (e.g. forward-branch Isc with averaged-curve
  Pmax).
- **The (Vmpp, Jmpp, Pmpp) triple is always mutually consistent and lies on the
  measured curve.** Clamps re-evaluate on the curve; they never multiply
  independent magnitudes.
- **MPP algorithm**: the measured global argmin of signed P anchors a local
  parabola; the window widens adaptively until curvature resolves (quad
  residual < 0.7× linear), then re-centres on its vertex (≤3 iterations). Do
  **not** revert to a global high-order polynomial — a quintic on ≥6 points is
  an exact interpolant with no smoothing and systematically underfits sharp MPP
  knees (−1.6 % Pmpp, verified).
- **Voc algorithm**: local fit `I = m·V + c` — I is the dependent variable
  because the noise lives in the measured current — root at `−c/m`, plus a
  quadratic refinement accepted only when it cuts the residual ≥2× (which
  rejects degenerate fits on linear data).
- **Rs/Rsh**: the fit window is selected by **voltage span**
  (±max(2·step, 25 mV)), not point count. A fixed 20-point window at coarse
  steps reaches into the diode knee (a 2000 Ω shunt was reported as 31 Ω).
- `flip_current` defaults to `False`; the function is quadrant-agnostic (Q2 or
  Q4 data both work).
- Output keys are `Rsh`/`Rs` — not the legacy `Rsc`/`Roc`.

## SPO metrics (`spo/spo_analysis.py`)

- **Power convention**: all SPO metrics use *generated* power
  `P_gen = −(V·I)`, positive while the cell delivers, in both Q2 and Q4.
  Therefore degradation ⇒ **negative** `drift_percent`; `max_power_mw` is the
  best sample; `total_energy_j` is the **signed** net integral. Live plots
  display `P_gen`. Reverting to raw `V·I` flips the drift sign in every report.
- Drift from a ~zero baseline is `NaN`; a length mismatch raises; out-of-order
  timestamps are sorted.
