---
name: physics-change
description: MANDATORY workflow for any change to metric computation in analysis/ or spo/ (Voc, Isc, FF, EFF, MPP, Rs, Rsh, SPO drift/energy). Use when modifying, fixing, or reviewing solar-cell metric math, sweep analysis, or SPO stability metrics.
---

# Physics-First Metric Changes

Any change to `analysis/analysis.py` or `spo/spo_analysis.py` follows this
exact order. Three sign/consistency bugs once survived 110 passing tests
because this order was inverted — do not skip steps.

## 1. Derive ground truth analytically (BEFORE touching code)

Expected values come from the generating physical model, never from running
the implementation:

- **J-V curves**: single-diode model `I(V) = -(Iph - I0*(exp(V/(n*Vt)) - 1) - V/Rsh)`
  with `Vt = 0.0257 V`. True Voc = brentq root of `I(V)=0`. True MPP =
  argmin of `V*I(V)` on a >=100k-point grid. True local resistance =
  `1/(dI/dV)` from the analytic derivative. See
  `tests/analysis/test_scientific_correctness.py` for worked helpers
  (`diode_current`, `true_mpp`, `true_voc`).
- **SPO series**: construct P_gen(t) analytically (e.g. linear decay);
  edge averages, drift, and trapezoid energy computed by hand in the test
  docstring. See `tests/spo/test_spo_scientific_correctness.py`.

## 2. Write the failing test

Add to the scientific-correctness test file. Every tolerance needs a
one-line physical rationale in the docstring. Run it; CONFIRM it fails
against current code for the right reason. If it passes already, your test
is circular — fix the test.

## 3. Implement, then verify numerically

Run the full suite AND a quick numeric probe of the edge cases:
truncated sweep (no I=0 crossing -> Voc must be NaN), Q2 p-i-n mirror
(Vmpp must be NEGATIVE), dead pixel (FF must be 0, never 1e28), merged
fwd+rev loop (all metrics from ONE averaged curve).

## 4. Contracts you must never break

- Vmpp is SIGNED (SPO hold polarity depends on it). Everything else is a magnitude.
- No zero-crossing => NaN. NEVER extrapolate Voc/Isc.
- Compliance-clamped points (|I| >= 0.99*compliance) are NOT measurements —
  suppressed from plot/file/metrics via `JVProcedure.filter_compliance_points`.
- SPO metrics use generated power `P_gen = -(V*I)`; degradation => negative drift.
- Reported (Vmpp, Jmpp, Pmpp) triple must be mutually consistent and on the curve.

## 5. Sync the documentation

Update README.md "Solar Cell Metrics — Calculation & Validation Guide"
(the professor-facing spec) and, if a contract changed, `docs/metrics.md`.
Only touch CLAUDE.md if one of its hard rules actually moved.
