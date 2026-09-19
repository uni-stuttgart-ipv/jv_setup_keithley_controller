# tests/analysis/test_scientific_correctness.py
"""
First-principles scientific correctness tests for compute_jv_metrics.

DESIGN RULE: every expected value in this file is derived analytically or by
dense numerical evaluation of the *physical model* that generated the data —
never by running compute_jv_metrics and transcribing its output. If these
tests disagree with the implementation, the implementation is wrong.

Physics used:
- Single-diode model: I(V) = -(Iph - I0*(exp(V/(n*Vt)) - 1) - V/Rsh)
  with Vt = kT/q = 0.0257 V at 300 K.
- True MPP: min of P(V) = V*I(V), found on a 400k-point grid (grid error
  < 0.005 mV, negligible vs. the tolerances used).
- True Voc: root of I(V) = 0 via brentq (machine precision).
- True local resistance at V0: R = 1/(dI/dV at V0), analytic derivative.
"""
import numpy as np
import pytest
from pytest import approx
from scipy.optimize import brentq

from solarjv_analyzer.analysis.analysis import compute_jv_metrics

VT = 0.0257  # thermal voltage at ~300 K [V]


def diode_current(v, iph, i0, n, rsh=np.inf):
    """Illuminated single-diode model, instrument sign convention (Q4: I<0)."""
    shunt = v / rsh if np.isfinite(rsh) else 0.0
    return -(iph - i0 * (np.exp(v / (n * VT)) - 1.0) - shunt)


def true_mpp(iph, i0, n, rsh=np.inf, vmax=1.2):
    """Ground-truth (Vmpp, |Pmpp|) by dense sampling of the analytic model."""
    v = np.linspace(0.0, vmax, 400_001)
    p = v * diode_current(v, iph, i0, n, rsh)
    k = int(np.argmin(p))
    return v[k], -p[k]


def true_voc(iph, i0, n, rsh=np.inf):
    """Ground-truth Voc: I(V) = 0 solved to machine precision."""
    return brentq(lambda v: diode_current(v, iph, i0, n, rsh), 0.05, 1.5)


# ----------------------------------------------------------------------------
# MPP accuracy (realistic diode curve, realistic 5 mV sampling)
# ----------------------------------------------------------------------------

class TestMppAccuracy:
    IPH, I0, N = 0.020, 1e-12, 1.5  # 20 mA cell, ideality 1.5 (perovskite-like)

    def test_mpp_accuracy_smooth_diode(self):
        """Pmpp within 0.5% and Vmpp within 15 mV of the analytic MPP.

        Tolerance rationale: the curve is smooth and noiseless and sampled
        every 5 mV; any estimator that respects the local shape of P(V)
        achieves ~0.1% / ~5 mV. A global 5th-order polynomial underfits the
        MPP knee and misses by -1.6% / -34 mV, which is a measurement-grade
        error (larger than typical cert-lab reproducibility of ~1%).
        """
        v = np.linspace(-0.05, 1.0, 211)  # 5 mV steps
        i = diode_current(v, self.IPH, self.I0, self.N)
        vmpp_t, pmpp_t = true_mpp(self.IPH, self.I0, self.N)

        m = compute_jv_metrics(v, i, area_cm2=0.16, incident_power_mw_per_cm2=100.0)

        assert m["Pmpp"] == approx(pmpp_t * 1e3, rel=0.005), \
            f"Pmpp {m['Pmpp']:.4f} mW vs true {pmpp_t*1e3:.4f} mW"
        assert abs(m["Vmpp"] - vmpp_t * 1e3) < 15.0, \
            f"Vmpp {m['Vmpp']:.1f} mV vs true {vmpp_t*1e3:.1f} mV"

    def test_mpp_accuracy_noisy_diode(self):
        """Same cell with 0.2 mA (1% of Isc) Gaussian noise, seeded.

        A local fit should average the noise; tolerance 1.5% / 25 mV.
        """
        rng = np.random.default_rng(3)
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, self.IPH, self.I0, self.N) + rng.normal(0, 2e-4, v.size)
        vmpp_t, pmpp_t = true_mpp(self.IPH, self.I0, self.N)

        m = compute_jv_metrics(v, i, area_cm2=0.16, incident_power_mw_per_cm2=100.0)

        assert m["Pmpp"] == approx(pmpp_t * 1e3, rel=0.015)
        assert abs(m["Vmpp"] - vmpp_t * 1e3) < 25.0

    def test_mpp_point_lies_on_measured_curve(self):
        """Physical consistency: the reported (Vmpp, Jmpp, Pmpp) triple must
        satisfy Pmpp = |Vmpp * Impp| AND Impp must lie on the measured J-V
        curve (within 1%). A clamped or fabricated operating point that is
        not on the curve is not a measurement result.
        """
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, self.IPH, self.I0, self.N)
        area = 0.16
        m = compute_jv_metrics(v, i, area_cm2=area, incident_power_mw_per_cm2=100.0)

        vmpp_v = m["Vmpp"] / 1e3
        impp_reported = m["Jmpp"] * area / 1e3          # mA/cm2 -> A
        impp_on_curve = abs(np.interp(vmpp_v, v, i))
        assert impp_reported == approx(impp_on_curve, rel=0.01)
        assert m["Pmpp"] == approx(abs(vmpp_v * impp_on_curve) * 1e3, rel=0.01)


# ----------------------------------------------------------------------------
# Merged hysteresis loop: all metrics must come from the SAME curve
# ----------------------------------------------------------------------------

class TestHysteresisConsistency:
    def test_merged_loop_metrics_are_self_consistent(self):
        """Forward branch I=-(0.1-0.1V), reverse branch I=-(0.2-0.2V) on the
        same exact voltage grid. The function averages duplicates, giving
        I_avg = -(0.15-0.15V): a linear cell with, analytically,
          Voc = 1.0 V, Isc = 0.15 A, Vmpp = 0.5 V, Pmpp = 37.5 mW,
          FF = Pmpp/(Voc*Isc) = 25.00% exactly (any linear cell has FF=25%).
        Mixing branches (e.g. Isc from the forward branch = 0.10 A with Pmax
        from the averaged curve) yields FF = 37.5% — a physically impossible
        description of any single curve in the dataset.
        """
        v_f = np.arange(0.0, 1.001, 0.01).round(2)
        i_f = -(0.10 - 0.10 * v_f)
        v_r = v_f[::-1]
        i_r = -(0.20 - 0.20 * v_r)

        m = compute_jv_metrics(np.r_[v_f, v_r], np.r_[i_f, i_r],
                               area_cm2=1.0, incident_power_mw_per_cm2=100.0)

        assert m["Isc"] == approx(0.15, rel=0.01), "Isc must be from the averaged curve"
        assert m["Voc"] == approx(1000.0, rel=0.01)
        assert m["Pmpp"] == approx(37.5, rel=0.01)
        assert m["FF"] == approx(25.0, abs=0.5), \
            "FF must be Pmax/(Voc*Isc) of one single curve"


# ----------------------------------------------------------------------------
# Voc accuracy at a diode knee
# ----------------------------------------------------------------------------

class TestVocAccuracy:
    IPH, I0, N = 0.020, 1e-12, 1.5

    def test_voc_noiseless_diode(self):
        """Voc within 2 mV of the analytic root of I(V)=0.

        Tolerance rationale: data is noiseless with 5 mV spacing and the
        crossing is bracketed; interpolation error of any curvature-aware
        estimator is < 1 mV. A straight-line fit across the exponential
        knee is biased by the chord (≈ -6 mV here), which exceeds typical
        instrument voltage accuracy (~1 mV) by design of this test.
        """
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, self.IPH, self.I0, self.N)
        voc_t = true_voc(self.IPH, self.I0, self.N)

        m = compute_jv_metrics(v, i, area_cm2=0.16, incident_power_mw_per_cm2=100.0)
        assert abs(m["Voc"] - voc_t * 1e3) < 2.0, \
            f"Voc {m['Voc']:.2f} mV vs true {voc_t*1e3:.2f} mV"

    def test_voc_noisy_diode(self):
        """Same with 0.5 mA seeded noise; tolerance 4 mV."""
        rng = np.random.default_rng(7)
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, self.IPH, self.I0, self.N) + rng.normal(0, 5e-4, v.size)
        voc_t = true_voc(self.IPH, self.I0, self.N)

        m = compute_jv_metrics(v, i, area_cm2=0.16, incident_power_mw_per_cm2=100.0)
        assert abs(m["Voc"] - voc_t * 1e3) < 4.0


# ----------------------------------------------------------------------------
# Rsh / Rs: local slope must be measured LOCALLY (voltage span, not count)
# ----------------------------------------------------------------------------

class TestResistanceLocality:
    def test_rsh_sharp_diode_coarse_steps(self):
        """n=1 diode with a true 2000-ohm shunt, sampled at 50 mV steps.

        Physics: near V=0 the diode term is negligible (dI/dV_diode ~ 4e-11 S),
        so the local slope IS the shunt: Rsh_true = 2000 ohm to <0.1%.
        A fixed 20-point window at this step size reaches ~0.75 V — deep into
        the exponential knee — and destroys the estimate (reports ~31 ohm,
        64x off). The window must be selected by VOLTAGE SPAN.
        """
        iph, i0, n, rsh = 0.020, 1e-12, 1.0, 2000.0
        v = np.arange(-0.30, 0.901, 0.05).round(3)
        i = diode_current(v, iph, i0, n, rsh)

        m = compute_jv_metrics(v, i, area_cm2=1.0, incident_power_mw_per_cm2=100.0)
        assert m["Rsh"] == approx(2000.0, rel=0.10), \
            f"Rsh {m['Rsh']:.1f} ohm vs true 2000 ohm"

    def test_rs_diode_with_shunt_fine_steps(self):
        """n=1.5 diode + 2000-ohm shunt at 10 mV steps.

        Truth: Rs_local = 1/(dI/dV at Voc) with the analytic derivative
        dI/dV = 1/Rsh + (I0/(n*Vt))*exp(Voc/(n*Vt)). Tolerance 10%: the
        exponential varies ~2x over a +/-25 mV window, so an unweighted local
        fit lands within ~5% of the analytic point value; 10% separates a
        local estimate from a knee-contaminated one.
        """
        iph, i0, n, rsh = 0.020, 1e-12, 1.5, 2000.0
        voc_t = true_voc(iph, i0, n, rsh)
        rs_t = 1.0 / (1.0 / rsh + (i0 / (n * VT)) * np.exp(voc_t / (n * VT)))

        v = np.arange(-0.10, 1.001, 0.01).round(3)
        i = diode_current(v, iph, i0, n, rsh)

        m = compute_jv_metrics(v, i, area_cm2=1.0, incident_power_mw_per_cm2=100.0)
        assert m["Rs"] == approx(rs_t, rel=0.10), \
            f"Rs {m['Rs']:.3f} ohm vs true {rs_t:.3f} ohm"


# ----------------------------------------------------------------------------
# Contact-threshold gate, architecture polarity, and 4-probe geometry metrics
# ----------------------------------------------------------------------------

class TestContactThresholdGate:
    """`contact_threshold` is a minimum |Isc| below which the cell is judged to
    have a failed contact (or be dark), so the performance metrics are
    suppressed to NaN rather than reported as if the cell were healthy.

    Ground truth is analytic: a pure shunt I = -V/R gives Isc = 0 exactly
    (the I(V) line passes through the origin), which is below any positive
    threshold; a 20 mA illuminated diode gives |Isc| = 20 mA >> 1 mA."""

    def test_bad_contact_suppressed(self):
        v = np.linspace(-0.2, 0.2, 81)
        i = -v / 500.0  # pure shunt: Isc = 0

        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0,
                               contact_threshold_a=1e-3)

        assert m["contact_ok"] is False
        for k in ("EFF", "FF", "Voc", "Jsc", "Vmpp", "Jmpp", "Pmpp", "Isc"):
            assert np.isnan(m[k]), f"{k} should be NaN for a bad-contact cell"

    def test_good_contact_passes(self):
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, 0.020, 1e-12, 1.5)  # Isc ~ 20 mA >> 1 mA

        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0,
                               contact_threshold_a=1e-3)

        assert m["contact_ok"] is True
        assert np.isfinite(m["EFF"])

    def test_gate_off_by_default(self):
        # No contact_threshold_a passed -> gate disabled -> contact_ok True.
        v = np.linspace(-0.2, 0.2, 81)
        i = -v / 500.0
        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0)
        assert m["contact_ok"] is True


class TestArchitecturePolarity:
    """`architecture` ("p-i-n" / "n-i-p") declares the device's physical
    polarity: p-i-n -> Q2 (V<0, I>0, negative signed Vmpp), n-i-p -> Q4
    (V>0, I<0, positive signed Vmpp). The backend cross-checks the measured
    signed Vmpp against the declaration and flags a mismatch (e.g. a miswired
    or mislabelled cell) WITHOUT changing any metric — the math stays
    architecture-agnostic per the documented contract."""

    def test_pin_matches_q2(self):
        v_q4 = np.linspace(-0.05, 1.0, 211)
        i_q4 = diode_current(v_q4, 0.020, 1e-12, 1.5)
        v = -v_q4  # mirror to Q2: V<0
        i = -i_q4  # I>0

        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0,
                               architecture="p-i-n")
        assert m["polarity_ok"] is True
        assert m["Vmpp"] < 0  # signed negative for p-i-n / Q2

    def test_pin_mismatch_q4(self):
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, 0.020, 1e-12, 1.5)  # Q4 -> positive Vmpp

        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0,
                               architecture="p-i-n")
        assert m["polarity_ok"] is False

    def test_nip_matches_q4(self):
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, 0.020, 1e-12, 1.5)

        m = compute_jv_metrics(v, i, area_cm2=0.16,
                               incident_power_mw_per_cm2=100.0,
                               architecture="n-i-p")
        assert m["polarity_ok"] is True
        assert m["Vmpp"] > 0


class TestFourProbeGeometry:
    """The 4-probe geometry fields derive two extra outputs from the shunt
    resistance: bulk shunt resistivity rho = Rsh * A / t (through-thickness,
    from R = rho*t/A) and sheet resistance Rsq = rho / t * lateral_factor
    (the "lateral" correction factor scales the in-plane sheet resistance).
    t is the sample thickness; A is the device area.

    Ground truth is analytic: near V=0 the diode term is negligible so the
    measured Rsh equals the model's shunt exactly (see TestResistanceLocality),
    giving rho = 2000 * 0.16 / 0.05 = 6400 ohm.cm and
    Rsq = 6400 / 0.05 = 128000 ohm/sq for the values below."""

    RSH = 2000.0
    AREA = 0.16
    T_UM = 500.0
    T_CM = T_UM * 1e-4  # 0.05 cm

    def test_shunt_resistivity_and_sheet_resistance(self):
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, 0.020, 1e-12, 1.0, rsh=self.RSH)

        m = compute_jv_metrics(v, i, area_cm2=self.AREA,
                               incident_power_mw_per_cm2=100.0,
                               sample_thickness_um=self.T_UM,
                               lateral_factor=1.0)

        rho_t = self.RSH * self.AREA / self.T_CM
        rsq_t = rho_t / self.T_CM
        assert m["Rho_shunt"] == approx(rho_t, rel=0.02), \
            f"Rho_shunt {m['Rho_shunt']:.1f} vs true {rho_t:.1f}"
        assert m["Rsq"] == approx(rsq_t, rel=0.02), \
            f"Rsq {m['Rsq']:.1f} vs true {rsq_t:.1f}"

    def test_lateral_factor_scales_sheet_resistance_only(self):
        v = np.linspace(-0.05, 1.0, 211)
        i = diode_current(v, 0.020, 1e-12, 1.0, rsh=self.RSH)

        m = compute_jv_metrics(v, i, area_cm2=self.AREA,
                               incident_power_mw_per_cm2=100.0,
                               sample_thickness_um=self.T_UM,
                               lateral_factor=2.0)

        rho_t = self.RSH * self.AREA / self.T_CM
        assert m["Rho_shunt"] == approx(rho_t, rel=0.02)   # unaffected
        assert m["Rsq"] == approx(rho_t / self.T_CM * 2.0, rel=0.02)
