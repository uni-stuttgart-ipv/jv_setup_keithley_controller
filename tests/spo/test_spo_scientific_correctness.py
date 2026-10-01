# tests/spo/test_spo_scientific_correctness.py
"""
First-principles correctness tests for compute_spo_metrics.

DESIGN RULE: expected values are derived from the physics of the constructed
time series, never from the implementation's own output.

Sign convention under test (physics): during a stability hold the quantity of
interest is the GENERATED power P_gen = -(V*I), which is positive whenever
the cell delivers power (Q4: V>0, I<0 — and Q2: V<0, I>0). A cell whose
output decays must show NEGATIVE drift; "max power" must be the best sample;
energy must be the net generated energy (consumption intervals subtract).
"""
import numpy as np
import pytest

from solarjv_analyzer.spo.spo_analysis import compute_spo_metrics


class TestSpoSignConvention:
    def _degrading_cell(self):
        """V=0.6 V hold, |I| decays linearly 25 mA -> 16.667 mA over 100 s:
        generated power decays 15 mW -> 10 mW. 101 samples.

        Edge averages (first/last 5 samples, exact):
          P_gen(t) = 15 - 0.05*t  [mW]
          initial = mean(P_gen at t=0..4)   = 15 - 0.05*2  = 14.90 mW
          final   = mean(P_gen at t=96..100)= 15 - 0.05*98 = 10.10 mW
          drift   = (10.10 - 14.90)/14.90 * 100 = -32.2148...%
        """
        t = np.linspace(0.0, 100.0, 101)
        p_gen_mw = 15.0 - 0.05 * t
        v = np.full_like(t, 0.6)
        i = -(p_gen_mw / 1000.0) / 0.6  # Q4: negative current
        return t, i, v

    def test_degrading_cell_reports_negative_drift(self):
        t, i, v = self._degrading_cell()
        m = compute_spo_metrics(t, i, v)
        initial_true = 15.0 - 0.05 * np.mean([0, 1, 2, 3, 4])
        final_true = 15.0 - 0.05 * np.mean([96, 97, 98, 99, 100])
        drift_true = (final_true - initial_true) / abs(initial_true) * 100.0

        assert m["drift_percent"] == pytest.approx(drift_true, rel=0.01)
        assert m["drift_percent"] < 0, "a degrading cell MUST report negative drift"
        assert m["initial_power_mw"] == pytest.approx(initial_true, rel=0.01)
        assert m["final_power_mw"] == pytest.approx(final_true, rel=0.01)

    def test_generated_power_is_positive_for_healthy_hold(self):
        t, i, v = self._degrading_cell()
        m = compute_spo_metrics(t, i, v)
        # mean of P_gen(t) = 15 - 0.05t over [0,100] = 12.5 mW
        assert m["mean_power_mw"] == pytest.approx(12.5, rel=0.01)
        assert m["mean_power_mw"] > 0

    def test_max_min_power_semantics(self):
        """max_power must be the BEST sample (15 mW), min_power the WORST (10)."""
        t, i, v = self._degrading_cell()
        m = compute_spo_metrics(t, i, v)
        assert m["max_power_mw"] == pytest.approx(15.0, rel=0.01)
        assert m["min_power_mw"] == pytest.approx(10.0, rel=0.01)
        assert m["max_power_mw"] > m["min_power_mw"]

    def test_q2_hold_identical_magnitudes(self):
        """Same cell measured in Q2 (V<0, I>0) must give identical positive
        power metrics — generated power is quadrant-independent."""
        t, i, v = self._degrading_cell()
        m4 = compute_spo_metrics(t, i, v)
        m2 = compute_spo_metrics(t, -i, -v)
        for key in ("mean_power_mw", "drift_percent", "max_power_mw",
                    "min_power_mw", "total_energy_j"):
            assert m2[key] == pytest.approx(m4[key], rel=1e-6), key


class TestSpoEnergy:
    def test_energy_is_net_generated_energy(self):
        """Piecewise series: generate +10 mW for 50 s, then consume 10 mW
        (wrong-polarity interval) for the next 50 s ramping through zero.
        Net energy by trapezoid on the SIGNED generated power:
          segment 1 (0..50 s, 10 -> 10 mW):   0.5 J
          segment 2 (50..100 s, 10 -> -10 mW): 0.0 J
          net = 0.5 J.  (An |P| integration would report 1.0 J.)
        """
        t = np.array([0.0, 50.0, 100.0])
        p_gen_mw = np.array([10.0, 10.0, -10.0])
        v = np.full_like(t, 0.6)
        i = -(p_gen_mw / 1000.0) / 0.6
        m = compute_spo_metrics(t, i, v)
        assert m["total_energy_j"] == pytest.approx(0.5, rel=0.01)

    def test_energy_constant_power_exact(self):
        """10 mW x 100 s = 1.0 J exactly."""
        t = np.linspace(0, 100, 101)
        v = np.full_like(t, 0.6)
        i = np.full_like(t, -(0.010 / 0.6))
        m = compute_spo_metrics(t, i, v)
        assert m["total_energy_j"] == pytest.approx(1.0, rel=0.001)


class TestSpoInputValidation:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_spo_metrics(np.array([0.0, 1.0, 2.0]),
                                np.array([-0.01, -0.01]),
                                np.array([0.6, 0.6, 0.6]))

    def test_unsorted_time_is_handled(self):
        """Samples delivered out of order must not corrupt the integral."""
        t = np.linspace(0, 100, 101)
        v = np.full_like(t, 0.6)
        i = np.full_like(t, -(0.010 / 0.6))
        rng = np.random.default_rng(1)
        order = rng.permutation(t.size)
        m = compute_spo_metrics(t[order], i[order], v[order])
        assert m["total_energy_j"] == pytest.approx(1.0, rel=0.001)

    def test_zero_initial_power_gives_nan_drift(self):
        """Drift from a truly-zero baseline is undefined — NaN, not a number."""
        t = np.linspace(0, 10, 11)
        v = np.full_like(t, 0.6)
        i = np.zeros_like(t)
        i[6:] = -0.001  # power appears only late in the run
        m = compute_spo_metrics(t, i, v)
        assert np.isnan(m["drift_percent"])
