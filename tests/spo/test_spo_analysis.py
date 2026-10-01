"""Tests for spo_analysis.py — compute_spo_metrics()."""

import numpy as np
import pytest
from solarjv_analyzer.spo.spo_analysis import compute_spo_metrics, SPO_METRICS_UNITS


class TestComputeSpoMetrics:
    """Unit tests for SPO stability-metric computation."""

    def test_normal_time_series(self):
        """Steady power output over 10 samples."""
        t = np.linspace(0, 10, 10)
        i = np.full(10, -0.015)
        v = np.full(10, 0.6)
        m = compute_spo_metrics(t, i, v)
        assert m["sample_count"] == 10
        expected_power_mw = -(0.6 * -0.015) * 1000  # +9.0 mW generated power
        assert m["mean_power_mw"] == pytest.approx(expected_power_mw, rel=1e-4)
        assert m["std_power_mw"] == pytest.approx(0.0, abs=1e-3)
        assert abs(m["drift_percent"]) < 1e-3
        assert m["total_energy_j"] > 0

    def test_empty_data_returns_zero_metrics(self):
        m = compute_spo_metrics(np.array([]), np.array([]), np.array([0.6]))
        assert m["sample_count"] == 0
        assert m["mean_power_mw"] == 0.0

    def test_single_sample(self):
        m = compute_spo_metrics(np.array([0.0]), np.array([-0.01]), np.array([0.5]))
        assert m["sample_count"] == 1
        assert m["hold_duration_s"] == 0.0
        assert m["total_energy_j"] == 0.0

    def test_nan_filtering(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        i = np.array([-0.01, np.nan, -0.01, -0.01])
        v = np.array([0.5, 0.5, np.nan, 0.5])
        m = compute_spo_metrics(t, i, v)
        assert m["sample_count"] == 2  # only index 0 and 3 survive

    def test_scalar_voltage_broadcast(self):
        """Single voltage value broadcast across all samples."""
        t = np.linspace(0, 5, 5)
        i = np.array([-0.01, -0.011, -0.0105, -0.01, -0.0095])
        m = compute_spo_metrics(t, i, 0.6)
        assert m["sample_count"] == 5
        assert m["hold_voltage_v"] == pytest.approx(0.6)

    def test_drift_calculation(self):
        """Linearly changing current — drift should be non-zero."""
        t = np.linspace(0, 60, 30)
        i = np.linspace(-0.015, -0.012, 30)  # current magnitude shrinking
        v = np.full(30, 0.6)
        m = compute_spo_metrics(t, i, v)
        # Generated power decays 9 mW -> 7.2 mW → drift is NEGATIVE
        assert m["drift_percent"] < -1.0  # significant degradation
        # max power = best sample (~9 mW) > min power = worst (~7.2 mW)
        assert m["max_power_mw"] > m["min_power_mw"]
        assert m["max_power_mw"] == pytest.approx(9.0, rel=0.01)
        assert m["min_power_mw"] == pytest.approx(7.2, rel=0.01)

    def test_metrics_keys_match_units(self):
        """Every key in metrics output has a corresponding entry in SPO_METRICS_UNITS."""
        expected_keys = {label for label, _ in SPO_METRICS_UNITS}
        t, i, v = np.array([0.0, 1.0]), np.array([-0.01, -0.01]), np.array([0.5, 0.5])
        m = compute_spo_metrics(t, i, v)
        assert set(m.keys()) == expected_keys

    # -------------------------------------------------------------------
    # First-principles physics checks
    # -------------------------------------------------------------------

    def test_energy_integration_exact_constant_power(self):
        """Constant 10 mW over 100 s → exactly 1.0 Joule (P×t)."""
        t = np.linspace(0, 100, 101)
        p_mw = 10.0  # constant 10 mW
        p_w = p_mw / 1000.0
        v = np.full(101, 0.6)
        i = np.full(101, -p_w / 0.6)
        m = compute_spo_metrics(t, i, v)
        assert m["total_energy_j"] == pytest.approx(1.0, rel=0.01)

    def test_drift_formula_first_principles(self):
        """Drift% = (P_final - P_initial) / |P_initial| × 100.
        Uses 5-point edge averages: initial = first 5, final = last 5."""
        t = np.linspace(0, 100, 11)
        v = np.full(11, 0.5)
        # Current magnitude drops linearly from 0.020 → 0.010 A
        i = -np.linspace(0.020, 0.010, 11)
        # Generated power P_gen = -(V*I) drops from 10 mW to 5 mW.
        # First 5 samples avg: P_gen = [10.0, 9.5, 9.0, 8.5, 8.0] → 9.0 mW
        # Last 5 samples avg:  P_gen = [7.0, 6.5, 6.0, 5.5, 5.0] → 6.0 mW
        # Drift = (6.0 - 9.0) / 9.0 × 100 = -33.333% (degrading → negative)
        m = compute_spo_metrics(t, i, v)
        assert m["drift_percent"] == pytest.approx(-33.333, rel=0.05)
        assert m["initial_power_mw"] == pytest.approx(9.0, rel=0.05)
        assert m["final_power_mw"] == pytest.approx(6.0, rel=0.05)

    def test_drift_near_zero_initial_power(self):
        """When initial power is near zero, drift should not divide-by-zero."""
        t = np.array([0.0, 1.0, 2.0])
        v = np.array([0.5, 0.5, 0.5])
        i = np.array([0.0, -0.001, -0.002])
        m = compute_spo_metrics(t, i, v)
        # With 3 samples, edge=min(5,3)=3; both initial and final avg all 3
        # P_gen_avg = -(0.0 + -0.0005 + -0.001)/3 * 1000 = +0.5 mW
        # Drift = (0.5 - 0.5) / 0.5 = 0.0%
        assert m["drift_percent"] == pytest.approx(0.0, abs=1e-6)
        assert m["initial_power_mw"] == pytest.approx(0.5, abs=0.1)
