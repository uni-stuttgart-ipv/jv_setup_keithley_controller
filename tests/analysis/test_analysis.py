# tests/analysis/test_analysis.py

import pytest
import numpy as np
from pytest import approx

# Import the functions to be tested from your module
from solarjv_analyzer.analysis.analysis import (
    compute_jv_metrics,
    _llr_voc_from_unsorted,
    _llr_jsc_from_unsorted,
    _interpolate_current_at_voltage_unsorted,
    _interpolate_voltage_at_current_unsorted,
    _calculate_slope_resistance_at_voltage,
)

# Define a default tolerance for float comparisons
DEFAULT_TOLERANCE = approx(0, rel=1e-6, abs=1e-9)


# --- Tests for Helper Functions ---

def test_interpolate_voltage_at_current_unsorted(ideal_linear_cell_data):
    """
    Tests the helper that finds voltage at a target current from sweep-order data.
    """
    v = ideal_linear_cell_data["v_raw"]
    # Use processed current (I_proc = 0.1 - 0.1*V)
    i_proc = 0.1 - 0.1 * v  # Positive current for generation quadrant

    # Test at Voc (I=0)
    voc = _interpolate_voltage_at_current_unsorted(v, i_proc, target_current=0.0)
    assert voc == approx(1.0)
    
    # Test at Isc (V=0 -> I=0.1)
    v_at_isc = _interpolate_voltage_at_current_unsorted(v, i_proc, target_current=0.1)
    assert v_at_isc == approx(0.0)
    
    # Test at Vmax (I=0.05)
    v_at_pmax = _interpolate_voltage_at_current_unsorted(v, i_proc, target_current=0.05)
    assert v_at_pmax == approx(0.5)


def test_interpolate_current_at_voltage_unsorted(ideal_linear_cell_data):
    """
    Tests the helper that finds current at a target voltage from sweep-order data.
    """
    v = ideal_linear_cell_data["v_raw"]
    i_proc = 0.1 - 0.1 * v  # Positive current for generation quadrant

    # Test at Isc (V=0)
    isc = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=0.0)
    assert isc == approx(0.1)
    
    # Test at Voc (V=1.0)
    i_at_voc = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=1.0)
    assert i_at_voc == approx(0.0)

    # Test at Vmax (V=0.5)
    i_at_pmax = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=0.5)
    assert i_at_pmax == approx(0.05)


def test_llr_helpers_different_fit_windows(ideal_linear_cell_data):
    """
    Tests the Local Linear Regression helpers for Voc and Isc with different fit windows.
    On perfect linear data, LLR should be exact regardless of window size.
    """
    v = ideal_linear_cell_data["v_raw"]
    i_proc = 0.1 - 0.1 * v  # Positive current for generation quadrant
    
    # Test various fit window sizes
    for fit_window in [3, 5, 10, 15, 20]:
        # Test LLR Voc
        voc_llr = _llr_voc_from_unsorted(v, i_proc, fit_window=fit_window)
        assert voc_llr == approx(1.0, abs=1e-10), f"Voc failed with fit_window={fit_window}"
        
        # Test LLR Isc
        isc_llr = _llr_jsc_from_unsorted(v, i_proc, fit_window=fit_window)
        assert isc_llr == approx(0.1, abs=1e-10), f"Isc failed with fit_window={fit_window}"


def test_calculate_slope_resistance_at_voltage(ideal_linear_cell_data):
    """
    Tests the dV/dI calculation. For our linear cell, dI/dV = -0.1,
    so dV/dI = 1 / -0.1 = -10.0 ohms.
    """
    v = ideal_linear_cell_data["v_raw"]
    i_proc = 0.1 - 0.1 * v  # Positive current for generation quadrant
    
    # Test Rsh (at V=0.0) - updated from Rsc to Rsh
    rsh = _calculate_slope_resistance_at_voltage(v, i_proc, target_voltage=0.0)
    assert rsh == approx(10.0)
    
    # Test Rs (at V=1.0) - updated from Roc to Rs
    rs = _calculate_slope_resistance_at_voltage(v, i_proc, target_voltage=1.0)
    assert rs == approx(10.0)


# --- Tests for Main Function: compute_jv_metrics ---

def test_compute_jv_metrics_invalid_inputs():
    """
    Tests that the function raises ValueError for invalid parameters.
    """
    v_ok = np.array([0.0, 1.0])
    i_ok = np.array([0.1, 0.0])
    
    # Test non-positive area
    with pytest.raises(ValueError, match="area_cm2 must be > 0"):
        compute_jv_metrics(v_ok, i_ok, area_cm2=0, incident_power_mw_per_cm2=100)
    with pytest.raises(ValueError, match="area_cm2 must be > 0"):
        compute_jv_metrics(v_ok, i_ok, area_cm2=-1, incident_power_mw_per_cm2=100)
        
    # Test non-positive incident power
    with pytest.raises(ValueError, match="incident_power_mw_per_cm2 must be > 0"):
        compute_jv_metrics(v_ok, i_ok, area_cm2=1.0, incident_power_mw_per_cm2=0)
    
    # Test empty arrays
    with pytest.raises(ValueError, match="Input voltage/current arrays must be non-empty"):
        compute_jv_metrics(np.array([]), i_ok, area_cm2=1.0, incident_power_mw_per_cm2=100)
    with pytest.raises(ValueError, match="Input voltage/current arrays must be non-empty"):
        compute_jv_metrics(v_ok, np.array([]), area_cm2=1.0, incident_power_mw_per_cm2=100)


def test_compute_jv_metrics_ideal_linear_cell(ideal_linear_cell_data):
    """
    Performs an integration test using the ideal linear cell data.
    Checks all calculated metrics against their expected values.
    """
    data = ideal_linear_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"], # Pass raw (negative) current
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
        # Use default flip_current=False
    )
    
    expected = data["expected_metrics"]
    
    # Use pytest.approx for all float comparisons
    assert metrics["EFF"] == approx(expected["EFF"])
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["Voc"] == approx(expected["Voc"])
    assert metrics["Jsc"] == approx(expected["Jsc"])
    assert metrics["Vmax"] == approx(expected["Vmax"])
    assert metrics["Jmax"] == approx(expected["Jmax"])
    assert metrics["Isc"] == approx(expected["Isc"])
    assert metrics["Rsh"] == approx(expected["Rsh"])
    assert metrics["Rs"] == approx(expected["Rs"])
    assert metrics["A"] == approx(expected["A"])
    assert metrics["Incd. Pwr"] == approx(expected["Incd. Pwr"])


def test_compute_jv_metrics_q2_data_no_flip(q2_cell_data):
    """
    Tests that the function works on Q2 data (V<0, I>0)
    without needing the flip_current flag, as it is quadrant-agnostic.
    """
    data = q2_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"], # Pass Q2 data (positive current, negative voltage)
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
        flip_current=False, 
    )
    
    # Results should be identical to Q4 case (magnitudes)
    expected = data["expected_metrics"]
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["EFF"] == approx(expected["EFF"])
    assert metrics["Isc"] == approx(expected["Isc"])
    assert metrics["Voc"] == approx(expected["Voc"])


def test_compute_jv_metrics_hysteresis(hysteresis_cell_data):
    """
    Tests the hysteresis fixture data.
    Confirms the code correctly averages forward and reverse sweeps.
    """
    data = hysteresis_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
        # Use default flip_current=False
    )
    
    expected = data["expected_metrics"]
    
    # Check the key metrics from the averaged curve
    assert metrics["Voc"] == approx(expected["Voc"])
    assert metrics["Jsc"] == approx(expected["Jsc"])
    assert metrics["Vmax"] == approx(expected["Vmax"])
    assert metrics["Jmax"] == approx(expected["Jmax"])
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["EFF"] == approx(expected["EFF"])


def test_compute_jv_metrics_noisy_data(noisy_cell_data):
    """
    Tests robustness with noisy data.
    The polynomial fitting for Pmax should handle noise well.
    """
    data = noisy_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # Check that noisy results are within reasonable tolerance
    assert metrics["EFF"] == expected["EFF"]
    assert metrics["FF"] == expected["FF"]
    assert metrics["Voc"] == expected["Voc"]
    assert metrics["Jsc"] == expected["Jsc"]
    assert metrics["Vmax"] == expected["Vmax"]
    assert metrics["Jmax"] == expected["Jmax"]


def test_compute_jv_metrics_dark_curve(dark_curve_data):
    """
    Tests that dark curves (no power generation) are handled correctly.
    Should result in zero efficiency and fill factor.
    """
    data = dark_curve_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # Dark curves should have zero efficiency
    assert metrics["EFF"] == expected["EFF"]
    assert metrics["FF"] == expected["FF"]
    assert metrics["Vmax"] == expected["Vmax"]
    assert metrics["Jmax"] == expected["Jmax"]


def test_compute_jv_metrics_few_points(few_points_data):
    """
    Tests behavior with very few data points.
    Should still produce reasonable results with fallbacks.
    """
    data = few_points_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # With few points, results should be approximate
    assert metrics["EFF"] == expected["EFF"]
    assert metrics["FF"] == expected["FF"]


def test_compute_jv_metrics_vmax_clamp():
    """
    Tests the sanity check: Vmax cannot be greater than Voc.
    Creates a curve where the polynomial fit finds Pmax at V > Voc.
    """
    # Create data where the true maximum power is actually at V > Voc
    # We'll create an artificial "bump" in the power curve after Voc
    
    # Linear region up to Voc
    v_linear = np.linspace(0.0, 1.0, 21)
    i_linear_proc = 0.1 - 0.1 * v_linear  # Isc=0.1, Voc=1.0
    
    # Create artificial power "bump" after Voc
    # This will trick the polynomial fit into thinking Pmax is > Voc
    v_bump = np.linspace(1.01, 1.3, 15)
    
    # Create a current that initially increases (creating negative power more negative)
    # This makes the polynomial fit think there's a better Pmax after Voc
    i_bump_proc = 0.02 + 0.05 * (v_bump - 1.0) - 0.2 * (v_bump - 1.0)**2
    
    v_raw = np.concatenate((v_linear, v_bump))
    i_proc = np.concatenate((i_linear_proc, i_bump_proc))
    i_raw = -i_proc  # Create Q4 (negative) current
    
    # Calculate power to verify our test data creates the scenario we want
    power = v_raw * i_proc
    gen_mask = power < 0
    if np.any(gen_mask):
        v_gen = v_raw[gen_mask]
        power_gen = power[gen_mask]
        # Find where the minimum power would be without clamping
        idx_min_power = int(np.nanargmin(power_gen))
        v_pmax_unclamped = float(v_gen[idx_min_power])
        print(f"DEBUG - Unclamped Vmax would be: {v_pmax_unclamped*1000:.1f} mV")
    
    metrics = compute_jv_metrics(
        v_raw=v_raw,
        i_raw=i_raw,
        area_cm2=1.0,
        incident_power_mw_per_cm2=100.0,
    )
    
    print(f"DEBUG - Voc: {metrics['Voc']} mV, Vmax: {metrics['Vmax']} mV")
    print(f"DEBUG - Should clamp: {abs(metrics['Vmax']) > abs(metrics['Voc'])}")
    
    # The fundamental test: Vmax should never exceed Voc due to clamping
    assert abs(metrics["Vmax"]) <= abs(metrics["Voc"]), "Vmax should be clamped to <= Voc"
    
    # Since we're creating an artificial scenario, we can't predict exact values
    # But we can verify the clamping logic works by checking consistency
    assert metrics["FF"] > 0, "FF should be positive"
    assert metrics["EFF"] > 0, "EFF should be positive"
    
    # Verify that Vmax and Jmax are physically consistent
    # (This tests that Pmax was recalculated after any clamping)
    calculated_pmax = abs(metrics["Vmax"] * metrics["Jmax"] * 1e-3)  # mV * mA/cm² → mW
    reported_pmax = abs(metrics["Voc"] * metrics["Jsc"] * 1e-3 * metrics["FF"] / 100)  # Pmax = Voc*Jsc*FF
    assert calculated_pmax == approx(reported_pmax, rel=0.1), "Pmax should be consistent after clamping"


def test_compute_jv_metrics_parameter_names_updated():
    """
    Tests that the parameter names in output are updated to Rsh and Rs.
    """
    v_raw = np.array([0.0, 0.5, 1.0])
    i_raw = np.array([-0.1, -0.05, 0.0])  # Q4 data
    
    metrics = compute_jv_metrics(
        v_raw=v_raw,
        i_raw=i_raw,
        area_cm2=1.0,
        incident_power_mw_per_cm2=100.0,
    )
    
    # Check that new parameter names exist
    assert "Rsh" in metrics
    assert "Rs" in metrics
    # Check that old parameter names do NOT exist
    assert "Rsc" not in metrics
    assert "Roc" not in metrics

def test_compute_jv_metrics_high_series_resistance(high_rs_cell_data):
    """
    Tests analysis on a cell with high series resistance.
    Validates Rs calculation and Pmax finding on sloped curves.
    """
    data = high_rs_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # Check key metrics
    assert metrics["EFF"] == approx(expected["EFF"])
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["Voc"] == approx(expected["Voc"])
    assert metrics["Jsc"] == approx(expected["Jsc"])
    assert metrics["Vmax"] == approx(expected["Vmax"])
    assert metrics["Jmax"] == approx(expected["Jmax"])
    # Key assertion: Rs should be correctly calculated
    assert metrics["Rs"] == expected["Rs"]


def test_compute_jv_metrics_s_shaped_curve(s_shaped_cell_data):
    """
    Tests the critical advantage of 5th-order polynomial fitting.
    Validates that the algorithm finds the true global Pmax on S-shaped curves,
    not getting stuck in local maxima that would fool simple argmin methods.
    """
    data = s_shaped_cell_data
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # The key test: polyfit should find the global Pmax, not the local one
    # This proves why 5th-order polyfit is superior to simple argmin
    assert metrics["Vmax"] == expected["Vmax"]  # Should be ~900mV, not ~500mV
    assert metrics["EFF"] == expected["EFF"]    # Should be ~79%, not lower
    assert metrics["FF"] == expected["FF"]      # Should be ~72%, not lower
    
    # Additional consistency checks
    assert metrics["Voc"] == expected["Voc"]
    assert metrics["Jsc"] == expected["Jsc"]
    assert metrics["Jmax"] == expected["Jmax"]