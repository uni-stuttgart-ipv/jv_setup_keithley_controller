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
    _interpolate_current_at_voltage_sorted,
    _calculate_local_resistance,
)

# Define a default tolerance for float comparisons
# approx() default is rel=1e-6
DEFAULT_TOLERANCE = approx(0, rel=1e-6, abs=1e-9)


# --- Tests for Helper Functions ---

def test_interpolate_voltage_at_current_unsorted(ideal_linear_cell_data):
    """
    Tests the helper that finds voltage at a target current from sweep-order data.
    """
    v = ideal_linear_cell_data["v_raw"]
    # Use processed current (I_proc = 0.1 - 0.1*V)
    i_proc = -1.0 * ideal_linear_cell_data["i_raw"]

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
    i_proc = -1.0 * ideal_linear_cell_data["i_raw"]

    # Test at Isc (V=0)
    isc = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=0.0)
    assert isc == approx(0.1)
    
    # Test at Voc (V=1.0)
    i_at_voc = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=1.0)
    assert i_at_voc == approx(0.0)

    # Test at Vmax (V=0.5)
    i_at_pmax = _interpolate_current_at_voltage_unsorted(v, i_proc, target_voltage=0.5)
    assert i_at_pmax == approx(0.05)


def test_interpolate_current_at_voltage_sorted(ideal_linear_cell_data):
    """
    Tests the helper that finds current at a target voltage from sorted data.
    """
    v_sorted = np.sort(ideal_linear_cell_data["v_raw"])
    i_sorted = -1.0 * ideal_linear_cell_data["i_raw"][np.argsort(ideal_linear_cell_data["v_raw"])]

    # Test at Isc (V=0)
    isc = _interpolate_current_at_voltage_sorted(v_sorted, i_sorted, target_voltage=0.0)
    assert isc == approx(0.1)
    
    # Test at Voc (V=1.0)
    i_at_voc = _interpolate_current_at_voltage_sorted(v_sorted, i_sorted, target_voltage=1.0)
    assert i_at_voc == approx(0.0)
    
    # Test interpolation between points
    i_interp = _interpolate_current_at_voltage_sorted(v_sorted, i_sorted, target_voltage=0.25)
    # Should be halfway between 0.1 (at V=0.0) and 0.05 (at V=0.5)
    assert i_interp == approx(0.075)


def test_llr_helpers(ideal_linear_cell_data):
    """
    Tests the Local Linear Regression helpers for Voc and Isc.
    On perfect linear data, LLR should be exact.
    """
    v = ideal_linear_cell_data["v_raw"]
    i_proc = -1.0 * ideal_linear_cell_data["i_raw"]
    
    # LLR uses a window (default 20), but data is perfectly linear
    
    # Test LLR Voc
    voc_llr = _llr_voc_from_unsorted(v, i_proc, fit_window=20)
    assert voc_llr == approx(1.0)
    
    # Test LLR Isc
    isc_llr = _llr_jsc_from_unsorted(v, i_proc, fit_window=20)
    assert isc_llr == approx(0.1) # Isc is the intercept


def test_calculate_local_resistance(ideal_linear_cell_data):
    """
    Tests the dV/dI calculation. For our linear cell, dI/dV = -0.1,
    so dV/dI = 1 / -0.1 = -10.0 ohms.
    """
    v = ideal_linear_cell_data["v_raw"]
    i_proc = -1.0 * ideal_linear_cell_data["i_raw"]
    
    # The fit uses 7 points
    
    # Test Rsc (at V=0.0)
    rsc = _calculate_local_resistance(v, i_proc, target_voltage=0.0)
    assert rsc == approx(-10.0)
    
    # Test Roc (at V=1.0)
    roc = _calculate_local_resistance(v, i_proc, target_voltage=1.0)
    assert roc == approx(-10.0)


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
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
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
    assert metrics["Rsc"] == approx(expected["Rsc"])
    assert metrics["Roc"] == approx(expected["Roc"])
    assert metrics["A"] == approx(expected["A"])
    assert metrics["Incd. Pwr"] == approx(expected["Incd. Pwr"])


def test_compute_jv_metrics_no_flip(ideal_linear_cell_data):
    """
    Tests the 'flip_current=False' flag by passing in already-processed current.
    The results should be identical to the standard test.
    """
    data = ideal_linear_cell_data
    # Pass processed current (i_proc) instead of i_raw
    i_proc = -1.0 * data["i_raw"]
    
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=i_proc, # Pass processed current
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
        flip_current=False, # Tell the function not to flip it
    )
    
    expected = data["expected_metrics"]
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["EFF"] == approx(expected["EFF"])
    assert metrics["Isc"] == approx(expected["Isc"])


def test_compute_jv_metrics_hysteresis(hysteresis_cell_data):
    """
    Tests the hysteresis fixture data.
    Confirms Voc/Isc are from the fwd sweep (first crossing) and
    Vmax/Jmax are from the rev sweep (last-point-wins de-duplication).
    """
    data = hysteresis_cell_data
    metrics = compute_jv_metrics(
        v_raw=data["v_raw"],
        i_raw=data["i_raw"],
        area_cm2=data["area_cm2"],
        incident_power_mw_per_cm2=data["incident_power_mw_per_cm2"],
    )
    
    expected = data["expected_metrics"]
    
    # Check the key metrics that differ due to hysteresis logic
    assert metrics["Voc"] == approx(expected["Voc"])
    assert metrics["Jsc"] == approx(expected["Jsc"])
    assert metrics["Vmax"] == approx(expected["Vmax"])
    assert metrics["Jmax"] == approx(expected["Jmax"])
    assert metrics["FF"] == approx(expected["FF"])
    assert metrics["EFF"] == approx(expected["EFF"])


def test_compute_jv_metrics_vmax_clamp():
    """
    Tests the sanity check: Vmax cannot be greater than Voc.
    Creates a curve where Pmax occurs at V > Voc.
    
    We provide > 20 points, with the data around Voc (V=1.0) being
    perfectly linear to ensure the LLR calculation is correct.
    """
    # Create 11 perfectly linear points from V=0.0 to V=1.0
    v_linear = np.linspace(0.0, 1.0, 11)
    i_linear_proc = 0.1 - 0.1 * v_linear # Isc=0.1, Voc=1.0
    
    # Create 10 non-linear points *after* Voc
    v_nonlinear = np.linspace(1.01, 1.1, 10)
    # Create a shape where power *increases* after Voc
    # i_proc = 0.05 at V=1.1
    i_nonlinear_proc = (v_nonlinear - 1.0) * 0.5 
    
    v_raw = np.concatenate((v_linear, v_nonlinear))
    i_proc = np.concatenate((i_linear_proc, i_nonlinear_proc))
    i_raw = -i_proc
    
    # In this data:
    # - Voc = 1.0V (from the linear part, which LLR will find)
    # - Pmax will be at V=1.1, where I_proc=0.05, P=0.055
    
    metrics = compute_jv_metrics(
        v_raw=v_raw,
        i_raw=i_raw,
        area_cm2=1.0,
        incident_power_mw_per_cm2=100.0,
    )
    
    # Voc should be 1.0V (1000 mV) from the linear fit
    assert metrics["Voc"] == approx(1000.0)
    
    # Vmax would be 1.1V (1100 mV) but should be clamped to Voc (1000 mV)
    assert metrics["Vmax"] == approx(1000.0) 
    
    # Jmax is 50.0 (from 0.05A at V=1.1)
    assert metrics["Jmax"] == approx(50.0)
    
    # FF = (Vmax_clamped * Imax) / (Voc * Isc)
    # Isc = 0.1
    # FF = (1.0 * 0.05) / (1.0 * 0.1) = 0.5
    assert metrics["FF"] == approx(50.0)