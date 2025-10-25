# tests/conftest.py
import sys
import os
import pytest
import numpy as np

# Add the 'src' directory (parent of 'solarjv_analyzer') to the Python path
# This allows pytest to find the module without it being formally installed.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


@pytest.fixture
def ideal_linear_cell_data() -> dict:
    """
    Provides data for a "perfect" linear cell (resistor) with many points.
    Uses np.arange().round() to ensure 0.0, 0.5, and 1.0 are included.
    
    Metrics are based on the equation: I_proc = 0.1 - 0.1*V
    - Voc: 1.0 V
    - Isc: 0.1 A
    - Vmax: 0.5 V
    - Imax: 0.05 A
    - FF: 25.0
    - Eff: 25.0
    - Rsc/Roc: -10.0 Ohm
    """
    # Create points from -0.1 to 1.1 with a 0.01 step
    v_raw = np.arange(-0.1, 1.11, 0.01).round(decimals=2)
    # i_raw is negative, as 'flip_current' defaults to True
    # i_proc = 0.1 - 0.1*V
    i_raw = -(0.1 - 0.1 * v_raw)
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": 25.0,
            "FF": 25.0,
            "Voc": 1000.0,
            "Jsc": 100.0,
            "Vmax": 500.0,
            "Jmax": 50.0,
            "Isc": 0.1,
            "Rsc": -10.0,
            "Roc": -10.0,
            "A": 1.0,
            "Incd. Pwr": 100.0,
        }
    }


@pytest.fixture
def hysteresis_cell_data() -> dict:
    """
    Provides data with hysteresis.
    Uses np.arange().round() to ensure all key points are included.
    
    Forward Sweep: I_proc = 0.1 - 0.1*V (Linear)
    - Fwd Voc: 1.0 V
    - Fwd Isc: 0.1 A

    Reverse Sweep: I_proc = 0.12 - 0.1*V (Shifted)
    - True Pmax at V=0.6V (I_proc=0.06A) -> Pmax = 0.036 W
    
    Expected Metrics:
    - Voc: 1.0 V (from fwd)
    - Isc: 0.1 A (from fwd)
    - Vmax: 0.6 V (from rev, which is now explicitly included)
    - Imax: 0.06 A (from rev)
    - FF: (0.6 * 0.06) / (1.0 * 0.1) = 0.36
    - Eff: 0.036 W / 0.1 W = 0.36
    """
    # Forward sweep (includes 0.0 and 1.0)
    v_fwd = np.arange(-0.1, 1.11, 0.01).round(decimals=2)
    i_fwd_proc = 0.1 - 0.1 * v_fwd
    i_fwd = -i_fwd_proc
    
    # Reverse sweep (includes 0.6)
    v_rev = np.arange(1.1, -0.11, -0.01).round(decimals=2)
    i_rev_proc = 0.12 - 0.1 * v_rev # Different Isc, Pmax
    i_rev = -i_rev_proc
    
    v_raw_hys = np.concatenate((v_fwd, v_rev))
    i_raw_hys = np.concatenate((i_fwd, i_rev))

    return {
        "v_raw": v_raw_hys,
        "i_raw": i_raw_hys,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": 36.0,
            "FF": 36.0,
            "Voc": 1000.0, # From fwd sweep
            "Jsc": 100.0, # From fwd sweep
            "Vmax": 600.0,  # From rev sweep
            "Jmax": 60.0,   # From rev sweep
            "Isc": 0.1, # From fwd sweep
        }
    }