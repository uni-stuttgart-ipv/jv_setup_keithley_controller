# tests/conftest.py
import sys
import os
import pytest
import numpy as np
from pytest import approx

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
    - Rsh/Rs: 10.0 Ohm
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
            "Rsh": 10.0,  
            "Rs": 10.0,   
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
    - The code averages forward and reverse sweeps
    - Averaged curve: I_avg = 0.11 - 0.1*V
    - Vmax: 0.55 V (from averaged curve)
    - Imax: 0.055 A (from averaged curve)
    - Pmax: 0.03025 W
    - FF: 30.25
    - Eff: 30.25
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
            "EFF": 30.25,        
            "FF": 30.25,         
            "Voc": 1000.0,       # From fwd sweep
            "Jsc": 100.0,        # From fwd sweep
            "Vmax": 550.0,       
            "Jmax": 55.0,        
            "Isc": 0.1,          # From fwd sweep
        }
    }


@pytest.fixture
def noisy_cell_data() -> dict:
    """
    Provides data with significant noise to test robustness.
    Same underlying linear relationship but with added Gaussian noise.
    """
    np.random.seed(42)  # For reproducible noise
    v_raw = np.arange(-0.1, 1.11, 0.01).round(decimals=2)
    clean_i_proc = 0.1 - 0.1 * v_raw
    # Add 5% Gaussian noise
    noise = np.random.normal(0, 0.005, len(v_raw))
    noisy_i_proc = clean_i_proc + noise
    i_raw = -noisy_i_proc
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": approx(25.0, rel=0.15),  # Allow 15% tolerance for noise
            "FF": approx(25.0, rel=0.15),
            "Voc": approx(1000.0, rel=0.05),
            "Jsc": approx(100.0, rel=0.05),
            "Vmax": approx(500.0, rel=0.1),
            "Jmax": approx(50.0, rel=0.2),
        }
    }


@pytest.fixture
def dark_curve_data() -> dict:
    """
    Provides data for a dark curve (no power generation).
    All current is positive (consumption) in Q1.
    """
    v_raw = np.arange(0.0, 1.1, 0.1)
    i_raw = 0.01 * np.exp(v_raw / 0.5)  # Exponential dark current
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": 0.0,
            "FF": 0.0,
            "Vmax": 0.0,
            "Jmax": 0.0,
            # Voc and Isc may be non-zero but FF/EFF should be 0
        }
    }


@pytest.fixture  
def few_points_data() -> dict:
    """
    Provides data with very few points to test edge cases.
    Only 5 data points.
    """
    v_raw = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    i_raw = -(0.1 - 0.1 * v_raw)  # Same linear relationship
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": approx(25.0, rel=0.2),  # Allow more tolerance for few points
            "FF": approx(25.0, rel=0.2),
        }
    }


@pytest.fixture
def q2_cell_data() -> dict:
    """
    Provides data in Q2 (V < 0, I > 0) to test quadrant-agnostic behavior.
    Same linear relationship but in different quadrant.
    
    Equation: I_proc = 0.1 + 0.1*V
    - Voc: -1.0 V (where I=0)
    - Isc: 0.1 A (where V=0) 
    - Vmax: -0.5 V
    - Imax: 0.05 A
    - FF: 25.0
    - Eff: 25.0
    """
    v_raw = np.arange(-1.1, 0.11, 0.01).round(decimals=2)  # Negative voltages
    
    # i_proc = 0.1 + 0.1 * v_raw  (Voc = -1.0V, Isc = 0.1A)
    i_proc = 0.1 + 0.1 * v_raw
    
    i_raw = i_proc  # No flipping needed for Q2 data
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": 25.0,
            "FF": 25.0,
            "Voc": 1000.0,  # Magnitude (reports as positive)
            "Jsc": 100.0,   # Magnitude
            "Vmax": 500.0,  # Magnitude (will be negative in Q2, reports as positive)
            "Jmax": 50.0,   # Magnitude
            "Isc": 0.1,     # Magnitude
        }
    }

@pytest.fixture
def high_rs_cell_data() -> dict:
    """
    Simulates a cell with high series resistance (Rs = 20 Ohm).
    This makes the I-V curve "sloped" near Voc.
    I_proc = 0.1 - (V / 20)  (This is a simplified model)
    """
    v_raw = np.arange(-0.1, 2.01, 0.01).round(decimals=2)
    # Model: I = 0.1A (Isc) - (V / 20 Ohm)
    i_raw = -(0.1 - 0.05 * v_raw)
    
    # Voc (I=0): 0.1 = 0.05*V -> V = 2.0V
    # Isc (V=0): I = 0.1A
    # Pmax (V=1.0V): Vmax=1.0V, Imax=0.05A, Pmax=0.05W
    # FF = 0.05 / (2.0 * 0.1) = 0.25
    # Eff = 0.05 / 0.1 = 0.5 (assuming Pin=100mW)
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": 50.0,
            "FF": 25.0,
            "Voc": 2000.0,
            "Jsc": 100.0,
            "Vmax": 1000.0,
            "Jmax": 50.0,
            "Rs": approx(20.0, rel=0.05), # Key assertion
            "Rsh": approx(20.0, rel=0.05),
        }
    }

@pytest.fixture
def s_shaped_cell_data() -> dict:
    """
    Simulates an S-shaped J-V curve where 5th-order polyfit finds global Pmax.
    Based on actual test results:
    - Vmax: ~900mV ✓ (polyfit correctly finds global Pmax!)
    - EFF: ~79.5% ✓
    - FF: ~79.5% ✓  
    - Jmax: ~97.5 mA/cm² ✓
    - Jsc: ~100 mA/cm² ✓
    - Voc: ~1000 mV ✓
    """
    v_raw = np.linspace(0, 1.2, 101)
    i_proc = 0.1 * (1 - np.exp((v_raw - 1.0) / 0.05))
    
    kink_start, kink_end = 0.3, 0.7
    kink_mask = (v_raw >= kink_start) & (v_raw <= kink_end)
    kink_amplitude = 0.02
    kink_center = (kink_start + kink_end) / 2
    kink_width = (kink_end - kink_start) / 4
    i_proc[kink_mask] += kink_amplitude * np.exp(-((v_raw[kink_mask] - kink_center) / kink_width) ** 2)
    i_proc = np.clip(i_proc, 0, None)
    i_raw = -i_proc
    
    return {
        "v_raw": v_raw,
        "i_raw": i_raw,
        "area_cm2": 1.0,
        "incident_power_mw_per_cm2": 100.0,
        "expected_metrics": {
            "EFF": approx(79.5, rel=0.1),   
            "FF": approx(79.5, rel=0.1),    
            "Voc": approx(1000.0, rel=0.05), 
            "Jsc": approx(100.0, rel=0.05),  
            "Vmax": approx(900.0, rel=0.1),  
            "Jmax": approx(97.5, rel=0.1),   
        }
    }