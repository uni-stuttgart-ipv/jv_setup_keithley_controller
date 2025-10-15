import numpy as np
from typing import List, Dict

def _local_dv_di(
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    target_voltage: float
) -> float:
    """
    Calculates the local derivative dV/dI around a target voltage.

    This is done by performing a linear fit on the V-I data within a small
    window around the target voltage. The slope of this fit (V vs I)
    represents the local resistance. This is a helper function.
    """
    # Define a small window around the target voltage
    window = (voltage_v.max() - voltage_v.min()) * 0.02  # Use a 2% window
    if window == 0:
        window = 0.01  # Fallback for flat voltage data

    mask = np.abs(voltage_v - target_voltage) <= window

    # Ensure we have at least 3 points for a meaningful fit.
    # If not, take the 3 points closest to the target voltage.
    if np.sum(mask) < 3:
        closest_indices = np.argsort(np.abs(voltage_v - target_voltage))[:3]
        mask = np.zeros_like(voltage_v, dtype=bool)
        mask[closest_indices] = True

    v_window = voltage_v[mask]
    i_window = current_a[mask]

    if v_window.size < 2:
        return float('inf')

    try:
        # Fit V = m*I + b, where the slope m = dV/dI = Resistance
        slope, _ = np.polyfit(i_window, v_window, 1)
        return float(slope)
    except (np.linalg.LinAlgError, ValueError):
        # This can happen if data in the window is perfectly vertical
        return float('inf')


def calculate_resistances(
    voltage_v: List[float],
    current_a: List[float],
    voc_v: float
) -> Dict[str, float]:
    """
    Calculates series (Rsc) and shunt (Roc) resistances from J-V data.

    - **Series Resistance (Rsc)** is the slope of the J-V curve near the
      open-circuit voltage (V = Voc).
    - **Shunt Resistance (Roc)** is the slope of the J-V curve near the
      short-circuit current (V = 0).

    Args:
        voltage_v: A list or array of voltage measurements in Volts.
        current_a: A list or array of corresponding current measurements in Amperes.
        voc_v: The calculated open-circuit voltage in Volts.

    Returns:
        A dictionary containing the series resistance ('r_sc') and shunt
        resistance ('r_oc') in Ohms.
    """
    if not voltage_v or not current_a:
        return {'r_sc': float('inf'), 'r_oc': float('inf')}

    v_arr = np.asarray(voltage_v, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)

    # Data must be sorted for some of the numpy operations
    sort_indices = np.argsort(v_arr)
    v_sorted = v_arr[sort_indices]
    i_sorted = i_arr[sort_indices]

    # Calculate Series Resistance (Rsc) at V = Voc
    # Use a fallback if voc_v is not a valid number
    target_voc = voc_v if np.isfinite(voc_v) else v_sorted.max()
    r_sc = _local_dv_di(v_sorted, i_sorted, target_voltage=target_voc)

    # Calculate Shunt Resistance (Roc) at V = 0
    r_oc = _local_dv_di(v_sorted, i_sorted, target_voltage=0.0)

    # The conventional names are R_series (or R_s) and R_shunt (or R_sh).
    # The original code used Rsc and Roc, which is a common but sometimes
    # confusing convention. Rsc (short-circuit) is slope at Voc, and
    # Roc (open-circuit) is slope at V=0.
    return {
        'r_sc': r_sc,
        'r_oc': r_oc
    }