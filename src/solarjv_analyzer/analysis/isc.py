import numpy as np
from typing import List

def calculate_isc(voltage_v: List[float], current_a: List[float]) -> float:
    """
    Calculates the short-circuit current (Isc) from J-V data.

    Isc is the current at which the net voltage is zero. This function finds
    this point by linearly interpolating the J-V curve for the highest
    accuracy. A fallback to the nearest measured point is included for robustness.

    Args:
        voltage_v: A list or array of voltage measurements in Volts.
        current_a: A list or array of corresponding current measurements in Amperes.

    Returns:
        The calculated short-circuit current (Isc) in Amperes.
    """
    if not voltage_v or not current_a:
        return 0.0

    v_arr = np.asarray(voltage_v, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)

    # For interpolation, data must be sorted by the x-coordinate (voltage).
    # This was done in the original combined analysis function.
    sort_indices = np.argsort(v_arr)
    v_sorted = v_arr[sort_indices]
    i_sorted = i_arr[sort_indices]

    try:
        # Isc is the current (y-value) where voltage (x-value) is 0.0 V.
        # np.interp is the direct and efficient way to perform this interpolation.
        isc = float(np.interp(0.0, v_sorted, i_sorted))
    except Exception:
        # Fallback for any unexpected errors, returning the current at the
        # voltage point closest to zero.
        isc = float(i_sorted[np.argmin(np.abs(v_sorted))])

    return isc