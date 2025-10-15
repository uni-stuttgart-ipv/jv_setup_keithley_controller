import numpy as np
from typing import List

def calculate_voc(voltage_v: List[float], current_a: List[float]) -> float:
    """
    Calculates the open-circuit voltage (Voc) from J-V data.

    Voc is the voltage at which the net current is zero. This function finds
    this point by linearly interpolating the J-V curve for the highest
    accuracy. A fallback to the nearest measured point is included for robustness.

    Args:
        voltage_v: A list or array of voltage measurements in Volts.
        current_a: A list or array of corresponding current measurements in Amperes.

    Returns:
        The calculated open-circuit voltage (Voc) in Volts.
    """
    if not voltage_v or not current_a:
        return 0.0

    v_arr = np.asarray(voltage_v, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)

    # Ensure data is sorted by voltage for correct interpolation, as was done
    # in the original combined analysis function.
    sort_indices = np.argsort(v_arr)
    v_sorted = v_arr[sort_indices]
    i_sorted = i_arr[sort_indices]
    
    try:
        # Find the indices where the current crosses zero
        sign_change_indices = np.where(np.diff(np.sign(i_sorted)))[0]

        if sign_change_indices.size > 0:
            # Take the first crossing
            idx = sign_change_indices[0]
            
            # Get the two points that bracket the zero-crossing
            v1, v2 = v_sorted[idx], v_sorted[idx + 1]
            i1, i2 = i_sorted[idx], i_sorted[idx + 1]

            # Linearly interpolate to find the voltage (x) at current=0 (y)
            return float(np.interp(0.0, [i1, i2], [v1, v2]))
        else:
            # Fallback: if no zero-crossing is found, return the voltage
            # at the point of minimum absolute current.
            return float(v_sorted[np.argmin(np.abs(i_sorted))])
            
    except Exception:
        # Generic fallback for any unexpected errors
        return float(v_sorted[np.argmin(np.abs(i_sorted))])