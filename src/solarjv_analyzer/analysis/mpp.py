import numpy as np
from typing import List, Dict

def find_mpp(voltage_v: List[float], current_a: List[float], device_area_cm2: float) -> Dict[str, float]:
    """
    Finds the Maximum Power Point (MPP) from J-V data.

    This function calculates the power density for each point on the J-V
    curve and identifies the maximum power point. It returns the corresponding
    voltage, current density, and power density at this point.

    Args:
        voltage_v: A list or array of voltage measurements in Volts.
        current_a: A list or array of corresponding current measurements in Amperes.
        device_area_cm2: The active area of the device in cm^2.

    Returns:
        A dictionary containing the voltage at MPP ('v_mpp' in V), current density
        at MPP ('j_mpp' in mA/cm^2), and the maximum power density
        ('p_mpp' in mW/cm^2).
    """
    if not voltage_v or not current_a or device_area_cm2 <= 0:
        return {'v_mpp': 0.0, 'j_mpp': 0.0, 'p_mpp': 0.0}

    v_arr = np.asarray(voltage_v, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)

    # Convert current (A) to current density (mA/cm^2)
    j_arr_mAcm2 = (i_arr / device_area_cm2) * 1000.0

    # Calculate power density in mW/cm^2 (P = V * J)
    p_arr_mWcm2 = v_arr * j_arr_mAcm2

    # Find the index of the maximum power point
    try:
        if p_arr_mWcm2.size == 0:
            raise ValueError("Power array is empty.")
        mpp_index = np.argmax(p_arr_mWcm2)
    except ValueError:
        return {'v_mpp': 0.0, 'j_mpp': 0.0, 'p_mpp': 0.0}

    # Extract the values at the MPP
    v_mpp = v_arr[mpp_index]
    j_mpp = j_arr_mAcm2[mpp_index]
    p_mpp = p_arr_mWcm2[mpp_index]

    return {
        'v_mpp': float(v_mpp),
        'j_mpp': float(j_mpp),
        'p_mpp': float(p_mpp)
    }