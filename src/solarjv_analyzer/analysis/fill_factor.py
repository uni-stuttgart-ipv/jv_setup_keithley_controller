import numpy as np
from typing import Dict

def calculate_fill_factor(
    voc_v: float,
    isc_a: float,
    mpp_results: Dict[str, float],
    device_area_cm2: float
) -> float:
    """
    Calculates the Fill Factor (FF) of a solar cell.

    The Fill Factor is a measure of the squareness of the J-V curve. It is
    defined as the ratio of the maximum power from the solar cell to the
    product of the open-circuit voltage (Voc) and short-circuit current (Isc).

    FF = P_max / (Voc * Isc)

    Args:
        voc_v: The open-circuit voltage in Volts.
        isc_a: The short-circuit current in Amperes.
        mpp_results: A dictionary from the find_mpp function containing the
                     maximum power point value ('p_mpp').
        device_area_cm2: The active area of the device in cm^2, used to
                         convert Isc to current density (Jsc).

    Returns:
        The Fill Factor as a percentage (e.g., 75.0 for 75%). Returns 0.0
        if the calculation cannot be performed.
    """
    if device_area_cm2 <= 0:
        return 0.0

    # Extract the maximum power density (Pmax) from the MPP results
    p_mpp_mWcm2 = mpp_results.get('p_mpp', 0.0)

    # Convert short-circuit current (A) to short-circuit current density (mA/cm^2)
    jsc_mAcm2 = (isc_a / device_area_cm2) * 1000.0

    # Calculate the denominator of the FF equation (the ideal power)
    ideal_power = voc_v * jsc_mAcm2

    # To prevent division by zero, if ideal power is negligible, FF is zero.
    if abs(ideal_power) < 1e-12:
        return 0.0

    # Calculate Fill Factor as a ratio
    ff_ratio = p_mpp_mWcm2 / ideal_power

    # Sanity check to ensure the result is physically plausible
    if not np.isfinite(ff_ratio) or ff_ratio < 0:
        return 0.0

    # Return as a percentage
    return float(ff_ratio * 100.0)