import numpy as np
from typing import List, Dict

# Import all the individual calculation modules from this package
from .voc import calculate_voc
from .isc import calculate_isc
from .mpp import find_mpp
from .fill_factor import calculate_fill_factor
from .efficiency import calculate_efficiency
from .resistances import calculate_resistances

def run_full_analysis(
    voltage_v: List[float],
    current_a: List[float],
    device_area_cm2: float,
    incident_power_mw_per_cm2: float,
    flip_current: bool = True
) -> Dict[str, float]:
    """
    Performs a full analysis of solar cell J-V data.

    This function is the main entry point for the analysis package. It calls
    all the individual calculation modules in the correct order, manages the
    dependencies between them (e.g., Voc is needed for Fill Factor), and
    assembles the final results into a single, formatted dictionary.

    Args:
        voltage_v: A list of voltage measurements in Volts.
        current_a: A list of corresponding current measurements in Amperes.
        device_area_cm2: The active area of the device in cm^2.
        incident_power_mw_per_cm2: The incident optical power in mW/cm^2.
        flip_current: If True, the sign of the current data is inverted to
                      match the standard convention where photocurrent is positive.

    Returns:
        A dictionary containing all standard solar cell performance metrics,
        with keys matching those expected by the GUI.
    """
    if not voltage_v or not current_a or device_area_cm2 <= 0:
        return {}  # Return empty dict if inputs are invalid

    v_arr = np.asarray(voltage_v, dtype=float)
    i_arr = np.asarray(current_a, dtype=float)

    # 1. Handle current flipping, a key step from the original logic
    if flip_current:
        i_arr = -i_arr

    # 2. Sort data once for all subsequent calculations
    sort_indices = np.argsort(v_arr)
    v_sorted = v_arr[sort_indices]
    i_sorted = i_arr[sort_indices]

    # 3. Perform individual calculations in order of dependency
    voc = calculate_voc(v_sorted.tolist(), i_sorted.tolist())
    isc = calculate_isc(v_sorted.tolist(), i_sorted.tolist())
    mpp_results = find_mpp(v_sorted.tolist(), i_sorted.tolist(), device_area_cm2)
    ff = calculate_fill_factor(voc, isc, mpp_results, device_area_cm2)
    eff = calculate_efficiency(mpp_results, incident_power_mw_per_cm2)
    res_results = calculate_resistances(v_sorted.tolist(), i_sorted.tolist(), voc)

    # 4. Assemble the final results dictionary with correct units and keys
    jsc_mAcm2 = (isc / device_area_cm2) * 1000.0

    final_metrics = {
        "EFF - Efficency": eff,
        "FF- fill factor": ff,
        "Voc - open circuit volatge": voc * 1000.0,  # V to mV
        "Jsc - short circ. current density": jsc_mAcm2,
        "Vmax": mpp_results.get('v_mpp', 0.0) * 1000.0, # V to mV
        "Jmax": mpp_results.get('j_mpp', 0.0),
        "Isc - short circ. current": isc,
        "Rsc - short circ. resistence": res_results.get('r_sc', float('inf')),
        'Roc  open ""': res_results.get('r_oc', float('inf')),
        "A - Area": device_area_cm2,
        "Incd. Pwr": incident_power_mw_per_cm2,
    }

    return final_metrics