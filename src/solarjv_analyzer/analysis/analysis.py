import logging
import numpy as np

logger = logging.getLogger(__name__)

# Defines the display labels and units for the final analysis results.
ANALYSIS_LABELS_UNITS = [
    ("EFF","%"),
    ("FF","%"),
    ("Voc","mV"),
    ("Jsc","mA/cm2"),
    ("Vmax","mV"),
    ("Jmax","mA/cm2"),
    ("Isc","A"),
    ("Rsc","Ohm"),
    ("Roc","Ohm"),
    ("A","cm2"),
    ("Incd. Pwr","mW/cm2"),
]


def compute_jv_metrics(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    area_cm2: float,
    incident_power_mw_per_cm2: float,
    flip_current: bool = True,
    noise_current_a: float = 1e-9,
) -> dict:
    """
    Computes standard solar cell performance metrics from raw J-V data.

    This function serves as the primary entry point for the analysis module. It
    validates inputs, prepares the data, and calls a series of helper functions
    to calculate all relevant metrics (Voc, Isc, FF, Eff, etc.). Voc and Isc are
    calculated from the original sweep order to handle hysteresis, while Pmax
    is calculated from a voltage-sorted array.

    Args:
        v_raw: A NumPy array of raw voltage measurements in Volts.
        i_raw: A NumPy array of raw current measurements in Amperes.
        area_cm2: The active area of the device in cm^2.
        incident_power_mw_per_cm2: The incident optical power in mW/cm^2.
        flip_current: If True, the sign of the current data is inverted to
                      match the convention where photocurrent is positive.
        noise_current_a: A threshold below which current is treated as noise.

    Returns:
        A dictionary containing all calculated performance metrics.

    Raises:
        ValueError: If input data is invalid (e.g., empty lists, non-positive area).
    """
    # Defensive parameter parsing and validation
    area = float(area_cm2) if area_cm2 is not None else None
    if area is None or area <= 0:
        raise ValueError(f"area_cm2 must be > 0 (received {area}).")

    pin_mw_cm2 = (
        float(incident_power_mw_per_cm2) if incident_power_mw_per_cm2 is not None else None
    )
    if pin_mw_cm2 is None or pin_mw_cm2 <= 0:
        raise ValueError(f"incident_power_mw_per_cm2 must be > 0 (received {pin_mw_cm2}).")

    # Convert inputs to numpy arrays
    v_raw = np.asarray(v_raw, dtype=float)
    i_raw = np.asarray(i_raw, dtype=float)
    if v_raw.size == 0 or i_raw.size == 0:
        raise ValueError("Input voltage/current arrays must be non-empty.")

    # Flip current sign to match convention (photocurrent is positive)
    # This must be done on the raw, unsorted data for Voc/Isc calculation
    if flip_current:
        i_proc = -1.0 * i_raw
    else:
        i_proc = i_raw.copy()
    v_proc = v_raw.copy()

    # Try local linear regression (LLR) around the crossing in sweep order; fallback to interpolation helpers
    fit_window = 3
    half_win = max(1, fit_window // 2)

    # Voc: try LLR first
    try:
        voc_llr = _llr_voc_from_unsorted(v_proc, i_proc, fit_window=fit_window)
        if voc_llr is not None:
            voc = float(voc_llr)
            logger.debug("Voc computed via local linear regression")
        else:
            # fallback to sweep-order interpolation helper
            voc = float(_interpolate_voltage_at_current_unsorted(voltage_v=v_proc, current_a=i_proc, target_current=0.0))
            logger.debug("Voc computed via sweep-order interpolation (fallback)")
    except Exception:
        voc = 0.0
        logger.warning("Voc computation failed; setting Voc = 0.0")

    # Isc: try LLR first
    try:
        isc_llr = _llr_jsc_from_unsorted(v_proc, i_proc, fit_window=fit_window)
        if isc_llr is not None:
            isc = float(isc_llr)
            logger.debug("Isc computed via local linear regression")
        else:
            isc = float(_interpolate_current_at_voltage_unsorted(voltage_v=v_proc, current_a=i_proc, target_voltage=0.0))
            logger.debug("Isc computed via sweep-order interpolation (fallback)")
    except Exception:
        isc = 0.0
        logger.warning("Isc computation failed; setting Isc = 0.0")

    if abs(isc) < noise_current_a:
        logger.debug(f"Isc magnitude ({isc:.3e} A) below noise threshold {noise_current_a:.3e} A")

    # For Pmax, a voltage-sorted axis is required for a stable power calculation.
    # Use a STABLE sort to ensure duplicate voltages keep their sweep order
    idx_sort = np.argsort(v_raw, kind='stable')
    v_sorted = v_raw[idx_sort].astype(float)
    i_sorted_raw = i_raw[idx_sort].astype(float)
    
    if flip_current:
        i_sorted_proc = -1.0 * i_sorted_raw
    else:
        i_sorted_proc = i_sorted_raw.copy()

    # Re-remove duplicates on sorted arrays, keeping the last measurement
    if v_sorted.size > 1:
        # This logic robustly finds the *last* index for each unique voltage
        uniq_v_s, first_idx_s = np.unique(v_sorted, return_index=True)
        seen_s = {}
        for idx, vv in enumerate(v_sorted):
            seen_s[vv] = idx
        
        last_indices_s = []
        for vv in uniq_v_s:
            last_indices_s.append(seen_s[vv])
        
        last_indices_s = np.array(last_indices_s, dtype=int)
        
        v_sorted = v_sorted[last_indices_s]
        i_sorted = i_sorted_proc[last_indices_s]
    else:
        i_sorted = i_sorted_proc


    # Compute power and find the maximum power point (Pmax)
    power = v_sorted * i_sorted
    if power.size:
        idx_pmax = int(np.nanargmax(power))
    else:
        idx_pmax = 0
    v_pmax = float(v_sorted[idx_pmax])
    i_pmax = float(i_sorted[idx_pmax])

    # Sanity check: Vmax cannot be physically greater than Voc.
    if voc > 0 and v_pmax > voc:
        logger.debug(f"Vmax {v_pmax:.4f} V exceeds Voc {voc:.4f} V. Clamping Vmax to Voc.")
        v_pmax = voc

    # Sanity check: Imax cannot be physically greater than Isc.
    if abs(i_pmax) > max(abs(isc), 1e-20):
        logger.debug(f"Imax {i_pmax:.6e} A exceeds Isc {isc:.6e} A. Clamping Imax to Isc.")
        i_pmax = isc

    # Calculate current densities
    jsc_mAcm2 = (isc / area) * 1000.0
    jmax_mAcm2 = (i_pmax / area) * 1000.0

    # Calculate Fill Factor
    denom = voc * isc
    ff = 0.0
    if abs(denom) > 0:
        ff = (v_pmax * i_pmax) / denom
    ff_percent = ff * 100.0

    # Calculate resistances using local linear fits
    try:
        rsc = float(_calculate_local_resistance(voltage_v=v_sorted, current_a=i_sorted, target_voltage=0.0))
    except Exception:
        rsc = float("inf")
        logger.warning("Rsc calculation failed; setting to inf.")

    try:
        roc = float(_calculate_local_resistance(voltage_v=v_sorted, current_a=i_sorted, target_voltage=voc))
    except Exception:
        roc = float("inf")
        logger.warning("Roc calculation failed; setting to inf.")

    # Calculate efficiency
    pin_w = float(pin_mw_cm2) * float(area) * 1e-3
    p_pmax_w = float(v_pmax * i_pmax)
    eff_percent = 0.0
    if pin_w > 0:
        eff_percent = (p_pmax_w / pin_w) * 100.0

    # Log warnings for unusual results
    if ff_percent < 0 or ff_percent > 1000:
        logger.warning(f"Unusual FF detected: {ff_percent:.2f}%")

    if eff_percent < 0 or eff_percent > 100:
        logger.warning(f"Unusual Efficiency detected: {eff_percent:.6f}%")

    # Format output dictionary consistent with UI labels/units
    out = {
        "EFF": round(eff_percent, 6) if eff_percent < 1 else round(eff_percent, 2),
        "FF": round(ff_percent, 2),
        "Voc": round(voc * 1e3, 2),
        "Jsc": round(jsc_mAcm2, 6) if abs(jsc_mAcm2) < 0.01 else round(jsc_mAcm2, 3),
        "Vmax": round(v_pmax * 1e3, 2),
        "Jmax": round(jmax_mAcm2, 3),
        "Isc": round(isc, 6),
        "Rsc": round(rsc, 6) if np.isfinite(rsc) else float("inf"),
        "Roc": round(roc, 6) if np.isfinite(roc) else float("inf"),
        "A": float(area),
        "Incd. Pwr": float(pin_mw_cm2),
    }

    return out


# Insert LLR helpers before helpers section

from scipy.stats import linregress

def _llr_voc_from_unsorted(voltage_v: np.ndarray, current_a: np.ndarray, fit_window: int = 20, slope_eps: float = 1e-12):
    """Local linear regression Voc (operates in sweep order).

    Returns Voc (float) on success or None on failure (so caller can fallback).
    """
    n = len(voltage_v)
    if n == 0:
        return None
    half = max(1, fit_window // 2)

    # find first sign change of current (I crossing 0) in sweep order
    y = current_a
    sign = np.sign(y)
    crossings = np.where(sign[:-1] * sign[1:] <= 0)[0]

    if len(crossings) > 0:
        k = int(crossings[0])
        left = max(0, k - half)
        right = min(n, k + 1 + half)
    else:
        # fall back to closest point to I=0
        k = int(np.nanargmin(np.abs(y)))
        left = max(0, k - half)
        right = min(n, k + half + 1)

    v_win = voltage_v[left:right]
    i_win = current_a[left:right]
    if len(v_win) < 2:
        return None

    lr = linregress(v_win, i_win)
    if np.isnan(lr.slope) or abs(lr.slope) < slope_eps:
        return None
    voc = -lr.intercept / lr.slope
    return float(voc)


def _llr_jsc_from_unsorted(voltage_v: np.ndarray, current_a: np.ndarray, fit_window: int = 20):
    """Local linear regression Isc (intercept at V=0) in sweep order.

    Returns Isc (float) on success or None on failure (so caller can fallback).
    """
    n = len(voltage_v)
    if n == 0:
        return None
    half = max(1, fit_window // 2)

    # find first adjacent sign change of voltage around 0 in sweep order
    x = voltage_v
    signx = np.sign(x)
    crossings = np.where(signx[:-1] * signx[1:] <= 0)[0]

    if len(crossings) > 0:
        k = int(crossings[0])
        left = max(0, k - half)
        right = min(n, k + 1 + half)
    else:
        k = int(np.nanargmin(np.abs(x)))
        left = max(0, k - half)
        right = min(n, k + half + 1)

    v_win = x[left:right]
    i_win = current_a[left:right]
    if len(v_win) < 2:
        return None

    lr = linregress(v_win, i_win)
    return float(lr.intercept)

# --------------------------------------------------------------------------
# --- HELPER FUNCTIONS ---
# --------------------------------------------------------------------------

def _interpolate_current_at_voltage_unsorted(
    voltage_v: np.ndarray, current_a: np.ndarray, target_voltage: float
) -> float:
    """
    Interpolates the current at a target voltage from unsorted data.

    This function first attempts to find a pair of adjacent points in the
    original sweep order that bracket the target voltage for direct linear
    interpolation. If no such pair exists (e.g., if the voltage sweep
    skipped over the target), it falls back to a local linear fit using the
    N nearest points.

    Args:
        voltage_v: A NumPy array of voltage measurements in Volts (unsorted).
        current_a: A NumPy array of corresponding current measurements in Amperes.
        target_voltage: The voltage at which to find the corresponding current.

    Returns:
        The interpolated current in Amperes.
    """
    # Attempt to find a bracketing pair in the original sweep order.
    for i in range(len(voltage_v) - 1):
        v_before, v_after = voltage_v[i], voltage_v[i + 1]
        if (v_before <= target_voltage <= v_after) or (v_after <= target_voltage <= v_before):
            i_before, i_after = current_a[i], current_a[i + 1]
            # Avoid division by zero if voltage points are identical.
            if v_after == v_before:
                return float((i_before + i_after) / 2.0)
            # Perform linear interpolation.
            fraction = (target_voltage - v_before) / (v_after - v_before)
            return float(i_before + fraction * (i_after - i_before))

    # Fallback to a local linear fit if no bracketing pair is found.
    num_points_for_fit = min(7, len(voltage_v))
    indices_closest = np.argsort(np.abs(voltage_v - target_voltage))[:num_points_for_fit]
    
    local_voltages = voltage_v[indices_closest]
    local_currents = current_a[indices_closest]

    if len(local_voltages) < 2:
        return float(local_currents[0] if len(local_currents) == 1 else 0.0)

    # Perform a linear least-squares fit (I = m*V + c).
    fit_matrix = np.vstack([local_voltages, np.ones_like(local_voltages)]).T
    slope, intercept = np.linalg.lstsq(fit_matrix, local_currents, rcond=None)[0]
    return float(slope * target_voltage + intercept)


def _interpolate_voltage_at_current_unsorted(
    voltage_v: np.ndarray, current_a: np.ndarray, target_current: float
) -> float:
    """
    Interpolates the voltage at a target current from unsorted data.

    This function finds the first pair of adjacent points in the original sweep
    order that bracket the target current and performs linear interpolation.

    Args:
        voltage_v: A NumPy array of voltage measurements in Volts (unsorted).
        current_a: A NumPy array of corresponding current measurements in Amperes.
        target_current: The current at which to find the corresponding voltage.

    Returns:
        The interpolated voltage in Volts.
    """
    # Calculate the current relative to the target to find zero-crossings.
    current_from_target = current_a - target_current
    
    # Find where the sign of the relative current changes, indicating a crossing.
    sign = np.sign(current_from_target)
    crossing_indices = np.where(sign[:-1] * sign[1:] <= 0)[0]
    
    # If no zero-crossing is found, return the voltage at the point
    # where the current is closest to the target.
    if len(crossing_indices) == 0:
        closest_index = np.nanargmin(np.abs(current_from_target))
        return float(voltage_v[closest_index])
        
    # Use the first crossing found in the sweep order.
    crossing_index = int(crossing_indices[0])
    
    # Get the voltage and current points that bracket the target current.
    v_before = voltage_v[crossing_index]
    v_after = voltage_v[crossing_index + 1]
    i_from_target_before = current_from_target[crossing_index]
    i_from_target_after = current_from_target[crossing_index + 1]
    
    # Avoid division by zero if current points are identical.
    if i_from_target_after == i_from_target_before:
        return float((v_before + v_after) / 2.0)
        
    # Perform linear interpolation to find the voltage at the crossing.
    fraction = -i_from_target_before / (i_from_target_after - i_from_target_before)
    return float(v_before + fraction * (v_after - v_before))


def _interpolate_current_at_voltage_sorted(
    sorted_voltages: np.ndarray, sorted_currents: np.ndarray, target_voltage: float
) -> float:
    """
    Interpolates the current at a target voltage, assuming voltage is sorted.

    Args:
        sorted_voltages: A NumPy array of voltages, sorted ascending.
        sorted_currents: A NumPy array of corresponding currents.
        target_voltage: The voltage at which to find the corresponding current.

    Returns:
        The interpolated current in Amperes.
    """
    # Handle cases where the target is outside the measurement range.
    if target_voltage <= sorted_voltages[0]:
        return float(sorted_currents[0])
    if target_voltage >= sorted_voltages[-1]:
        return float(sorted_currents[-1])
        
    # Use binary search to efficiently find the insertion point for the target voltage.
    index_after_target = np.searchsorted(sorted_voltages, target_voltage)
    
    # Get the voltage and current points that bracket the target.
    v_before = sorted_voltages[index_after_target - 1]
    v_after = sorted_voltages[index_after_target]
    i_before = sorted_currents[index_after_target - 1]
    i_after = sorted_currents[index_after_target]
    
    # Avoid division by zero if voltage points are identical.
    if v_after == v_before:
        return float((i_before + i_after) / 2.0)
        
    # Perform linear interpolation.
    fraction = (target_voltage - v_before) / (v_after - v_before)
    return float(i_before + fraction * (i_after - i_before))


def _calculate_local_resistance(
    voltage_v: np.ndarray, current_a: np.ndarray, target_voltage: float
) -> float:
    """
    Calculates the local resistance (dV/dI) near a target voltage.

    This is achieved by performing a linear fit on the V-I data for a small
    number of points closest to the target voltage. The resistance is the
    inverse of the slope (conductance) of the I-vs-V fit.

    Args:
        voltage_v: A NumPy array of voltage measurements in Volts.
        current_a: A NumPy array of corresponding current measurements in Amperes.
        target_voltage: The voltage point around which to calculate resistance.

    Returns:
        The local resistance in Ohms. Returns infinity if the fit is poor or vertical.
    """
    # Select the N closest points to the target voltage for a stable local fit.
    num_points_for_fit = min(7, len(voltage_v))
    indices_closest = np.argsort(np.abs(voltage_v - target_voltage))[:num_points_for_fit]
    
    local_voltages = voltage_v[indices_closest]
    local_currents = current_a[indices_closest]
    
    # Use a linear least-squares fit for I vs. V (I = m*V + c).
    # The slope 'm' represents the conductance (dI/dV).
    fit_matrix = np.vstack([local_voltages, np.ones_like(local_voltages)]).T
    conductance, _ = np.linalg.lstsq(fit_matrix, local_currents, rcond=None)[0]
    
    # Resistance (dV/dI) is the inverse of conductance.
    # Avoid division by zero for purely horizontal lines.
    if abs(conductance) < 1e-12:
        return float("inf")
        
    return 1.0 / conductance