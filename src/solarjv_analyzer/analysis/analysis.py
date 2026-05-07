import logging
import numpy as np
from scipy.stats import linregress
from typing import Dict

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
    ("Rsh","Ohm"),  
    ("Rs","Ohm"),   
    ("A","cm2"),
    ("Incd. Pwr","mW/cm2"),
]


def compute_jv_metrics(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    area_cm2: float,
    incident_power_mw_per_cm2: float,
    flip_current: bool = False,
    noise_current_a: float = 1e-9,
) -> dict:
    """
    Computes standard solar cell performance metrics from raw J-V data.
    This script is now robust and will analyze the true power-generating
    quadrant (Q2 or Q4) regardless of wiring polarity.
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

    # Convert inputs
    v_raw = np.asarray(v_raw, dtype=float)
    i_raw = np.asarray(i_raw, dtype=float)

    if v_raw.size == 0 or i_raw.size == 0:
        raise ValueError("Input voltage/current arrays must be non-empty.")

    # Remove any NaN values
    valid_mask = ~np.isnan(v_raw) & ~np.isnan(i_raw)
    v_raw, i_raw = v_raw[valid_mask], i_raw[valid_mask]
    
    # Check again in case NaN filtering made them empty
    if v_raw.size == 0 or i_raw.size == 0:
        raise ValueError("Input voltage/current arrays are all-NaN.")

    # Data quality checks
    if len(v_raw) < 10:
        logger.warning("Very few data points - results may be unreliable")
    
    v_range = np.max(v_raw) - np.min(v_raw)
    if v_range < 0.1:  # Less than 100mV range
        logger.warning("Small voltage range - check instrument settings")

    # Flip current sign ONLY if user explicitly requests it
    if flip_current:
        i_proc = -1.0 * i_raw
    else:
        i_proc = i_raw.copy()
    v_proc = v_raw.copy()

    # Verify we're in a power-generating quadrant
    v_mean, i_mean = np.mean(v_proc), np.mean(i_proc)
    if not ((v_mean < 0 and i_mean > 0) or (v_mean > 0 and i_mean < 0)):
        logger.warning("Data may not be in power-generating quadrant (Q2 or Q4)")

    # Use adaptive fit window for better robustness
    fit_window = min(15, max(5, len(v_proc) // 4))

    # Voc: LLR first, interpolation as backup
    try:
        voc_llr = _llr_voc_from_unsorted(v_proc, i_proc, fit_window=fit_window)
        if voc_llr is not None:
            voc = float(voc_llr)
        else:
            # Fallback to interpolation if LLR returns None
            voc = _interpolate_voltage_at_current_unsorted(v_proc, i_proc, 0.0)
    except Exception:
        # Final fallback if both methods fail
        voc = 0.0
        logger.warning("Voc computation failed; setting Voc = 0.0")

    # Isc: LLR first, interpolation as backup
    try:
        isc_llr = _llr_jsc_from_unsorted(v_proc, i_proc, fit_window=fit_window)
        if isc_llr is not None:
            isc = float(isc_llr)
        else:
            # Fallback to interpolation if LLR returns None
            isc = _interpolate_current_at_voltage_unsorted(v_proc, i_proc, 0.0)
    except Exception:
        # Final fallback if both methods fail
        isc = 0.0
        logger.warning("Isc computation failed; setting Isc = 0.0")

    if abs(isc) < noise_current_a:
        logger.debug(f"Isc magnitude ({isc:.3e} A) below noise threshold {noise_current_a:.3e} A")

    # Sort data for Pmax calculation
    idx_sort = np.argsort(v_proc, kind='stable')
    v_sorted = v_proc[idx_sort]
    i_sorted = i_proc[idx_sort]

    # Re-remove duplicates using np.unique to handle hysteresis safely
    if v_sorted.size > 1:
        uniq_v, idx, counts = np.unique(v_sorted, return_index=True, return_counts=True)
        
        # Only average where there are duplicates
        if np.any(counts > 1):
            uniq_i = np.zeros_like(uniq_v)
            # Use np.add.at for a fast, vectorized sum of currents at each unique voltage index
            np.add.at(uniq_i, np.searchsorted(uniq_v, v_sorted), i_sorted)
            uniq_i /= counts  # Compute the average current
            
            v_sorted = uniq_v
            i_sorted = uniq_i

    power = v_sorted * i_sorted
    
    # Filter for only power-generating quadrants (where P < 0)
    gen_mask = power < 0
    if np.any(gen_mask):
        v_gen = v_sorted[gen_mask]
        i_gen = i_sorted[gen_mask]
        power_gen = power[gen_mask]
        
        # Robust Pmax calculation using polynomial fitting
        if len(v_gen) > 5:  # Need enough points for a stable fit
            try:
                # Fit a 5th-order polynomial to P(V)
                p_fit_coeffs = np.polyfit(v_gen, power_gen, 5)
                p_fit_func = np.poly1d(p_fit_coeffs)
                
                # Find the derivative of the fit: dP/dV
                p_deriv_func = p_fit_func.deriv()
                
                # Find the roots (where dP/dV = 0)
                roots = p_deriv_func.r
                
                # Filter for real roots within the voltage range
                real_roots = roots[np.isreal(roots)].real
                valid_roots = real_roots[
                    (real_roots >= np.min(v_gen)) & (real_roots <= np.max(v_gen))
                ]
                
                if len(valid_roots) > 0:
                    # Evaluate the power fit at these roots
                    power_at_roots = p_fit_func(valid_roots)
                    
                    # The Pmax is the minimum power among these roots
                    best_root_idx = np.nanargmin(power_at_roots)
                    v_pmax = float(valid_roots[best_root_idx])
                    p_max_w = float(power_at_roots[best_root_idx])
                    
                    # Interpolate Imax from the original data at the new Vmax
                    i_pmax = float(np.interp(v_pmax, v_sorted, i_sorted))
                else:
                    # Fallback to argmin if fit fails or finds no roots
                    logger.debug("Polyfit for Pmax failed, falling back to argmin.")
                    idx_pmax = int(np.nanargmin(power_gen))
                    v_pmax = float(v_gen[idx_pmax])
                    i_pmax = float(i_gen[idx_pmax])
                    p_max_w = float(power_gen[idx_pmax])
            except (np.linalg.LinAlgError, ValueError):
                 # Fallback if polyfit fails
                logger.warning("Polyfit for Pmax failed, falling back to argmin.")
                idx_pmax = int(np.nanargmin(power_gen))
                v_pmax = float(v_gen[idx_pmax])
                i_pmax = float(i_gen[idx_pmax])
                p_max_w = float(power_gen[idx_pmax])
        else:
            # Fallback for very few data points
            idx_pmax = int(np.nanargmin(power_gen))
            v_pmax = float(v_gen[idx_pmax])
            i_pmax = float(i_gen[idx_pmax])
            p_max_w = float(power_gen[idx_pmax])
    else:
        # No power generation found (e.g., a dark curve)
        logger.warning("No power generation (P < 0) found. Data may be a dark curve.")
        v_pmax, i_pmax, p_max_w = 0.0, 0.0, 0.0

    # FIXED: Proper clamping with Pmax recalculation
    vmax_clamped = v_pmax
    imax_clamped = i_pmax

    # Sanity check: Vmax magnitude cannot be > Voc magnitude
    if abs(v_pmax) > abs(voc):
        logger.debug(f"|Vmax| {abs(v_pmax):.4f} V exceeds |Voc| {abs(voc):.4f} V. Clamping Vmax to Voc.")
        vmax_clamped = np.sign(v_pmax) * abs(voc)

    # Sanity check: Imax magnitude cannot be > Isc magnitude
    if abs(i_pmax) > abs(isc):
        logger.debug(f"|Imax| {abs(i_pmax):.6e} A exceeds |Isc| {abs(isc):.6e} A. Clamping Imax to Isc.")
        imax_clamped = np.sign(i_pmax) * abs(isc)

    # RECALCULATE Pmax using the (potentially) clamped values
    # This ensures Pmax, Vmax, and Imax are always consistent.
    p_max_w = float(vmax_clamped * imax_clamped)
    v_pmax = float(vmax_clamped)
    i_pmax = float(imax_clamped)

    # Calculate current densities (report as positive magnitude)
    jsc_mAcm2 = abs(isc / area) * 1000.0
    jmax_mAcm2 = abs(i_pmax / area) * 1000.0

    # Calculate Fill Factor
    denom = voc * isc
    ff = 0.0
    if abs(denom) > 0:
        ff = abs(p_max_w) / abs(denom)  # Use absolute values for positive FF
    ff_percent = ff * 100.0

    # Calculate resistances (ensure positive values)
    try:
        rsh = float(_calculate_slope_resistance_at_voltage(voltage_v=v_sorted, current_a=i_sorted, target_voltage=0.0))
        rsh = abs(rsh)  # Ensure positive resistance
    except Exception:
        rsh = float("inf")
        logger.warning("Rsh calculation failed; setting to inf.")

    try:
        rs = float(_calculate_slope_resistance_at_voltage(voltage_v=v_sorted, current_a=i_sorted, target_voltage=voc))
        rs = abs(rs)  # Ensure positive resistance
    except Exception:
        rs = float("inf")
        logger.warning("Rs calculation failed; setting to inf.")

    # Calculate efficiency
    pin_w = float(pin_mw_cm2) * float(area) * 1e-3
    p_pmax_w_positive = abs(p_max_w) # Use positive magnitude of Pmax
    eff_percent = 0.0
    if pin_w > 0:
        eff_percent = (p_pmax_w_positive / pin_w) * 100.0

    # Log warnings for unusual results
    if ff_percent < 0 or ff_percent > 100:
        logger.warning(f"Unusual FF detected: {ff_percent:.2f}%")
    if eff_percent < 0 or eff_percent > 100:
        logger.warning(f"Unusual Efficiency detected: {eff_percent:.6f}%")

    # Format output dictionary (report magnitudes)
    out = {
        "EFF": round(eff_percent, 2),
        "FF": round(ff_percent, 2),
        "Voc": round(abs(voc) * 1e3, 2), # Report Voc magnitude
        "Jsc": round(jsc_mAcm2, 3),
        "Vmax": round(abs(v_pmax) * 1e3, 2), # Report Vmax magnitude
        "Jmax": round(jmax_mAcm2, 3),
        "Isc": round(abs(isc), 6), # Report Isc magnitude
        "Rsh": round(rsh, 6) if np.isfinite(rsh) else float("inf"),  # Changed from Rsc to Rsh
        "Rs": round(rs, 6) if np.isfinite(rs) else float("inf"),     # Changed from Roc to Rs
        "A": float(area),
        "Incd. Pwr": float(pin_mw_cm2),
    }

    return out


def _llr_voc_from_unsorted(voltage_v: np.ndarray, current_a: np.ndarray, fit_window: int = 15):
    """
    Local linear regression Voc (operates in sweep order).
    Finds V where I=0.
    """
    n = len(voltage_v)
    if n < 2: return None
    half = max(1, fit_window // 2)

    y = current_a
    valid_mask = ~np.isnan(voltage_v) & ~np.isnan(y)
    voltage_v, y = voltage_v[valid_mask], y[valid_mask]
    n_valid = len(y)
    if n_valid < 2: return None
    
    sign = np.sign(y)
    crossings = np.where(sign[:-1] * sign[1:] <= 0)[0]

    if len(crossings) > 0:
        k = int(crossings[0])
        left = max(0, k - half)
        right = min(n_valid, k + 1 + half)
    else:
        logger.warning("No I=0 crossing found. Voc may be out of range. Extrapolating from closest point.")
        k = int(np.nanargmin(np.abs(y)))
        left = max(0, k - half)
        right = min(n_valid, k + half + 1)

    v_win = voltage_v[left:right]
    i_win = y[left:right]
    if len(v_win) < 2: return None

    # Fit V = m*I + c. Voc is the intercept (V) when I=0.
    try:
        lr = linregress(i_win, v_win) 
    except (ValueError, TypeError):
        logger.warning("Linregress failed for Voc.")
        return None
        
    if np.isnan(lr.slope) or np.isnan(lr.intercept):
        return None
        
    return float(lr.intercept)


def _llr_jsc_from_unsorted(voltage_v: np.ndarray, current_a: np.ndarray, fit_window: int = 15):
    """
    Local linear regression Isc (intercept at V=0) in sweep order.
    Finds I where V=0.
    """
    n = len(voltage_v)
    if n < 2: return None
    half = max(1, fit_window // 2)

    x = voltage_v
    valid_mask = ~np.isnan(x) & ~np.isnan(current_a)
    x, current_a = x[valid_mask], current_a[valid_mask]
    n_valid = len(x)
    if n_valid < 2: return None

    signx = np.sign(x)
    crossings = np.where(signx[:-1] * signx[1:] <= 0)[0]

    if len(crossings) > 0:
        k = int(crossings[0])
        left = max(0, k - half)
        right = min(n_valid, k + 1 + half)
    else:
        logger.warning("No V=0 crossing found. Extrapolating Isc from closest point.")
        k = int(np.nanargmin(np.abs(x)))
        left = max(0, k - half)
        right = min(n_valid, k + half + 1)

    v_win = x[left:right]
    i_win = current_a[left:right]
    if len(v_win) < 2: return None

    # Fit I = m*V + c. Isc is the intercept (I) when V=0.
    try:
        lr = linregress(v_win, i_win)
    except (ValueError, TypeError):
        logger.warning("Linregress failed for Isc.")
        return None
        
    if np.isnan(lr.intercept):
        return None
        
    return float(lr.intercept)


def _calculate_slope_resistance_at_voltage(
    voltage_v: np.ndarray, current_a: np.ndarray, target_voltage: float
) -> float:
    """
    Calculates local resistance as |dV/dI| at a target voltage.
    Returns positive resistance value.
    """
    valid_mask = ~np.isnan(voltage_v) & ~np.isnan(current_a)
    voltage_v, current_a = voltage_v[valid_mask], current_a[valid_mask]
    if len(voltage_v) < 2: return float("inf")

    # Use a robust 20-point fit window
    num_points_for_fit = min(20, len(voltage_v))
    indices_closest = np.argsort(np.abs(voltage_v - target_voltage))[:num_points_for_fit]
    
    local_voltages = voltage_v[indices_closest]
    local_currents = current_a[indices_closest]
    if len(local_voltages) < 2: return float("inf")
    
    fit_matrix = np.vstack([local_voltages, np.ones_like(local_voltages)]).T
    try:
        conductance, _ = np.linalg.lstsq(fit_matrix, local_currents, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("inf")
    
    if abs(conductance) < 1e-12:
        return float("inf")
    return abs(1.0 / conductance)

# --------------------------------------------------------------------------
# --- HELPER FUNCTIONS ---
# --------------------------------------------------------------------------

def _interpolate_current_at_voltage_unsorted(
    voltage_v: np.ndarray, current_a: np.ndarray, target_voltage: float
) -> float:
    """Fallback interpolation for Isc."""
    valid_mask = ~np.isnan(voltage_v) & ~np.isnan(current_a)
    voltage_v, current_a = voltage_v[valid_mask], current_a[valid_mask]
    if len(voltage_v) == 0: return 0.0

    for i in range(len(voltage_v) - 1):
        v_before, v_after = voltage_v[i], voltage_v[i + 1]
        if (v_before <= target_voltage <= v_after) or (v_after <= target_voltage <= v_before):
            i_before, i_after = current_a[i], current_a[i + 1]
            if v_after == v_before: return float((i_before + i_after) / 2.0)
            fraction = (target_voltage - v_before) / (v_after - v_before)
            return float(i_before + fraction * (i_after - i_before))

    num_points_for_fit = min(7, len(voltage_v))
    indices_closest = np.argsort(np.abs(voltage_v - target_voltage))[:num_points_for_fit]
    local_voltages = voltage_v[indices_closest]
    local_currents = current_a[indices_closest]
    if len(local_voltages) < 2:
        return float(local_currents[0] if len(local_currents) == 1 else 0.0)

    fit_matrix = np.vstack([local_voltages, np.ones_like(local_voltages)]).T
    try:
        slope, intercept = np.linalg.lstsq(fit_matrix, local_currents, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    return float(slope * target_voltage + intercept)


def _interpolate_voltage_at_current_unsorted(
    voltage_v: np.ndarray, current_a: np.ndarray, target_current: float
) -> float:
    """Fallback interpolation for Voc."""
    valid_mask = ~np.isnan(voltage_v) & ~np.isnan(current_a)
    voltage_v, current_a = voltage_v[valid_mask], current_a[valid_mask]
    if len(voltage_v) == 0: return 0.0
    
    current_from_target = current_a - target_current
    sign = np.sign(current_from_target)
    crossing_indices = np.where(sign[:-1] * sign[1:] <= 0)[0]
    
    if len(crossing_indices) == 0:
        closest_index = np.nanargmin(np.abs(current_from_target))
        return float(voltage_v[closest_index])
        
    crossing_index = int(crossing_indices[0])
    v_before = voltage_v[crossing_index]
    v_after = voltage_v[crossing_index + 1]
    i_from_target_before = current_from_target[crossing_index]
    i_from_target_after = current_from_target[crossing_index + 1]
    
    if i_from_target_after == i_from_target_before:
        return float((v_before + v_after) / 2.0)
        
    fraction = -i_from_target_before / (i_from_target_after - i_from_target_before)
    return float(v_before + fraction * (v_after - v_before))