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
    ("Vmpp","mV"),
    ("Jmpp","mA/cm2"),
    ("Pmpp","mW"),
    ("Isc","A"),
    ("Rsh","Ohm"),
    ("Rs","Ohm"),
    ("Rho_shunt","Ohm.cm"),
    ("Rsq","Ohm/sq"),
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
    architecture: str = None,
    contact_threshold_a: float = None,
    sample_thickness_um: float = None,
    lateral_factor: float = 1.0,
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

    # Sort data and average exact-duplicate voltages FIRST (merged forward +
    # reverse loops). All metrics below — Voc, Isc, Pmax, FF, Rs, Rsh — are
    # then computed from this SAME averaged curve. Mixing branches (e.g. Voc
    # from the forward branch with Pmax from the averaged curve) produces a
    # fill factor that describes no physical curve in the dataset.
    idx_sort = np.argsort(v_proc, kind='stable')
    v_sorted = v_proc[idx_sort]
    i_sorted = i_proc[idx_sort]

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

    # Voc: LLR first, interpolation as backup.
    # If the curve never crosses I=0 (dark or truncated sweep), refuse to
    # extrapolate — a linear fit extended beyond an exponential diode curve
    # fabricates a value. Report NaN so downstream consumers see "unknown".
    _i_sign = np.sign(i_sorted)
    _has_i_crossing = bool(np.any(_i_sign[:-1] * _i_sign[1:] <= 0))
    if _has_i_crossing:
        try:
            voc_llr = _llr_voc_from_unsorted(v_sorted, i_sorted, fit_window=fit_window)
            if voc_llr is not None:
                voc = float(voc_llr)
            else:
                # Fallback to interpolation if LLR returns None
                voc = _interpolate_voltage_at_current_unsorted(v_sorted, i_sorted, 0.0)
        except Exception:
            voc = float("nan")
            logger.warning("Voc computation failed; reporting Voc = NaN")
    else:
        voc = float("nan")
        logger.warning(
            "No I=0 crossing in sweep — Voc is outside the measured range. "
            "Reporting NaN instead of extrapolating a fabricated value."
        )

    # Isc: LLR first, interpolation as backup. Same no-extrapolation rule:
    # if the sweep never crosses V=0, Isc cannot be measured — report NaN.
    _v_sign = np.sign(v_sorted)
    _has_v_crossing = bool(np.any(_v_sign[:-1] * _v_sign[1:] <= 0))
    if _has_v_crossing:
        try:
            isc_llr = _llr_jsc_from_unsorted(v_sorted, i_sorted, fit_window=fit_window)
            if isc_llr is not None:
                isc = float(isc_llr)
            else:
                # Fallback to interpolation if LLR returns None
                isc = _interpolate_current_at_voltage_unsorted(v_sorted, i_sorted, 0.0)
        except Exception:
            isc = float("nan")
            logger.warning("Isc computation failed; reporting Isc = NaN")
    else:
        isc = float("nan")
        logger.warning(
            "No V=0 crossing in sweep — Isc is outside the measured range. "
            "Reporting NaN instead of extrapolating a fabricated value."
        )

    if abs(isc) < noise_current_a:
        logger.debug(f"Isc magnitude ({isc:.3e} A) below noise threshold {noise_current_a:.3e} A")

    power = v_sorted * i_sorted
    
    # Filter for only power-generating quadrants (where P < 0)
    gen_mask = power < 0
    if np.any(gen_mask):
        v_gen = v_sorted[gen_mask]
        i_gen = i_sorted[gen_mask]
        power_gen = power[gen_mask]
        
        # Robust Pmax: anchor at the measured global maximum-|P| point
        # (argmin of signed power — global by definition), then refine with a
        # LOCAL parabola fitted only to points near the peak. A local
        # quadratic follows the curvature at the MPP; a global high-order
        # polynomial systematically underfits sharp MPP knees.
        idx_pmax = int(np.nanargmin(power_gen))
        v_pmax = float(v_gen[idx_pmax])
        i_pmax = float(i_gen[idx_pmax])
        p_max_w = float(power_gen[idx_pmax])

        if len(v_gen) >= 3:
            try:
                step = float(np.median(np.abs(np.diff(v_gen)))) if len(v_gen) > 1 else 0.0
                full_span = float(v_gen.max() - v_gen.min())

                # Choose the smallest window around the anchor in which the
                # curvature is actually RESOLVED (quadratic residual clearly
                # below linear residual). Small windows track sharp MPP knees
                # without underfit; under noise, a too-small window has no
                # curvature signal, so widen until it does. On a globally
                # quadratic P(V) widening is unbiased.
                span = max(6.0 * step, 0.05)
                while True:
                    win = np.abs(v_gen - v_pmax) <= span
                    n_win = int(np.count_nonzero(win))
                    if n_win >= 6:
                        vw, pw = v_gen[win], power_gen[win]
                        lin = np.polyfit(vw, pw, 1)
                        rss_lin = float(np.sum((pw - np.polyval(lin, vw)) ** 2))
                        quad = np.polyfit(vw, pw, 2)
                        rss_quad = float(np.sum((pw - np.polyval(quad, vw)) ** 2))
                        if quad[0] > 0 and rss_quad < 0.7 * rss_lin:
                            break  # curvature resolved at this span
                    elif n_win >= 3:
                        break  # sparse data: fit what we have
                    if span >= full_span:
                        break
                    span = min(2.0 * span, full_span)

                # Iterative recentering: fit around the center, move the
                # center to the fitted vertex, refit. Under noise the raw
                # argmin anchor can sit off the true peak on flat maxima.
                center = v_pmax
                quad_best = None
                for _ in range(3):
                    win = np.abs(v_gen - center) <= span
                    if np.count_nonzero(win) < 3:
                        break
                    quad = np.polyfit(v_gen[win], power_gen[win], 2)
                    if quad[0] <= 0:  # not concave-up => no local minimum
                        break
                    v_vertex = -quad[1] / (2.0 * quad[0])
                    if not np.isfinite(v_vertex) or abs(v_vertex - center) > 2.0 * span:
                        break  # diverging fit — keep the measured point
                    v_vertex = float(np.clip(v_vertex, v_gen.min(), v_gen.max()))
                    quad_best = quad
                    if abs(v_vertex - center) < 0.25 * step + 1e-12:
                        center = v_vertex
                        break  # converged
                    center = v_vertex

                if quad_best is not None:
                    v_pmax = float(center)
                    p_max_w = float(np.polyval(quad_best, v_pmax))
                    # Derive Impp from the smoothed Pmax so the
                    # (Vmpp, Impp, Pmpp) triple is exactly consistent.
                    i_pmax = float(p_max_w / v_pmax) if v_pmax != 0 else 0.0
            except (np.linalg.LinAlgError, ValueError):
                logger.debug("Local parabola refinement failed; using measured argmin point.")
    else:
        # No power generation found (e.g., a dark curve)
        logger.warning("No power generation (P < 0) found. Data may be a dark curve.")
        v_pmax, i_pmax, p_max_w = 0.0, 0.0, 0.0

    # Sanity clamp: |Vmpp| cannot exceed |Voc|. Applied ONLY when violated,
    # and the operating point is re-evaluated ON the measured curve so the
    # reported (Vmpp, Impp, Pmpp) is a real point of the J-V characteristic —
    # never the product of two independently clamped magnitudes (which can
    # fabricate FF = 100%). Skipped when Voc is NaN (unknown).
    if np.isfinite(voc) and abs(v_pmax) > abs(voc):
        logger.debug(f"|Vmpp| {abs(v_pmax):.4f} V exceeds |Voc| {abs(voc):.4f} V. Clamping Vmpp to Voc.")
        v_pmax = float(np.sign(v_pmax) * abs(voc))
        i_pmax = float(np.interp(v_pmax, v_sorted, i_sorted))
        p_max_w = float(v_pmax * i_pmax)

    # |Impp| exceeding |Isc| indicates an Isc fit inconsistency, not a bad
    # MPP: warn, but do not fabricate an off-curve operating point.
    if np.isfinite(isc) and abs(i_pmax) > abs(isc):
        logger.warning(
            f"|Impp| {abs(i_pmax):.6e} A exceeds fitted |Isc| {abs(isc):.6e} A — "
            "check the Isc fit region; metrics left unclamped."
        )

    # Calculate current densities (report as positive magnitude)
    jsc_mAcm2 = abs(isc / area) * 1000.0
    jmax_mAcm2 = abs(i_pmax / area) * 1000.0

    # Calculate Fill Factor.
    # Guard against near-zero denominators (dead pixels): |Isc| must exceed
    # the noise floor and |Voc| must be non-trivial, otherwise FF explodes
    # to absurd values. NaN Voc/Isc also fail these checks (NaN > x is False).
    denom = voc * isc
    ff = 0.0
    if abs(isc) > noise_current_a and abs(voc) > 1e-6:
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
        if not np.isfinite(voc):
            raise ValueError("Voc is NaN — cannot locate the Rs fit window at V=Voc")
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

    # --- Contact-threshold gate ---
    # A cell whose |Isc| falls below `contact_threshold_a` has either a failed
    # contact or is dark: its performance metrics are meaningless, so they are
    # suppressed to NaN ("unknown") rather than reported as if healthy.
    contact_ok = True
    if contact_threshold_a is not None and contact_threshold_a > 0:
        if not np.isfinite(isc) or abs(isc) < contact_threshold_a:
            contact_ok = False
            logger.warning(
                f"|Isc| {abs(isc):.3e} A is below the contact threshold "
                f"{contact_threshold_a:.3e} A — treating the measurement as a "
                "failed contact / dark cell (performance metrics set to NaN)."
            )

    # --- Architecture polarity check ---
    # p-i-n devices generate in Q2 (negative signed Vmpp); n-i-p in Q4
    # (positive signed Vmpp). A mismatch means the cell is miswired or
    # mislabelled. This only FLAGS — the metric math stays architecture-agnostic.
    polarity_ok = True
    if architecture in ("p-i-n", "n-i-p") and np.isfinite(v_pmax) and v_pmax != 0:
        if architecture == "p-i-n" and v_pmax > 0:
            polarity_ok = False
            logger.warning(
                f"Measured Vmpp is positive ({v_pmax*1e3:.1f} mV, Q4) but the "
                "declared architecture is p-i-n (expects Q2 / negative Vmpp) — "
                "check device wiring or architecture label."
            )
        elif architecture == "n-i-p" and v_pmax < 0:
            polarity_ok = False
            logger.warning(
                f"Measured Vmpp is negative ({v_pmax*1e3:.1f} mV, Q2) but the "
                "declared architecture is n-i-p (expects Q4 / positive Vmpp) — "
                "check device wiring or architecture label."
            )

    # --- 4-probe geometry: shunt resistivity & sheet resistance ---
    # Bulk shunt resistivity rho = Rsh * A / t (from R = rho * t / A).
    # Sheet resistance Rsq = rho / t * lateral_factor, where the lateral
    # factor corrects the in-plane sheet resistance for finite lateral extent.
    rho = float("inf")
    rsq = float("inf")
    if sample_thickness_um is not None and sample_thickness_um > 0 and np.isfinite(rsh):
        t_cm = float(sample_thickness_um) * 1e-4  # um -> cm
        rho = float(rsh) * float(area) / t_cm
        rsq = rho / t_cm * float(lateral_factor)

    # Format output dictionary (report magnitudes)
    out = {
        "EFF": round(eff_percent, 2),
        "FF": round(ff_percent, 2),
        "Voc": round(abs(voc) * 1e3, 2), # Report Voc magnitude
        "Jsc": round(jsc_mAcm2, 3),
        "Vmpp": round(v_pmax * 1e3, 2), # Signed: negative for Q2 (p-i-n), positive for Q4 (n-i-p) — SPO hold polarity depends on this
        "Jmpp": round(jmax_mAcm2, 3),
        "Pmpp": round(abs(p_max_w) * 1e3, 3),  # Report Pmpp in mW
        "Isc": round(abs(isc), 6), # Report Isc magnitude
        "Rsh": round(rsh, 6) if np.isfinite(rsh) else float("inf"),  # Changed from Rsc to Rsh
        "Rs": round(rs, 6) if np.isfinite(rs) else float("inf"),     # Changed from Roc to Rs
        "Rho_shunt": round(rho, 6) if np.isfinite(rho) else float("inf"),
        "Rsq": round(rsq, 6) if np.isfinite(rsq) else float("inf"),
        "A": float(area),
        "Incd. Pwr": float(pin_mw_cm2),
        "contact_ok": contact_ok,
        "polarity_ok": polarity_ok,
    }

    # A failed contact invalidates every derived metric — report them as NaN
    # ("unknown") while preserving the geometry/illumination inputs.
    if not contact_ok:
        for key in ("EFF", "FF", "Voc", "Jsc", "Vmpp", "Jmpp", "Pmpp", "Isc",
                    "Rsh", "Rs", "Rho_shunt", "Rsq"):
            out[key] = float("nan")

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
        # No I=0 crossing: extrapolating a local linear fit beyond the data
        # fabricates Voc. Refuse and let the caller decide (reports NaN).
        logger.warning("No I=0 crossing found — Voc out of sweep range; returning None.")
        return None

    v_win = voltage_v[left:right]
    i_win = y[left:right]
    if len(v_win) < 2: return None

    # Fit I(V) — measurement noise lives in the CURRENT (the instrument
    # sources V, measures I), so I must be the dependent variable; fitting
    # V(I) inverts the error model and dilutes the slope under noise.
    # Voc is the root of the fit at I = 0.
    try:
        lr = linregress(v_win, i_win)
    except (ValueError, TypeError):
        logger.warning("Linregress failed for Voc.")
        return None

    if np.isnan(lr.slope) or np.isnan(lr.intercept) or lr.slope == 0:
        return None

    voc_linear = float(-lr.intercept / lr.slope)

    # Curvature refinement: a straight line fitted across the exponential
    # diode knee is biased by the chord (several mV). A local quadratic
    # captures the curvature. Accept it ONLY if it meaningfully reduces the
    # fit residual — on genuinely linear data the quadratic is numerically
    # degenerate (leading coefficient ~ 0, ill-conditioned roots) and the
    # linear root is the correct answer.
    if len(v_win) >= 5:
        try:
            resid_lin = float(np.sum((i_win - (lr.slope * v_win + lr.intercept)) ** 2))
            quad = np.polyfit(v_win, i_win, 2)
            resid_quad = float(np.sum((i_win - np.polyval(quad, v_win)) ** 2))
            if resid_quad < 0.5 * resid_lin:  # curvature is real, not noise
                roots = np.roots(quad)
                roots = roots[np.isreal(roots)].real
                if len(roots) > 0:
                    v_quad = float(roots[np.argmin(np.abs(roots - voc_linear))])
                    window_span = float(v_win.max() - v_win.min())
                    if np.isfinite(v_quad) and abs(v_quad - voc_linear) <= window_span:
                        return v_quad
        except (np.linalg.LinAlgError, ValueError):
            pass

    return voc_linear


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
        # No V=0 crossing: same no-extrapolation rule as Voc.
        logger.warning("No V=0 crossing found — Isc out of sweep range; returning None.")
        return None

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

    # Select the fit window by VOLTAGE SPAN, not point count. The validity of
    # a local dV/dI depends on how far the window extends in voltage: a fixed
    # 20-point window at coarse step sizes reaches into the exponential diode
    # knee and destroys the estimate (e.g. a true 2000-ohm shunt reported as
    # ~30 ohm at 50 mV steps).
    v_span_sorted = np.sort(voltage_v)
    step = float(np.median(np.abs(np.diff(v_span_sorted)))) if len(v_span_sorted) > 1 else 0.0
    span = max(2.0 * step, 0.025)  # at least ±25 mV, or ±2 steps

    window_mask = np.abs(voltage_v - target_voltage) <= span
    if np.count_nonzero(window_mask) >= 3:
        local_voltages = voltage_v[window_mask]
        local_currents = current_a[window_mask]
    else:
        # Sparse data near the target: fall back to the 5 nearest points.
        indices_closest = np.argsort(np.abs(voltage_v - target_voltage))[:5]
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