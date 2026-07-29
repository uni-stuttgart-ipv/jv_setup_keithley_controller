"""
SPO (Set-Point Operation) Metrics Computation

Computes stability-test metrics from a constant-voltage hold time series:
mean/std power, drift between the start and end of the run, energy yield,
and peak/trough power. Mirrors the style of analysis/analysis.py but is
kept self-contained inside the spo/ package.
"""

import logging
from typing import Optional

import numpy as np

try:
    from numpy import trapezoid
except ImportError:
    # numpy < 2.0 compatibility
    from numpy import trapz as trapezoid

logger = logging.getLogger(__name__)

# Defines the display labels and units for the final SPO report.
SPO_METRICS_UNITS = [
    ("hold_voltage_v", "V"),
    ("hold_duration_s", "s"),
    ("mean_current_a", "A"),
    ("mean_power_mw", "mW"),
    ("std_power_mw", "mW"),
    ("drift_percent", "%"),
    ("max_power_mw", "mW"),
    ("min_power_mw", "mW"),
    ("initial_power_mw", "mW"),
    ("final_power_mw", "mW"),
    ("total_energy_j", "J"),
    ("sample_count", ""),
]


def _empty_metrics() -> dict:
    return {label: 0.0 for label, _ in SPO_METRICS_UNITS}


def compute_spo_metrics(
    time_s: np.ndarray,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    area_cm2: Optional[float] = None,
) -> dict:
    """
    Compute SPO stability metrics from a constant-voltage hold time series.

    Args:
        time_s: Elapsed time in seconds for each sample.
        current_a: Measured current in Amperes for each sample.
        voltage_v: Measured (or held) voltage in Volts. May be a scalar or
            an array the same length as time_s.
        area_cm2: Optional device area (reserved for future normalized-power
            metrics; currently unused but accepted for API stability).

    Returns:
        dict: Metrics keyed by the names in SPO_METRICS_UNITS.
    """
    t = np.asarray(time_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    v = np.asarray(voltage_v, dtype=float)

    if t.size == 0 or i.size == 0:
        logger.warning("compute_spo_metrics called with empty data.")
        return _empty_metrics()

    if v.size == 1 and t.size > 1:
        v = np.full_like(t, float(v))

    valid_mask = ~np.isnan(t) & ~np.isnan(i) & ~np.isnan(v)
    t, i, v = t[valid_mask], i[valid_mask], v[valid_mask]

    n = t.size
    if n == 0:
        logger.warning("compute_spo_metrics: all samples were NaN.")
        return _empty_metrics()

    power_w = v * i
    power_mw = power_w * 1000.0

    mean_current_a = float(np.mean(i))
    mean_power_mw = float(np.mean(power_mw))
    std_power_mw = float(np.std(power_mw))
    max_power_mw = float(np.max(power_mw))
    min_power_mw = float(np.min(power_mw))

    # Drift: compare average of first 5 points to average of last 5 points.
    edge = min(5, n)
    initial_power_mw = float(np.mean(power_mw[:edge]))
    final_power_mw = float(np.mean(power_mw[-edge:]))

    if abs(initial_power_mw) > 1e-12:
        drift_percent = float(
            (final_power_mw - initial_power_mw) / abs(initial_power_mw) * 100.0
        )
    else:
        drift_percent = 0.0
        logger.debug("Initial power near zero; drift_percent set to 0.0")

    if n > 1:
        total_energy_j = float(trapezoid(np.abs(power_w), t))
    else:
        total_energy_j = 0.0

    hold_voltage_v = float(np.mean(v))
    hold_duration_s = float(t[-1] - t[0]) if n > 1 else 0.0

    return {
        "hold_voltage_v": round(hold_voltage_v, 6),
        "hold_duration_s": round(hold_duration_s, 3),
        "mean_current_a": round(mean_current_a, 8),
        "mean_power_mw": round(mean_power_mw, 4),
        "std_power_mw": round(std_power_mw, 4),
        "drift_percent": round(drift_percent, 3),
        "max_power_mw": round(max_power_mw, 4),
        "min_power_mw": round(min_power_mw, 4),
        "initial_power_mw": round(initial_power_mw, 4),
        "final_power_mw": round(final_power_mw, 4),
        "total_energy_j": round(total_energy_j, 6),
        "sample_count": int(n),
    }
