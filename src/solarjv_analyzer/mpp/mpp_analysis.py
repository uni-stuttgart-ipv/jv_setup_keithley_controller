"""Metrics for an MPP run.

Two of these say whether the TRACKER behaved, which is the part a stability
number cannot tell you:

* `direction_flips` — a tracker sitting on the maximum alternates, because
  both directions look equally bad. Constant flipping is healthy. *No* flips
  means it walked one way the whole run and never found a peak.
* `clamp_hits` — the tracked voltage ran into its excursion window. Anything
  above zero means the run is suspect: the tracker wanted to go somewhere it
  was not allowed, so the reported power is not a maximum.

Powers are reported as MAGNITUDES, like `Pmpp` in the J-V report. Voltages are
SIGNED, like `Vmpp`, because a p-i-n cell contacted the usual way generates in
quadrant 2 and legitimately tracks around -1.1 V. See `analysis.py` and
`claude/device-architecture-convention.md`.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MPP_METRICS_UNITS = [
    ("start_voltage_v", "V"),
    ("final_voltage_v", "V"),
    ("tracked_v_min", "V"),
    ("tracked_v_max", "V"),
    ("mean_power_mw", "mW"),
    ("max_power_mw", "mW"),
    ("initial_power_mw", "mW"),
    ("final_power_mw", "mW"),
    ("stabilised_power_mw", "mW"),
    ("drift_percent", "%"),
    ("efficiency_percent", "%"),
    ("probe_count", ""),
    ("direction_flips", ""),
    ("clamp_hits", ""),
    ("sample_count", ""),
    ("run_time_s", "s"),
]

# How much of the tail counts as "stabilised". The number people quote from a
# tracking run is the plateau, not the mean over a settling transient.
STABILISED_TAIL_FRACTION = 0.10


def _empty_metrics() -> dict:
    return {name: 0.0 for name, _unit in MPP_METRICS_UNITS}


def compute_mpp_metrics(
    time_s,
    voltage_v,
    current_a,
    area_cm2: Optional[float] = None,
    incident_power_mw_cm2: Optional[float] = None,
    probe_count: int = 0,
    direction_flips: int = 0,
    clamp_hits: int = 0,
    start_voltage_v: Optional[float] = None,
) -> dict:
    """Summarise one MPP run.

    Args:
        time_s: elapsed seconds per sample.
        voltage_v: the voltage actually measured at each sample, signed.
        current_a: measured current, signed.
        area_cm2 / incident_power_mw_cm2: both needed for efficiency; without
            them efficiency is reported as 0.0 rather than guessed.
        probe_count / direction_flips / clamp_hits: counted by the tracker,
            not derivable from the time series.
        start_voltage_v: where the run began. Defaults to the first sample.

    Returns:
        dict keyed by the names in MPP_METRICS_UNITS.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(voltage_v, dtype=float)
    i = np.asarray(current_a, dtype=float)

    if t.size == 0 or v.size == 0 or i.size == 0:
        logger.warning("compute_mpp_metrics called with no data.")
        return _empty_metrics()

    n = min(t.size, v.size, i.size)
    t, v, i = t[:n], v[:n], i[:n]

    # Generated power is -(V * I): positive in quadrant 2 AND quadrant 4, so
    # this needs no knowledge of the device architecture or the wiring.
    power_w = -(v * i)
    power_mw = power_w * 1000.0

    tail = max(1, int(round(n * STABILISED_TAIL_FRACTION)))
    initial_mw = float(power_mw[0])
    final_mw = float(power_mw[-1])

    # Drift against the START, so a cell that degrades reads negative. Guarded
    # because a run that began at zero power has no meaningful percentage.
    if abs(initial_mw) > 1e-12:
        drift = (final_mw - initial_mw) / abs(initial_mw) * 100.0
    else:
        drift = 0.0

    efficiency = 0.0
    if area_cm2 and incident_power_mw_cm2:
        incident_mw = float(area_cm2) * float(incident_power_mw_cm2)
        if incident_mw > 0:
            efficiency = float(np.mean(power_mw[-tail:])) / incident_mw * 100.0

    metrics = {
        "start_voltage_v": float(start_voltage_v if start_voltage_v is not None
                                 else v[0]),
        "final_voltage_v": float(v[-1]),
        "tracked_v_min": float(np.min(v)),
        "tracked_v_max": float(np.max(v)),
        "mean_power_mw": float(np.mean(np.abs(power_mw))),
        "max_power_mw": float(np.max(np.abs(power_mw))),
        "initial_power_mw": abs(initial_mw),
        "final_power_mw": abs(final_mw),
        "stabilised_power_mw": float(np.mean(np.abs(power_mw[-tail:]))),
        "drift_percent": float(drift),
        "efficiency_percent": float(efficiency),
        "probe_count": int(probe_count),
        "direction_flips": int(direction_flips),
        "clamp_hits": int(clamp_hits),
        "sample_count": int(n),
        "run_time_s": float(t[-1]),
    }
    return metrics
