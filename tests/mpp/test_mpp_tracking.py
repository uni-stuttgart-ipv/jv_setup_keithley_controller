"""Does the tracker actually track?

Everything here runs headless against a simulated cell with a known maximum,
on a fake clock, so a five-minute run takes no time and no assertion depends
on timing. That is the point of proving the engine before any UI exists: if
the tracker does not converge here, it will not converge on a real cell, and
finding that out on the lab PC costs an afternoon per attempt.

The simulated cell's generated power is a parabola peaking at `sim_vmpp`. Real
P(V) is not a parabola, but it is smooth and concave near its maximum, which
is the only property perturb-and-observe relies on.
"""
import os

import pytest

from solarjv_analyzer.mpp.mpp_analysis import compute_mpp_metrics
from solarjv_analyzer.mpp.mpp_procedure import MppProcedure
from solarjv_analyzer.store import paths

VMPP = 0.85                     # where the simulated cell peaks


class FakeClock:
    """Time that only moves when the procedure sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


def make(tmp_path, **overrides):
    """An MPP run wired to the simulated cell and the fake clock."""
    proc = MppProcedure(simulation=True)
    # Both of these are monkey-patched by MppWorker in production; without
    # them pymeasure's Procedure raises NotImplementedError.
    proc.emit = lambda *a, **k: None
    proc.should_stop = lambda: False

    defaults = dict(
        run_time=60.0,            # minutes of simulated time
        start_voltage=0.60,
        probe_step=10.0,          # mV
        probe_interval=60.0,
        data_interval=10.0,
        probe_points=3,
        averages_per_point=1,     # the averaging is covered on its own below
        max_excursion=300.0,      # mV
        preconditioning_time=0.0,
        active_channel=1,
    )
    defaults.update(overrides)
    for name, value in defaults.items():
        setattr(proc, name, value)

    proc.sim_vmpp = VMPP
    proc.sim_noise = 0.0
    clock = FakeClock()
    proc._clock = clock
    proc._sleep = clock.sleep
    proc._custom_csv_path = str(tmp_path / "mpp_ch1_2026-10-13_raw.csv")
    return proc, clock


def tracked(proc):
    return [v for _t, v, _i, _p, _phase in proc.samples]


# --- convergence ------------------------------------------------------------

def test_it_climbs_to_the_maximum_from_below(tmp_path):
    proc, _ = make(tmp_path, start_voltage=0.60)
    proc.execute()

    assert proc.samples, "the run recorded nothing"
    assert max(tracked(proc)) > 0.60, "the tracker never moved upward"
    # Within a few probe steps of the peak, and it got there deliberately.
    assert abs(proc._voltage - VMPP) < 0.05, (
        f"settled at {proc._voltage:.3f} V, expected near {VMPP} V")
    assert proc.clamp_hits == 0


def test_it_walks_down_to_the_maximum_from_above(tmp_path):
    proc, _ = make(tmp_path, start_voltage=1.10)
    proc.execute()

    assert min(tracked(proc)) < 1.10, "the tracker never moved downward"
    assert abs(proc._voltage - VMPP) < 0.05, (
        f"settled at {proc._voltage:.3f} V, expected near {VMPP} V")


def test_it_hunts_once_it_is_on_the_peak(tmp_path):
    """Sitting at the maximum, both directions look bad, so it alternates.

    Flipping is the healthy state. A run with NO flips never found a peak.
    """
    proc, _ = make(tmp_path, start_voltage=VMPP, run_time=20.0)
    proc.execute()

    assert proc.direction_flips > 0, "the tracker never reversed at the peak"
    assert abs(proc._voltage - VMPP) < 0.05


# --- the safety clamp -------------------------------------------------------

def test_the_clamp_stops_it_leaving_its_window(tmp_path):
    """An unattended run must not walk the cell into forward bias.

    Started at 0.55 V with a +/-100 mV window, so the peak at 0.85 V is
    outside it: the tracker climbs, runs into the edge, and must stay there.
    (Starting further down would put the simulated cell outside its parabola,
    where it generates nothing and there is no gradient to climb — see
    `test_a_cell_producing_nothing_leaves_the_tracker_still`.)
    """
    proc, _ = make(tmp_path, start_voltage=0.55, max_excursion=100.0)
    proc.execute()

    high = 0.55 + 0.100
    assert proc.clamp_hits > 0, "the tracker should have wanted to leave"
    assert max(tracked(proc)) <= high + 1e-9, (
        f"tracked to {max(tracked(proc)):.3f} V, outside the +/-100 mV window")
    assert min(tracked(proc)) >= 0.55 - 0.100 - 1e-9
    # And it kept running rather than stopping.
    assert proc.samples


# --- the noise threshold ----------------------------------------------------

def test_the_threshold_off_by_default_lets_it_move(tmp_path):
    proc, _ = make(tmp_path, start_voltage=0.80)
    assert not proc.use_noise_threshold, "the threshold must default to off"
    proc.execute()

    assert abs(proc._voltage - VMPP) < 0.05


def test_a_two_percent_threshold_freezes_the_tracker(tmp_path):
    """The finding that put the threshold behind a checkbox, as a test.

    A 10 mV probe near a maximum changes the power by well under 1%, because
    P(V) is flat there by definition. A 2% acceptance gate therefore never
    opens and the tracker holds its start voltage for the whole run — the
    feature does nothing. Kept as a test so nobody turns 2% on as a default
    without seeing this.
    """
    proc, _ = make(tmp_path, start_voltage=0.80,
                   use_noise_threshold=True, noise_threshold_percent=2.0)
    proc.execute()

    assert proc.probe_count > 0, "it should still have probed"
    assert proc.direction_flips == proc.probe_count, (
        "every probe should have been rejected")
    assert abs(proc._voltage - 0.80) < 1e-9, (
        f"the tracker moved to {proc._voltage:.3f} V despite the 2% gate")


def test_a_workable_threshold_still_permits_movement(tmp_path):
    """Same run, a scale that matches what a probe actually gains."""
    proc, _ = make(tmp_path, start_voltage=0.80,
                   use_noise_threshold=True, noise_threshold_percent=0.1)
    proc.execute()

    assert abs(proc._voltage - VMPP) < 0.05


# --- measurement ------------------------------------------------------------

def test_a_point_is_the_mean_of_several_readings(tmp_path):
    """Averaging is what makes a noisy cell trackable at all."""
    proc, _ = make(tmp_path, averages_per_point=20, run_time=5.0)
    proc.sim_noise = 0.002        # 10% of sim_pmax
    reads = []
    original = proc._read_once
    proc._read_once = lambda: (reads.append(1), original())[1]

    proc._measure_point()

    assert len(reads) == 20, "averages_per_point readings were not taken"


def test_phase_is_recorded_for_every_sample(tmp_path):
    """So the plot can mark probes and the decisions stay auditable."""
    proc, _ = make(tmp_path, run_time=10.0)
    proc.execute()

    phases = {phase for _t, _v, _i, _p, phase in proc.samples}
    assert phases == {"baseline", "probe"}


# --- abort ------------------------------------------------------------------

def test_an_abort_keeps_everything_measured_so_far(tmp_path):
    proc, _ = make(tmp_path)
    stop = {"flag": False}
    proc.should_stop = lambda: stop["flag"]

    original = proc._record
    def record(elapsed, phase):
        score = original(elapsed, phase)
        if len(proc.samples) >= 12:
            stop["flag"] = True
        return score
    proc._record = record

    proc.execute()

    assert 12 <= len(proc.samples) <= 16, (
        f"abort left {len(proc.samples)} samples")
    assert proc.report is not None


# --- the journal and the report --------------------------------------------

def test_the_journal_is_invisible_to_the_store_while_tracking(tmp_path):
    """A 3-hour run must not be published and deleted underneath itself."""
    proc, _ = make(tmp_path, run_time=5.0)
    proc.execute()

    journal = proc.report.raw_path
    assert journal.endswith(paths.JOURNAL_SUFFIX)
    assert paths.is_journal_name(os.path.basename(journal))
    assert os.path.exists(journal), "the journal should survive until finalize"
    assert not os.path.exists(proc.csv_path), (
        "nothing should carry the publishable name during the run")


def test_finalize_writes_the_report_and_clears_the_journal(tmp_path):
    proc, _ = make(tmp_path, run_time=5.0)
    proc.execute()

    metrics = compute_mpp_metrics(
        [s[0] for s in proc.samples],
        [s[1] for s in proc.samples],
        [s[2] for s in proc.samples],
        probe_count=proc.probe_count,
        direction_flips=proc.direction_flips,
        clamp_hits=proc.clamp_hits,
        start_voltage_v=float(proc.start_voltage),
    )
    target = proc.report.finalize(metrics)

    assert os.path.exists(target)
    assert not os.path.exists(proc.report.raw_path), "the journal outlived the report"
    body = open(target, encoding="utf-8").read()
    for block in ("[[ EXPERIMENTAL PARAMETERS ]]", "[[ ANALYSIS SUMMARY ]]",
                  "[[ TIME SERIES DATA ]]"):
        assert block in body, f"{block} missing from the report"
    assert "Phase" in body


# --- metrics ----------------------------------------------------------------

def test_metrics_describe_the_run(tmp_path):
    proc, _ = make(tmp_path, start_voltage=0.60)
    proc.execute()

    metrics = compute_mpp_metrics(
        [s[0] for s in proc.samples],
        [s[1] for s in proc.samples],
        [s[2] for s in proc.samples],
        area_cm2=0.089, incident_power_mw_cm2=100.0,
        probe_count=proc.probe_count,
        direction_flips=proc.direction_flips,
        clamp_hits=proc.clamp_hits,
        start_voltage_v=0.60,
    )

    assert metrics["start_voltage_v"] == pytest.approx(0.60)
    assert metrics["tracked_v_max"] > 0.60
    assert metrics["sample_count"] == len(proc.samples)
    assert metrics["probe_count"] == proc.probe_count
    assert metrics["clamp_hits"] == 0
    # Powers are magnitudes, like Pmpp in the J-V report.
    assert metrics["max_power_mw"] > 0
    assert metrics["stabilised_power_mw"] > 0
    # It ends near the peak, so the stabilised power is close to the maximum.
    assert metrics["stabilised_power_mw"] == pytest.approx(
        metrics["max_power_mw"], rel=0.02)


def test_metrics_survive_an_empty_run():
    metrics = compute_mpp_metrics([], [], [])
    assert metrics["sample_count"] == 0
    assert metrics["mean_power_mw"] == 0.0


def test_generation_in_quadrant_two_is_positive_power():
    """A p-i-n cell contacted the usual way here tracks around -1.1 V.

    Power must come out positive there, exactly as it does in quadrant 4 —
    the tracker maximises -(V x I), which needs no architecture setting.
    """
    q4 = compute_mpp_metrics([0.0, 1.0], [0.85, 0.85], [-0.02, -0.02])
    q2 = compute_mpp_metrics([0.0, 1.0], [-1.11, -1.11], [0.02, 0.02])

    assert q4["mean_power_mw"] > 0
    assert q2["mean_power_mw"] > 0
    assert q2["tracked_v_max"] < 0, "the tracked voltage must stay signed"


def test_a_cell_producing_nothing_leaves_the_tracker_still(tmp_path):
    """A dark, disconnected or dead channel has no gradient to follow.

    Every probe reads the same zero, so nothing ever beats the baseline and
    the tracker returns to where it started each time. It should sit still
    and keep logging rather than wander the cell across its whole window
    looking for a peak that is not there.
    """
    proc, _ = make(tmp_path, start_voltage=0.0, run_time=20.0)
    proc.sim_pmax = 0.0

    proc.execute()

    assert proc.samples, "it should still have logged the dead channel"
    assert proc.probe_count > 0
    assert proc.direction_flips == proc.probe_count
    assert abs(proc._voltage) < 1e-9, (
        f"the tracker wandered to {proc._voltage:.3f} V with nothing to track")
