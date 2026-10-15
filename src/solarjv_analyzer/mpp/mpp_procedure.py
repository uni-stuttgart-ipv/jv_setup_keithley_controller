"""Maximum-power-point tracking by perturb-and-observe.

The loop, which is the professor's `mpp_tracking.py` unchanged in method:

1. **Baseline** — stay at the present voltage, taking a sample every
   `data_interval`, until `probe_interval` has passed since the last probe.
2. **Probe** — step the source by `direction x probe_step` and take
   `probe_points` samples. The source is LEFT at the probe voltage.
3. **Decide** — compare the sums of the last N samples of each. If the probe
   was better, stay: the step has already moved us. If not, flip `direction`
   and go back.

What differs from his script, and why:

* **Averaging in software.** He reads the 2400's hardware buffer; we loop
  `:READ?` and average, the same path `JVProcedure` and `SpoProcedure` use.
  No new instrument code, and no hardware-buffer autoranging surprises.
* **No `power_production_mode`.** Generation is `V x I < 0` in BOTH quadrant 2
  (p-i-n contacted the usual way here) and quadrant 4 (n-i-p). Maximising
  `-(V x I)` needs no setting and cannot be set wrong.
* **A clamp.** The tracked voltage may not leave a window around where it
  started. An unattended three-hour run must not walk the cell into forward
  bias. Hitting the clamp is counted and reported, not silently obeyed.

Power per sample is `mean(V x I)` over the averaged readings, not
`mean(V) x mean(I)` — the two differ once V and I are both moving.
"""

import logging
import math
import os
import random
import threading
import time

from PyQt5 import QtCore
from pymeasure.experiment.parameters import (
    BooleanParameter, FloatParameter, IntegerParameter)

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.utils.directory_manager import DirectoryManager
from solarjv_analyzer.mpp.mpp_report import MppReport

logger = logging.getLogger(__name__)

MAX_RUN_TIME_MIN = 180.0          # 3 hours, agreed with the professor


class MppProcedure(JVProcedure):
    """Tracks the maximum power point of one channel for a set time."""

    DATA_COLUMNS = ["Channel", "Time (s)", "Voltage (V)", "Current (A)",
                    "Power (W)", "Phase"]

    # -- parameters -------------------------------------------------------
    run_time = FloatParameter("Run Time", units="min", default=5.0)
    start_voltage = FloatParameter("Start Voltage", units="V", default=0.0)
    probe_step = FloatParameter("Probe Step", units="mV", default=10.0)
    probe_interval = FloatParameter("Probe Interval", units="s", default=60.0)
    data_interval = FloatParameter("Data Interval", units="s", default=10.0)
    probe_points = IntegerParameter("Probe Points", default=3)
    averages_per_point = IntegerParameter("Averages per Point", default=10)
    max_excursion = FloatParameter("Max Excursion", units="mV", default=300.0)
    # Off by default, deliberately: a 2% gate never opens anywhere the tracker
    # would actually operate (a 10 mV probe near an MPP moves the power by
    # 0.015%-0.6%), so it would freeze the tracker at its start voltage. Left
    # available so real runs can tell us what scale is right.
    use_noise_threshold = BooleanParameter("Use Noise Threshold", default=False)
    noise_threshold_percent = FloatParameter("Noise Threshold", units="%",
                                             default=2.0)
    preconditioning_time = FloatParameter("Pre-conditioning Time", units="s",
                                          default=5.0)
    active_channel = IntegerParameter("Active Channel", default=1)

    def __init__(self, *args, manager=None, mux=None, instrument=None,
                 username=None, csv_path=None, report_path=None, **kwargs):
        super().__init__(*args, manager=manager, mux=mux,
                         instrument=instrument, **kwargs)
        self.username = username
        self._report = None
        self._csv_path = None
        self._custom_csv_path = csv_path
        self._custom_report_path = report_path

        # Tracker state, also what the metrics are built from.
        self._voltage = 0.0
        self._direction = 1
        self.probe_count = 0
        self.direction_flips = 0
        self.clamp_hits = 0
        self.samples = []          # (t, v, i, power_w, phase)

        # A seam for tests: a fake clock makes a five-minute run take no time
        # and removes every timing race from the assertions.
        self._clock = time.monotonic
        self._sleep = time.sleep

        # Simulated cell, used only when `simulation` is on. A parabola in
        # power with a known peak: enough to prove the tracker converges,
        # respects the clamp, and flips direction at the top.
        self.sim_vmpp = 0.85
        self.sim_pmax = 0.02       # W
        self.sim_width = 0.35      # V to where the power reaches zero
        self.sim_noise = 0.0       # W, std dev

    # -- lifecycle --------------------------------------------------------
    def startup(self):
        """Configure a fixed source, as SPO does — the tracker moves it."""
        super().startup()
        self._ensure_idle()
        self.emit('status', 'Configuring instrument...')

        self._write("*RST", "Reset")
        time.sleep(0.1)
        self._check_errors("After reset")

        self._write(":SOUR:FUNC VOLT", "Source function")
        self._write(":SOUR:VOLT:MODE FIX", "Fixed voltage mode")
        self._write(f":SOUR:VOLT {float(self.start_voltage)}", "Start voltage")
        self._write(":SENS:FUNC 'CURR'", "Measure current")

        # `*RST` has just reset this to 2-wire. The J-V sweep that CHOSE the
        # start voltage ran 4-wire, so without this the tracker would begin
        # from a voltage determined with the leads compensated and applied
        # with them in circuit. Same reasoning as SPO.
        sense_cmd = ":SYST:RSEN ON" if self.sense_mode == "4-wire" else ":SYST:RSEN OFF"
        self._write(sense_cmd, f"Sense mode: {self.sense_mode}")

        self._write(f":SENS:CURR:PROT {self.compliance_current}", "Compliance")
        self._write(f":SENS:CURR:NPLC {float(self.nplc):.3f}", "NPLC")
        self._write(":SENS:CURR:RANG:AUTO ON", "Auto range")
        self._write(":FORM:ELEM VOLT,CURR", "Format: Voltage,Current")
        self._write(":FORM:DATA ASC", "Format: ASCII")
        self._check_errors("After MPP configuration")

        self.emit('status', 'Pre-conditioning...')

    # -- measurement ------------------------------------------------------
    def _read_once(self):
        """One (voltage, current) reading from the instrument."""
        if self._sim:
            return self._simulated_reading()
        response = self._query(":READ?", "MPP measurement")
        values = [v.strip() for v in response.split(',') if v.strip()]
        return float(values[0]), float(values[1])

    def _simulated_reading(self):
        """A cell whose generated power peaks at `sim_vmpp`.

        Power is a parabola, zero at `sim_width` either side of the peak and
        never negative. The current that implies is then `-power / V`, so
        `-(V x I)` reproduces the parabola exactly and the tracker sees a
        well-defined maximum to find.
        """
        v = self._voltage
        rel = (v - self.sim_vmpp) / self.sim_width
        power = self.sim_pmax * max(0.0, 1.0 - rel * rel)
        if self.sim_noise:
            power = max(0.0, power + random.gauss(0.0, self.sim_noise))
        current = -power / v if abs(v) > 1e-9 else 0.0
        return v, current

    def _measure_point(self):
        """Average `averages_per_point` readings into one sample.

        Returns (voltage, current, power_w), where power is the mean of the
        per-reading products.
        """
        n = max(1, int(self.averages_per_point))
        vs, cs, ps = [], [], []
        for _ in range(n):
            v, i = self._read_once()
            vs.append(v)
            cs.append(i)
            ps.append(v * i)
        return (sum(vs) / n, sum(cs) / n, sum(ps) / n)

    # -- the tracker ------------------------------------------------------
    def _clamped(self, voltage):
        """Hold the tracked voltage inside its window. Returns (v, hit)."""
        limit = abs(float(self.max_excursion)) / 1000.0
        low = float(self.start_voltage) - limit
        high = float(self.start_voltage) + limit
        if voltage < low:
            return low, True
        if voltage > high:
            return high, True
        return voltage, False

    def _apply_voltage(self, voltage):
        self._voltage = voltage
        self._write(f":SOUR:VOLT {voltage:.6f}", "Track voltage")

    def _accept(self, probe_sum, baseline_sum):
        """Is the probe better than staying put?

        Scores are generated power, so bigger is better in both quadrants.
        With the threshold on, the probe must win by a margin relative to the
        baseline rather than by any amount at all.
        """
        gain = probe_sum - baseline_sum
        if self.use_noise_threshold:
            margin = abs(baseline_sum) * float(self.noise_threshold_percent) / 100.0
            return gain > margin
        return gain > 0.0

    def _record(self, elapsed, phase):
        """Measure, store, write to the journal, and emit. Returns the score."""
        voltage, current, power_w = self._measure_point()
        self.samples.append((elapsed, voltage, current, power_w, phase))
        if self._report is not None:
            self._report.write_row(elapsed, voltage, current, power_w, phase)
        self.emit('results', {
            "Channel": int(self.active_channel),
            "Time (s)": round(elapsed, 3),
            "Voltage (V)": voltage,
            "Current (A)": current,
            "Power (W)": power_w,
            "Phase": phase,
        })
        return -power_w          # generated power: positive in Q2 and Q4

    def _nap(self, seconds, t0, duration):
        """Sleep in slices so an abort or the end of the run takes effect."""
        target = self._clock() + max(0.0, seconds)
        while self._clock() < target:
            if self.should_stop() or (self._clock() - t0) >= duration:
                return
            self._sleep(min(0.1, target - self._clock()))

    def execute(self):
        channel = int(self.active_channel)
        duration = min(float(self.run_time), MAX_RUN_TIME_MIN) * 60.0
        step = abs(float(self.probe_step)) / 1000.0

        logger.info(
            f"Starting MPP on Channel {channel}: start={float(self.start_voltage)} V, "
            f"run={duration:.0f}s, step={step * 1000:.0f} mV, "
            f"probe every {float(self.probe_interval):.0f}s, "
            f"threshold={'on' if self.use_noise_threshold else 'off'}")

        if self.mux is not None:
            self.mux.select_channel(channel)

        try:
            self._open_report(channel)

            self._write(":OUTP ON", "Output on")
            self._check_errors("After output on")

            self._voltage = float(self.start_voltage)
            self._direction = 1
            self._apply_voltage(self._voltage)

            precond = max(0.0, float(self.preconditioning_time))
            self._nap(precond, self._clock(), precond + 1.0)
            if self.should_stop():
                return

            self.emit('status', 'Tracking')
            t0 = self._clock()
            last_probe = t0

            def expired():
                return (self._clock() - t0) >= duration or self.should_stop()

            while not expired():
                # --- baseline, until it is time to probe ------------------
                baseline = []
                while (self._clock() - last_probe) < float(self.probe_interval):
                    if expired():
                        break
                    baseline.append(self._record(self._clock() - t0, "baseline"))
                    self._nap(float(self.data_interval), t0, duration)
                if expired():
                    break

                # --- probe: step, and LEAVE the source there --------------
                previous = self._voltage
                target, hit = self._clamped(previous + self._direction * step)
                if hit:
                    self.clamp_hits += 1
                    logger.warning(
                        f"MPP clamp hit at {target:+.3f} V "
                        f"(+/-{float(self.max_excursion):.0f} mV around "
                        f"{float(self.start_voltage):+.3f} V)")
                self._apply_voltage(target)

                probe = []
                for _ in range(max(1, int(self.probe_points))):
                    if expired():
                        break
                    probe.append(self._record(self._clock() - t0, "probe"))
                    self._nap(float(self.data_interval), t0, duration)
                self.probe_count += 1
                last_probe = self._clock()

                # --- decide ----------------------------------------------
                n = min(int(self.probe_points), len(baseline), len(probe))
                if n > 0:
                    if not self._accept(sum(probe[-n:]), sum(baseline[-n:])):
                        self._direction *= -1
                        self.direction_flips += 1
                        self._apply_voltage(previous)

                self.emit('progress',
                          min(100.0, (self._clock() - t0) / duration * 100.0))

            self.emit('progress', 100.0)
            self.emit('status', 'Complete' if not self.should_stop() else 'Aborted')
            logger.info(
                f"MPP finished on Channel {channel}: {len(self.samples)} samples, "
                f"{self.probe_count} probes, {self.direction_flips} flips, "
                f"{self.clamp_hits} clamp hits, final {self._voltage:+.3f} V")

        except Exception as e:
            logger.error(f"MPP execution failed: {e}")
            self._safety_abort()
            raise
        finally:
            if self.mux is not None:
                try:
                    self.mux.deselect_channel(channel)
                except Exception:
                    pass

    def _open_report(self, channel):
        """Decide where the journal lives and open it."""
        if self._custom_csv_path:
            self._csv_path = self._custom_csv_path
        else:
            # DirectoryManager is a process-wide singleton: set the mode, read
            # the path, and ALWAYS put it back, or every later J-V run files
            # itself under MPP.
            dir_manager = DirectoryManager()
            previous_mode = dir_manager.mode
            dir_manager.set_username(self.username)
            try:
                dir_manager.set_mode("MPP")
                directory = dir_manager.get_current_directory(create=True)
            finally:
                dir_manager.set_mode(previous_mode)
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            self._csv_path = os.path.join(
                directory, f"mpp_ch{channel}_{timestamp}_raw.csv")

        self._report = MppReport(self._csv_path)
        self._report.init({
            "Start Voltage": (float(self.start_voltage), "V"),
            "Run Time": (float(self.run_time), "min"),
            "Probe Step": (float(self.probe_step), "mV"),
            "Probe Interval": (float(self.probe_interval), "s"),
            "Data Interval": (float(self.data_interval), "s"),
            "Probe Points": (int(self.probe_points), ""),
            "Averages per Point": (int(self.averages_per_point), ""),
            "Max Excursion": (float(self.max_excursion), "mV"),
            "Noise Threshold": (
                f"{float(self.noise_threshold_percent)}%"
                if self.use_noise_threshold else "off", ""),
            "Channel": (int(self.active_channel), ""),
            "Device Area": (float(self.device_area), "cm2"),
            "User Name": (self.username or "", ""),
        })

    def shutdown(self):
        try:
            self._write(":OUTP OFF", "Output off")
        except Exception:
            pass
        if self._report is not None:
            try:
                self._report.close()
            except Exception:
                pass
        super().shutdown()

    @property
    def csv_path(self):
        """Logical path of the raw time series (None until execute() runs)."""
        return self._csv_path

    @property
    def report(self):
        """The MppReport managing the journal (None until execute() runs)."""
        return self._report


class MppWorker(QtCore.QThread):
    """Runs an MppProcedure off the GUI thread. Mirrors SpoWorker exactly."""

    results_ready = QtCore.pyqtSignal(dict)
    progress_changed = QtCore.pyqtSignal(float)
    status_changed = QtCore.pyqtSignal(str)
    run_finished = QtCore.pyqtSignal(object)
    run_failed = QtCore.pyqtSignal(str)

    def __init__(self, procedure: MppProcedure, parent=None):
        super().__init__(parent)
        self.procedure = procedure
        self._stop_event = threading.Event()
        # The same extension points pymeasure's own Worker monkey-patches.
        self.procedure.emit = self._emit
        self.procedure.should_stop = self._stop_event.is_set

    def _emit(self, topic, record):
        if topic == 'results':
            self.results_ready.emit(record)
        elif topic == 'progress':
            self.progress_changed.emit(float(record))
        elif topic == 'status':
            self.status_changed.emit(str(record))

    def abort(self):
        """Ask for a graceful stop; checked between samples."""
        self._stop_event.set()

    def run(self):
        try:
            self.procedure.startup()
            self.procedure.execute()
        except Exception as exc:
            logger.error(f"MPP worker failed: {exc}")
            self.run_failed.emit(str(exc))
        finally:
            try:
                self.procedure.shutdown()
            except Exception as exc:
                logger.error(f"MPP shutdown failed: {exc}")
            self.run_finished.emit(self.procedure)
