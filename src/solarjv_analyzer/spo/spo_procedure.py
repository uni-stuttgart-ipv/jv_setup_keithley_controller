"""
SPO (Set-Point Operation) Procedure

Passive stability test: the Keithley 2400 is set to source a fixed
"hold_voltage" (typically Vmax found from a quick J-V sweep) and the current
is sampled at a regular interval for "hold_duration" seconds. Every sample is
written to disk and flushed immediately via SpoReport, so a crash never loses
previously-recorded data.

SpoProcedure inherits instrument communication, safety-abort, and MUX
handling from JVProcedure so the SPO module never re-implements low-level
Keithley control.
"""

import logging
import os
import threading
import time

from PyQt5 import QtCore
from pymeasure.experiment.parameters import FloatParameter, IntegerParameter

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.utils.directory_manager import DirectoryManager
from solarjv_analyzer.spo.spo_report import SpoReport

logger = logging.getLogger(__name__)


class SpoProcedure(JVProcedure):
    """
    Holds a solar cell at a constant voltage and logs current/power over time.

    Signals (inherited from JVProcedure): results, progress, log, status, analysis.
    """

    DATA_COLUMNS = ["Channel", "Time (s)", "Voltage (V)", "Current (A)", "Power (W)"]

    # ---------------------------------------------------------------------
    # SPO-specific parameters
    hold_voltage = FloatParameter("Hold Voltage", units="V", default=0.6)
    hold_duration = FloatParameter("Hold Duration", units="s", default=300.0)
    sampling_interval = FloatParameter("Sampling Interval", units="s", default=1.0)
    preconditioning_time = FloatParameter("Pre-conditioning Time", units="s", default=5.0)
    active_channel = IntegerParameter("Active Channel", default=1)

    def __init__(self, *args, manager=None, mux=None, instrument=None, username=None, **kwargs):
        super().__init__(*args, manager=manager, mux=mux, instrument=instrument, **kwargs)
        self.username = username
        self._report = None
        self._csv_path = None

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def startup(self):
        """Connect to hardware (reused from JVProcedure) then configure a
        fixed-voltage source instead of a staircase sweep."""
        super().startup()
        self._ensure_idle()

        self.emit('status', 'Configuring instrument...')

        self._write("*RST", "Reset")
        time.sleep(0.1)
        self._check_errors("After reset")

        self._write(":SOUR:FUNC VOLT", "Source function")
        self._write(":SOUR:VOLT:MODE FIX", "Fixed voltage mode")
        self._write(f":SOUR:VOLT {float(self.hold_voltage)}", "Hold voltage")
        self._write(":SENS:FUNC 'CURR'", "Measure current")
        self._write(f":SENS:CURR:PROT {self.compliance_current}", "Compliance")
        self._write(f":SENS:CURR:NPLC {float(self.nplc):.3f}", "NPLC")
        self._write(":SENS:CURR:RANG:AUTO ON", "Auto range")
        self._write(":FORM:ELEM VOLT,CURR", "Format: Voltage,Current")
        self._write(":FORM:DATA ASC", "Format: ASCII")
        self._check_errors("After SPO configuration")

        self.emit('status', 'Pre-conditioning...')

    def _measure_point(self):
        """Read one (voltage, current) sample from the instrument."""
        if self._sim:
            import random
            noise = random.uniform(-0.02, 0.02)
            current = -0.015 * (1.0 + noise)
            return float(self.hold_voltage), current

        response = self._query(":READ?", "SPO measurement")
        values = [v.strip() for v in response.split(',') if v.strip()]
        voltage = float(values[0])
        current = float(values[1])
        return voltage, current

    def execute(self):
        """Pre-condition, then hold voltage and log current/power over time."""
        channel = int(self.active_channel)
        logger.info(
            f"Starting SPO on Channel {channel}: hold={float(self.hold_voltage)} V, "
            f"duration={float(self.hold_duration)}s, interval={float(self.sampling_interval)}s"
        )

        if self.mux is not None:
            self.mux.select_channel(channel)

        try:
            self._write(":OUTP ON", "Output on")
            self._check_errors("After output on")

            # --- Pre-conditioning: stabilize at hold voltage before logging ---
            precond = max(0.0, float(self.preconditioning_time))
            waited = 0.0
            step = 0.1
            while waited < precond:
                if self.should_stop():
                    return
                sleep_for = min(step, precond - waited)
                time.sleep(sleep_for)
                waited += sleep_for

            if self.should_stop():
                return

            # --- Open the raw CSV immediately (crash-safe, flushed per row) ---
            # DirectoryManager is a process-wide singleton (its __init__ only
            # applies args on first construction), so update it via its
            # setters and restore the previous mode afterward to avoid
            # side-effects on the JV file panel's "Main" directory display.
            dir_manager = DirectoryManager()
            previous_mode = dir_manager.mode
            dir_manager.set_username(self.username)
            dir_manager.set_mode("SPO")
            directory = dir_manager.get_current_directory(create=True)
            dir_manager.set_mode(previous_mode)
            timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
            self._csv_path = os.path.join(directory, f"spo_ch{channel}_{timestamp}_raw.csv")

            parameters = {
                "Hold Voltage": (float(self.hold_voltage), "V"),
                "Hold Duration": (float(self.hold_duration), "s"),
                "Sampling Interval": (float(self.sampling_interval), "s"),
                "Pre-conditioning Time": (float(self.preconditioning_time), "s"),
                "Channel": (channel, ""),
                "Device Area": (float(self.device_area), "cm2"),
                "User Name": (self.username or self.user_name or "", ""),
            }
            self._report = SpoReport(self._csv_path)
            self._report.init(parameters)

            self.emit('status', 'Running')

            duration = float(self.hold_duration)
            interval = max(0.05, float(self.sampling_interval))
            start_time = time.time()
            next_sample = start_time

            while True:
                elapsed = time.time() - start_time
                if elapsed >= duration or self.should_stop():
                    break

                voltage, current = self._measure_point()
                power = voltage * current

                self.emit('results', {
                    "Channel": channel,
                    "Time (s)": round(elapsed, 3),
                    "Voltage (V)": voltage,
                    "Current (A)": current,
                    "Power (W)": power,
                })
                self._report.write_row(elapsed, voltage, current, power)

                progress = min(100.0, (elapsed / duration) * 100.0) if duration > 0 else 100.0
                self.emit('progress', progress)

                next_sample += interval
                sleep_time = next_sample - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.emit('progress', 100.0)
            self.emit('status', 'Complete' if not self.should_stop() else 'Aborted')
            logger.info(f"SPO finished for Channel {channel}: {self._csv_path}")

        except Exception as e:
            logger.error(f"SPO execution failed: {e}")
            self._safety_abort()
            raise
        finally:
            if self.mux is not None:
                try:
                    self.mux.deselect_channel(channel)
                except Exception:
                    pass

    def shutdown(self):
        """Turn off output, close the raw CSV file, then defer to JVProcedure."""
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
        """Path to the raw, crash-safe time-series CSV (None until execute() opens it)."""
        return self._csv_path

    @property
    def report(self):
        """The SpoReport instance managing the raw CSV (None until execute() opens it)."""
        return self._report


class SpoWorker(QtCore.QThread):
    """
    Runs a SpoProcedure's startup/execute/shutdown lifecycle on a background
    thread, forwarding emit('results'/'progress'/'status') calls as Qt
    signals (thread-safe queued connections to the GUI thread).

    This mirrors the extension points PyMeasure's own Worker uses
    (monkey-patching `procedure.emit` / `procedure.should_stop`), but stays
    self-contained so the SPO module does not depend on PyMeasure's
    Manager/Browser machinery, which is tied to the JV experiment queue.
    """

    results_ready = QtCore.pyqtSignal(dict)
    progress_changed = QtCore.pyqtSignal(float)
    status_changed = QtCore.pyqtSignal(str)
    run_finished = QtCore.pyqtSignal(object)
    run_failed = QtCore.pyqtSignal(str)

    def __init__(self, procedure: SpoProcedure, parent=None):
        super().__init__(parent)
        self.procedure = procedure
        self._stop_event = threading.Event()

        # Monkey-patch the same extension points PyMeasure's Worker uses.
        self.procedure.emit = self._emit
        self.procedure.should_stop = self._stop_event.is_set

    def _emit(self, topic, record):
        if topic == 'results':
            self.results_ready.emit(record)
        elif topic == 'progress':
            self.progress_changed.emit(float(record))
        elif topic == 'status':
            self.status_changed.emit(str(record))
        # 'log' and 'analysis' topics are not used by SPO.

    def abort(self):
        """Request a graceful stop; the procedure checks should_stop() between samples."""
        self._stop_event.set()

    def run(self):
        try:
            self.procedure.startup()
            self.procedure.execute()
        except Exception as e:
            logger.error(f"SPO worker failed: {e}")
            self.run_failed.emit(str(e))
        finally:
            try:
                self.procedure.shutdown()
            except Exception as e:
                logger.error(f"SPO shutdown error: {e}")
            self.run_finished.emit(self.procedure)
