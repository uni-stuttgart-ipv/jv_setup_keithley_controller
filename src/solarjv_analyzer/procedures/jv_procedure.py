"""
Keithley 2400 J-V Measurement Procedure

Implements hardware-controlled staircase sweep for solar cell characterization.
Uses the instrument's built-in sweep capability with automatic NPLC calculation
from user-specified sweep rate.
"""

import logging
import time
import sys
from typing import Optional
from math import isclose

import numpy as np
import pyvisa
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import (
    FloatParameter, BooleanParameter, Parameter, IntegerParameter
)

from solarjv_analyzer.analysis.analysis import compute_jv_metrics, ANALYSIS_LABELS_UNITS

# Configure module logger
logger = logging.getLogger(__name__)


class JVProcedure(Procedure):
    """
    J-V sweep procedure for Keithley 2400 SourceMeter.

    Executes a hardware-controlled staircase sweep from start_voltage to stop_voltage
    with specified step_size. The sweep rate determines the measurement speed, and
    NPLC is automatically calculated to match the requested rate.

    Signals:
        results: Emits individual data points during sweep
        progress: Emits progress percentage (0-100)
        analysis: Emits computed J-V metrics after sweep completion
    """

    signals = ['results', 'progress', 'log', 'status', 'analysis']

    DATA_COLUMNS = ["Channel", "Voltage (V)", "Current (A)", "Time (s)", "Status"]
    ANALYSIS_LABELS_UNITS = ANALYSIS_LABELS_UNITS

    # =========================================================================
    # Experiment Parameters
    # =========================================================================

    # Hardware
    mux = Parameter("MUX Object", default=None)
    gpib_address = Parameter("GPIB Address", default="GPIB::1")

    # Sweep Configuration
    start_voltage = FloatParameter("Start Voltage", units="V", default=1.2)
    stop_voltage = FloatParameter("Stop Voltage", units="V", default=-0.2)
    step_size = FloatParameter("Step Size", units="V", default=-0.01)
    sweep_rate = FloatParameter("Sweep Rate", units="V/s", default=0.1)
    compliance_current = FloatParameter("Compliance Current", units="A", default=0.18)

    # Channel Selection
    channel1 = BooleanParameter("Channel 1", default=True)
    channel2 = BooleanParameter("Channel 2", default=True)
    channel3 = BooleanParameter("Channel 3", default=True)
    channel4 = BooleanParameter("Channel 4", default=True)
    channel5 = BooleanParameter("Channel 5", default=True)
    channel6 = BooleanParameter("Channel 6", default=True)
    active_channel = Parameter("Active Channel", default="1")

    # Measurement Settings
    nplc = FloatParameter("NPLC", default=0.1)
    delay_between_points = FloatParameter("Dwell Time", units="s", default=0.0)
    pre_sweep_delay = FloatParameter("Pre-Sweep Delay", units="s", default=0.0)
    measurement_range = Parameter("Measurement Range", default="Auto")
    sense_mode = Parameter("Sense Mode", default="2-wire")

    # Sample Information
    device_area = FloatParameter("Device Area", units="cm2", default=0.089)
    incident_power = FloatParameter("Incident Power", units="mW/cm2", default=100)
    user_name = Parameter("User Name", default="")

    # Advanced Timing
    auto_zero = BooleanParameter("Auto Zero", default=False)
    line_frequency = FloatParameter("Line Frequency", units="Hz", default=50.0)

    # Analysis (4-Probe)
    contact_threshold = FloatParameter("Contact Threshold", units="A", default=0.001)
    lateral_factor = FloatParameter("4-Probe Lateral Factor", default=1.0)
    probe_spacing = FloatParameter("4-Probe Spacing", units="um", default=2290)
    sample_thickness = FloatParameter("Sample Thickness", units="um", default=500)

    # Debug & Validation
    simulation = BooleanParameter("Simulation Mode", default=False)
    check_errors_between_points = BooleanParameter("Check Errors Between Points", default=False)
    enable_validation = BooleanParameter("Enable Validation", default=False)
    voltage_retry_attempts = IntegerParameter("Voltage Retry Attempts", default=3)

    # Current range mapping from GUI selection to Amperes
    RANGE_MAP = {
        "1A": 1.0,
        "100 mA": 0.1,
        "10 mA": 0.01,
        "1 mA": 0.001,
        "100 uA": 0.0001,
    }

    def __init__(self, *args, manager=None, mux=None, instrument=None, **kwargs):
        """Initialize the procedure with instrument references."""
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.mux = mux
        self.instrument = instrument
        self._sim = bool(self.simulation)

        # Data storage
        self._voltages = []
        self._currents = []
        self._expected_voltages = []
        self.analysis_results = {}
        self._last_metrics = None

        # Internal state
        self._total_points = 0
        self._configured = False
        self._visa_resource = None
        self._line_freq = float(self.line_frequency)

        logger.info(f"JVProcedure initialized (Channel {self.active_channel}, "
                    f"Rate={self.sweep_rate} V/s, AutoZero={'ON' if self.auto_zero else 'OFF'})")

    # -------------------------------------------------------------------------
    # Hardware Communication
    # -------------------------------------------------------------------------

    def _write(self, cmd: str, description: str = ""):
        """Send a command to the instrument with optional logging."""
        if self._sim:
            return
        try:
            logger.debug(f"Sending: {description or cmd}")
            self.instrument.write(cmd)
            time.sleep(0.02)
        except Exception as e:
            logger.error(f"Failed to send {description}: {e}")
            raise

    def _query(self, cmd: str, description: str = "") -> str:
        """Send a query and return the response."""
        if self._sim:
            return "0,No error"
        try:
            logger.debug(f"Query: {description or cmd}")
            response = self.instrument.ask(cmd)
            logger.debug(f"Response: {response[:200]}")
            return response
        except Exception as e:
            logger.error(f"Failed to query {description}: {e}")
            raise

    def _check_errors(self, context: str):
        """Check the instrument's error queue and log any errors."""
        if self._sim:
            return
        try:
            response = self._query(":SYST:ERR?", "Error check")
            if not response.startswith("0,"):
                logger.warning(f"Error at {context}: {response}")
        except Exception as e:
            logger.error(f"Error check failed: {e}")

    def _set_timeout(self, seconds: float):
        """Configure VISA timeout for the instrument."""
        if self._sim or not self.instrument:
            return
        try:
            if hasattr(self.instrument, 'adapter') and hasattr(self.instrument.adapter, 'connection'):
                self._visa_resource = self.instrument.adapter.connection
                self._visa_resource.timeout = int(seconds * 1000)
                logger.debug(f"VISA timeout: {seconds:.1f}s")
        except Exception as e:
            logger.warning(f"Could not set VISA timeout: {e}")

    def _safety_abort(self):
        """Emergency abort: turn off output and return instrument to idle."""
        if self._sim or not self.instrument:
            return
        try:
            logger.warning("Safety abort initiated")
            self.instrument.write(":OUTP OFF")
            time.sleep(0.05)
            self.instrument.write(":ABOR")
            time.sleep(0.05)
            self.instrument.write("*CLS")
            logger.info("Safety abort complete")
        except Exception as e:
            logger.error(f"Safety abort failed: {e}")

    def _ensure_idle(self):
        """Bring instrument to IDLE state before configuration."""
        if self._sim or not self.instrument:
            return
        logger.debug("Ensuring idle state")
        self._write(":OUTP OFF", "Output off")
        self._write(":ABOR", "Abort")
        time.sleep(0.1)
        self._write(":TRIG:CLE", "Clear triggers")
        self._write("*CLS", "Clear status")
        time.sleep(0.05)

    # -------------------------------------------------------------------------
    # Sweep Parameter Calculation
    # -------------------------------------------------------------------------

    def _generate_voltage_sequence(self) -> list:
        """
        Generate the exact voltage sequence for the sweep.

        Creates a list of voltages from start to stop with the specified step size,
        ensuring the stop voltage is included.

        Returns:
            List of voltage values for the sweep
        """
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = float(self.step_size)

        if step == 0:
            step = 0.1
        if stop < start:
            step = -abs(step)
        else:
            step = abs(step)

        voltages = []
        current = start

        if step > 0:
            while current <= stop + (0.5 * step):
                voltages.append(round(current, 6))
                current += step
        else:
            while current >= stop + (0.5 * step):
                voltages.append(round(current, 6))
                current += step

        # Ensure stop voltage is included
        if voltages and abs(voltages[-1] - stop) > abs(step) * 0.1:
            voltages.append(stop)

        return voltages

    def _calculate_nplc(self) -> tuple:
        """
        Calculate NPLC and source delay from the requested sweep rate.

        Uses professor's method: total_time = voltage_range / sweep_rate
        time_per_point = total_time / total_points
        NPLC = time_per_point * line_frequency

        Returns:
            tuple: (nplc_value, source_delay_seconds)
        """
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = abs(float(self.step_size))

        total_points = int(abs(stop - start) / step) + 1

        # If sweep_rate is not specified, use user-provided NPLC
        if self.sweep_rate <= 0:
            return float(self.nplc), float(self.delay_between_points)

        voltage_range = abs(stop - start)
        total_time = voltage_range / self.sweep_rate
        time_per_point = total_time / total_points

        nplc = time_per_point * self._line_freq
        nplc = max(0.01, min(10.0, nplc))

        # Account for auto-zero measurement cycles
        measurement_time = (nplc / self._line_freq) * (3 if self.auto_zero else 1)
        source_delay = max(0, time_per_point - measurement_time)

        logger.info(f"Sweep timing: {total_time:.2f}s total, {time_per_point*1000:.1f}ms/point, "
                    f"NPLC={nplc:.3f}, delay={source_delay*1000:.1f}ms")

        return nplc, source_delay

    # -------------------------------------------------------------------------
    # Instrument Configuration
    # -------------------------------------------------------------------------

    def _configure_instrument(self) -> int:
        """
        Configure Keithley 2400 for hardware-controlled staircase sweep.

        Sets up source mode, measurement parameters, buffer, and trigger model.
        The sweep automatically stores readings in the buffer.

        Returns:
            int: Number of points in the sweep
        """
        if self._configured:
            return self._total_points

        self._ensure_idle()

        # Generate voltage sequence and calculate timing
        self._expected_voltages = self._generate_voltage_sequence()
        total_points = len(self._expected_voltages)

        if total_points > 2500:
            raise ValueError(f"Sweep requires {total_points} points > 2500 buffer limit")

        self._total_points = total_points
        nplc_val, source_delay = self._calculate_nplc()

        # Calculate timeout with safety margin
        measurement_time = (nplc_val / self._line_freq) * (3 if self.auto_zero else 1)
        time_per_point = measurement_time + source_delay
        total_time = total_points * time_per_point
        self._set_timeout(max(30, total_time * 3))

        # Reset instrument
        self._write("*RST", "Reset")
        time.sleep(0.2)
        self._check_errors("After reset")

        # Configure source for staircase sweep
        start_v = self._expected_voltages[0]
        stop_v = self._expected_voltages[-1]
        step_v = self._expected_voltages[1] - self._expected_voltages[0]

        self._write(":SOUR:FUNC VOLT", "Source function")
        self._write(":SOUR:VOLT:MODE SWE", "Sweep mode")
        self._write(f":SOUR:VOLT:STAR {start_v}", "Start voltage")
        self._write(f":SOUR:VOLT:STOP {stop_v}", "Stop voltage")
        self._write(f":SOUR:VOLT:STEP {step_v}", "Step size")
        self._check_errors("After source config")

        # Configure measurement
        self._write(":SENS:FUNC 'CURR'", "Measure current")

        sense_cmd = ":SYST:RSEN ON" if self.sense_mode == "4-wire" else ":SYST:RSEN OFF"
        self._write(sense_cmd, f"Sense mode: {self.sense_mode}")

        self._write(f":SENS:CURR:NPLC {nplc_val:.3f}", "NPLC")

        # Measurement range
        if self.measurement_range == "Auto":
            self._write(":SENS:CURR:RANG:AUTO ON", "Auto range")
        else:
            self._write(":SENS:CURR:RANG:AUTO OFF", "Auto range off")
            range_val = self.RANGE_MAP.get(self.measurement_range)
            if range_val:
                self._write(f":SENS:CURR:RANG {range_val}", f"Fixed range: {self.measurement_range}")

        self._write(f":SENS:CURR:PROT {self.compliance_current}", "Compliance")
        self._check_errors("After measurement config")

        # Source delay
        self._write(f":SOUR:DEL {source_delay:.6f}", "Source delay")
        self._write(":SOUR:DEL:AUTO OFF", "Auto delay off")

        # Auto-zero
        if self.auto_zero:
            self._write(":SYST:AZER ON", "Auto-zero on")
        else:
            self._write(":SYST:AZER OFF", "Auto-zero off")

        # Buffer configuration (sweep automatically stores readings)
        self._write(":TRAC:CLE", "Clear buffer")
        self._write(f":TRAC:POIN {total_points}", "Buffer size")
        self._write(":TRAC:FEED SENS", "Buffer feed")
        self._write(":TRAC:FEED:CONT NEXT", "Buffer control")

        # Data format
        self._write(":FORM:ELEM VOLT,CURR", "Format: Voltage,Current")
        self._write(":FORM:DATA ASC", "Format: ASCII")

        # Trigger model
        self._write(":TRIG:SOUR IMM", "Trigger source")
        self._write(f":TRIG:COUN {total_points}", "Trigger count")
        self._write(":TRIG:DEL 0", "Trigger delay")
        self._write(":TRIG:OUTP NONE", "No output triggers")
        self._write(":ARM:OUTP NONE", "No arm triggers")

        time.sleep(0.1)
        self._check_errors("End of configuration")

        self._configured = True
        logger.info(f"Instrument configured: {total_points} points")
        return total_points

    # -------------------------------------------------------------------------
    # Sweep Execution
    # -------------------------------------------------------------------------

    def _execute_sweep(self, total_points: int, channel: int) -> bool:
        """
        Execute the sweep and collect measurement data.

        Uses :READ? which triggers the sweep, waits for completion, and returns
        all data. Results are parsed and emitted for real-time display.

        Args:
            total_points: Number of points expected in the sweep
            channel: Active channel number

        Returns:
            bool: True if all points were collected successfully
        """
        measured_voltages = []
        measured_currents = []

        # Enable output
        self._write(":OUTP ON", "Output on")
        if self.pre_sweep_delay > 0:
            time.sleep(self.pre_sweep_delay)
        self._check_errors("After output on")

        logger.info(f"Starting sweep on Channel {channel}")
        sweep_start = time.time()

        try:
            response = self._query(":READ?", "Sweep data")
        except pyvisa.errors.VisaIOError as e:
            logger.error(f"Sweep timeout: {e}")
            self._safety_abort()
            raise
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            self._safety_abort()
            raise

        duration = time.time() - sweep_start
        logger.info(f"Sweep completed in {duration:.2f}s")

        # Turn output off for safety
        self._write(":OUTP OFF", "Output off")

        # Parse results (format: VOLT,CURR,VOLT,CURR,...)
        values = [v.strip() for v in response.split(',') if v.strip()]
        logger.debug(f"Received {len(values)} values")

        points_parsed = 0
        for i in range(0, len(values) - 1, 2):
            try:
                voltage = float(values[i])
                current = float(values[i+1])

                measured_voltages.append(voltage)
                measured_currents.append(current)
                points_parsed += 1

                self.emit('results', {
                    "Channel": channel,
                    "Voltage (V)": voltage,
                    "Current (A)": current,
                    "Time (s)": duration,
                    "Status": "OK",
                })

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse point {i//2}: {e}")

        self.emit("progress", 100)
        logger.info(f"Parsed {points_parsed}/{total_points} points")

        self._voltages = measured_voltages
        self._currents = measured_currents
        self._check_errors("After sweep")

        if points_parsed < total_points:
            logger.warning(f"Only {points_parsed}/{total_points} points received")
            return False
        return True

    # -------------------------------------------------------------------------
    # Analysis and File Writing
    # -------------------------------------------------------------------------

    def _write_analysis_to_file(self, channel: int, metrics: dict):
        """Append analysis results to the data file."""
        try:
            results_obj = getattr(self, "results", None)
            data_path = None

            # Locate the data file
            for name in ("data_filename", "data_path", "filename", "datafile", "data_file"):
                p = getattr(results_obj, name, None)
                if isinstance(p, str) and p:
                    data_path = p
                    break

            if data_path is None and hasattr(results_obj, "_data_file"):
                try:
                    data_path = results_obj._data_file.name
                except:
                    pass

            if not data_path or not metrics:
                return

            # Avoid duplicate analysis blocks
            try:
                with open(data_path, 'r', encoding='utf-8') as f:
                    if f"Channel\t{channel}" in f.read():
                        logger.debug(f"Analysis for Channel {channel} already exists")
                        return
            except FileNotFoundError:
                pass

            # Write analysis block
            with open(data_path, "a", encoding="utf-8") as f:
                f.write("\n[[ANALYSIS]]\n")
                f.write(f"Channel\t{channel}\n")
                for label, unit in self.ANALYSIS_LABELS_UNITS:
                    val = metrics.get(label, 0.0)
                    if isinstance(val, float):
                        formatted = f"{val:.6e}" if abs(val) >= 1000 or (abs(val) < 0.01 and val != 0) else f"{val:.6f}"
                    else:
                        formatted = str(val)
                    f.write(f"{label}\t{formatted}\t{unit}\n")
                f.write("[[/ANALYSIS]]\n")

            logger.debug(f"Analysis written for Channel {channel}")

        except Exception as e:
            logger.warning(f"Failed to write analysis: {e}")

    def _validate_measurement(self):
        """Compare measured voltages against expected sequence (debug only)."""
        if not self._voltages or not self._expected_voltages:
            return

        measured = np.array(self._voltages)
        expected = np.array(self._expected_voltages[:len(measured)])

        diff = measured - expected
        logger.debug(f"Voltage validation: mean error={np.mean(np.abs(diff)):.6f}V, "
                    f"max error={np.max(np.abs(diff)):.6f}V")

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    def startup(self):
        """Initialize instrument connections."""
        logger.info("Initializing instrument")

        try:
            # Get references from manager if not provided
            if self.instrument is None and self.manager:
                self.instrument = getattr(self.manager, "keithley", None)
            if self.mux is None and self.manager:
                self.mux = getattr(self.manager, "mux", None)

            # Connect if needed
            if self.instrument is None and not self._sim:
                from solarjv_analyzer.instruments.instrument_manager import get_keithley
                self.instrument = get_keithley(address=self.gpib_address)
                logger.info(f"Connected to Keithley at {self.gpib_address}")

                # Query line frequency
                try:
                    resp = self.instrument.ask(":SYST:LFR?")
                    resp = resp.strip().upper()
                    if resp in ("50", "60"):
                        self._line_freq = float(resp)
                        logger.info(f"Line frequency: {self._line_freq} Hz")
                except Exception as e:
                    logger.warning(f"Could not query line frequency: {e}")

            self._configured = False
            self._last_metrics = None
            logger.info("Startup complete")

        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise

    def execute(self):
        """Run the complete J-V sweep procedure."""
        logger.info("=" * 50)
        logger.info("Starting J-V sweep")

        try:
            channel = int(self.active_channel)
            logger.info(f"Channel {channel}: {self.start_voltage}V → {self.stop_voltage}V, "
                       f"step={self.step_size}V, rate={self.sweep_rate}V/s")

            # Select MUX channel
            if self.mux is not None:
                self.mux.select_channel(channel)
                time.sleep(self.pre_sweep_delay)

            # Configure and execute
            total_points = self._configure_instrument()
            self._voltages = []
            self._currents = []

            success = self._execute_sweep(total_points, channel)

            if not success and not self.should_stop():
                raise RuntimeError(f"Sweep incomplete: {len(self._voltages)}/{total_points}")

            if self.enable_validation:
                self._validate_measurement()

            self._finalize_analysis(channel)

        except pyvisa.errors.VisaIOError as e:
            logger.error(f"VISA timeout: {e}")
            self._safety_abort()
            raise
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self._safety_abort()
            raise
        finally:
            if self.mux is not None:
                try:
                    self.mux.deselect_channel(channel)
                except:
                    pass
            self._configured = False
            logger.info("Sweep execution finished")
            logger.info("=" * 50)

    def _finalize_analysis(self, channel: int):
        """
        Compute J-V metrics, store results, and write to file.

        Calculates efficiency, fill factor, Voc, Jsc, and other parameters
        from the collected I-V data.
        """
        try:
            if not self._voltages or not self._currents:
                return

            min_len = min(len(self._voltages), len(self._currents))
            voltages = self._voltages[:min_len]
            currents = self._currents[:min_len]

            metrics = compute_jv_metrics(
                v_raw=voltages,
                i_raw=currents,
                area_cm2=float(self.device_area),
                incident_power_mw_per_cm2=float(self.incident_power),
            )

            self.analysis_results[channel] = metrics
            self._last_metrics = {"Channel": channel, **metrics}
            self.emit('analysis', {"Channel": channel, **metrics})

            logger.info(f"Channel {channel} Results: "
                       f"EFF={metrics.get('EFF', 0):.2f}%, "
                       f"FF={metrics.get('FF', 0):.2f}%, "
                       f"Voc={metrics.get('Voc', 0):.4f}V, "
                       f"Jsc={metrics.get('Jsc', 0):.4f}mA/cm²")

            self._write_analysis_to_file(channel, metrics)

        except Exception as e:
            logger.warning(f"Analysis failed: {e}")

    def shutdown(self):
        """Clean shutdown - return instrument to safe state."""
        logger.info("Shutting down")
        self._safety_abort()
        logger.info("Shutdown complete")