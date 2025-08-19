import logging
from typing import List, Dict, Optional
from math import isclose
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import FloatParameter, BooleanParameter, Parameter
import numpy as np
from time import sleep
from solarjv_analyzer.config import CONFIG
import time

logger = logging.getLogger(__name__)

class JVProcedure(Procedure):
    def _generate_voltages(self) -> np.ndarray:
        """Generate a robust list of voltages that works for up/down sweeps.
        Ensures the stop value is included and handles negative steps.
        """
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = float(self.step_size)
        if step == 0:
            step = 0.1
        step = abs(step)
        if stop < start:
            step = -step
        # Build using arange then ensure we include the exact stop within tolerance
        vals = np.arange(start, stop + (0.5 * step), step)
        # Clip the last point to be exactly stop if we overshot within tolerance
        if len(vals) and not isclose(vals[-1], stop, rel_tol=1e-9, abs_tol=1e-12):
            if (step > 0 and vals[-1] > stop) or (step < 0 and vals[-1] < stop):
                vals[-1] = stop
        return vals

    def _read_current(self) -> float:
        """Read current from the instrument with fallbacks for different drivers."""
        inst = self.instrument
        if inst is None:
            raise RuntimeError("Instrument is not initialized")
        # Preferred explicit measurement method
        if hasattr(inst, "measure_current") and callable(getattr(inst, "measure_current")):
            try:
                return float(inst.measure_current())
            except Exception:
                pass
        # Property fallback (many drivers expose .current)
        if hasattr(inst, "current"):
            try:
                return float(getattr(inst, "current"))
            except Exception:
                pass
        # Last resort: query-like method called read() if present
        if hasattr(inst, "read") and callable(getattr(inst, "read")):
            try:
                return float(inst.read())
            except Exception:
                pass
        raise RuntimeError("Could not read current from instrument")
    """
    Implements a multi-channel J–V sweep procedure for solar cells,
    logging data for voltage, current, elapsed time, and status.
    """
    # Columns: channel, voltage, current, elapsed time, and status
    DATA_COLUMNS = [
        "Channel",
        "Voltage (V)",
        "Current (A)",
        "Time (s)",
        "Status",
    ]

    # Parameters
    mux=Parameter("MUX Object", default=None)
    user_name = Parameter("User Name", default="")
    start_voltage = FloatParameter("Start Voltage (V)", default=0.0)
    stop_voltage = FloatParameter("Stop Voltage (V)", default=1.0)
    step_size = FloatParameter("Step Size (V)", default=0.1)
    compliance_current = FloatParameter("Compliance Current (A)", default=0.1)
    gpib_address = Parameter("GPIB Address", default="GPIB::1")
    channel1 = BooleanParameter("Channel 1", default=True)
    channel2 = BooleanParameter("Channel 2", default=True)
    channel3 = BooleanParameter("Channel 3", default=True)
    channel4 = BooleanParameter("Channel 4", default=True)
    channel5 = BooleanParameter("Channel 5", default=True)
    channel6 = BooleanParameter("Channel 6", default=True)
    nplc = FloatParameter("NPLC", default=1)
    delay_between_points = FloatParameter("Delay Between Points (s)", default=0.1)
    pre_sweep_delay = FloatParameter("Pre-Sweep Delay (s)", default=0.0)
    measurement_range = Parameter("Measurement Range", default="Auto")
    sense_mode = Parameter("Sense Mode", default="2-wire")
    device_area = FloatParameter("Device Area (cm^2)", default=0.089)
    incident_power = FloatParameter("Incident Power (mW/cm^2)", default=100)
    contact_threshold = FloatParameter("Contact Threshold (A)", default=0.001)
    lateral_factor = FloatParameter("4-Probe Lateral Factor", default=1.0)
    probe_spacing = FloatParameter("4-Probe Spacing (um)", default=2290)
    sample_thickness = FloatParameter("Sample Thickness (um)", default=500)
    active_channel = Parameter("Active Channel", default="1")
    simulation = BooleanParameter("Simulation Mode", default=False)


    def __init__(self, *args, manager: Optional[object] = None,
                 mux: Optional[object] = None,
                 instrument: Optional[object] = None,
                 **kwargs):
        """Optionally accept already-connected manager/mux/instrument to avoid double-opens."""
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.mux = mux
        self.instrument = instrument
        try:
            # cache simulation flag for fast checks
            self._sim = bool(self.simulation)
        except Exception:
            self._sim = False

    def startup(self) -> None:
        """
        Initialize instruments and record the start time.
        """
        logger.info("Startup: initializing instruments")

        # Prefer instances provided by the window's InstrumentManager
        if self.instrument is None and getattr(self, "manager", None) is not None:
            self.instrument = getattr(self.manager, "keithley", None)
        if self.mux is None and getattr(self, "manager", None) is not None:
            self.mux = getattr(self.manager, "mux", None)

        # Fallback: only try to create a real instrument if not in simulation
        if self.instrument is None and not self._sim:
            try:
                from solarjv_analyzer.instruments.instrument_manager import get_keithley
                addr = (getattr(self, "gpib_address", None) or "").strip() or CONFIG.GPIB_ADDRESS
                self.instrument = get_keithley(address=addr)
            except Exception as e:
                logger.error(f"Could not connect to Keithley: {e}")
                raise

        self._start_time = time.time()

        # Configure the SMU only if we have a real instrument and not in simulation
        if self.instrument is not None and not self._sim:
            try:
                self.instrument.apply_voltage()
                self.instrument.compliance_current = self.compliance_current
                self.instrument.source_voltage = self.start_voltage
                self.instrument.enable_source()
                # Try to set NPLC and delay in a version-tolerant way
                try:
                    if hasattr(self.instrument, "measure_current"):
                        self.instrument.measure_current(nplc=float(self.nplc))
                except Exception:
                    pass
                try:
                    if hasattr(self.instrument, "nplc"):
                        self.instrument.nplc = float(self.nplc)
                    if hasattr(self.instrument, "delay"):
                        self.instrument.delay = float(self.delay_between_points)
                except Exception:
                    pass
                if self.measurement_range != "Auto" and hasattr(self.instrument, "measurement_range"):
                    try:
                        self.instrument.measurement_range = self.measurement_range
                    except Exception:
                        pass
                if hasattr(self.instrument, "four_wire"):
                    self.instrument.four_wire = (self.sense_mode == "4-wire")
            except Exception as e:
                logger.error(f"Failed to configure Keithley: {e}")
                raise

    def execute(self) -> None:
        """
        Perform the J–V sweep across the selected channel,
        emitting results and progress updates.
        """
        logger.info("Beginning J–V sweep")
        ch = int(self.active_channel)
        voltages: np.ndarray = self._generate_voltages()
        total_steps: int = len(voltages)
        step: int = 0

        logger.info(f"Measuring Channel {ch}")
        if self.mux is not None:
            try:
                self.mux.select_channel(ch)
            except Exception as e:
                logger.warning(f"MUX select failed for channel {ch}: {e}")
        sleep(self.pre_sweep_delay)
        for v in voltages:
            if self.should_stop():
                logger.warning(
                    f"Aborting sweep early at Channel {ch}, Voltage {v:.3f} V"
                )
                if self.mux is not None:
                    try:
                        self.mux.deselect_channel(ch)
                    except Exception as e:
                        logger.warning(f"MUX deselect failed for channel {ch}: {e}")
                return
            # Program voltage then read current using a robust helper
            self.instrument.source_voltage = float(v)
            dwell = float(getattr(self, "delay_between_points", 0.0) or 0.0)
            if dwell > 0:
                sleep(dwell)
            current: float = self._read_current()
            elapsed: float = time.time() - self._start_time
            status: str = "OK"
            self.emit('results', {
                "Channel": ch,
                "Voltage (V)": v,
                "Current (A)": current,
                "Time (s)": elapsed,
                "Status": status,
            })
            step += 1
            progress: float = 100.0 * step / max(1, total_steps)
            self.emit("progress", min(progress, 100.0))
            logger.debug(
                f"Step {step}/{total_steps}: "
                f"Ch={ch}, V={v:.3f} V, I={current:.6e} A, "
                f"Elapsed={elapsed:.2f}s, Progress={progress:.1f}%"
            )
        if self.mux is not None:
            try:
                self.mux.deselect_channel(ch)
            except Exception as e:
                logger.warning(f"MUX deselect failed for channel {ch}: {e}")
        self.emit("progress", 100.0)
        logger.info("J–V sweep complete")

    def shutdown(self) -> None:
        """
        Clean up the instrument output and append metadata to the CSV.
        """
        logger.info("Shutting down J–V procedure and cleaning up")
        try:
            super().shutdown()
        except AttributeError:
            # If parent shutdown is unavailable, ignore
            pass
        try:
            if self.instrument is not None and not self._sim:
                if hasattr(self.instrument, "shutdown"):
                    self.instrument.shutdown()
                elif hasattr(self.instrument, "disable_source"):
                    try:
                        self.instrument.disable_source()
                    except Exception:
                        pass
        except Exception:
            pass
        # Placeholder for appending metadata to the data file:
        #   [[PARAMS]]
        #   DeviceArea=...
        #   Sample thickness=...
        # Actual metadata insertion will be implemented later.