import logging
from typing import List, Dict, Optional
from math import isclose
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import FloatParameter, BooleanParameter, Parameter
import numpy as np
from time import sleep
from solarjv_analyzer.config import CONFIG
import time
from solarjv_analyzer.analysis.analysis import compute_jv_metrics, ANALYSIS_LABELS_UNITS

logger = logging.getLogger(__name__)

class JVProcedure(Procedure):
    signals = ['results', 'progress', 'log', 'status', 'analysis']
    
    def _generate_voltages(self) -> np.ndarray:
        """Generate a robust list of voltages that works for up/down sweeps."""
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = float(self.step_size)
        if step == 0:
            step = 0.1
        step = abs(step)
        if stop < start:
            step = -step
        vals = np.arange(start, stop + (0.5 * step), step)
        if len(vals) and not isclose(vals[-1], stop, rel_tol=1e-9, abs_tol=1e-12):
            if (step > 0 and vals[-1] > stop) or (step < 0 and vals[-1] < stop):
                vals[-1] = stop
        return vals

    def _resolve_read_method(self):
        """
        Performance Optimization:
        Determines the fastest available method to read current ONCE at startup.
        """
        inst = self.instrument
        if inst is None: return None

        # Priority 1: Direct 'current' property (Fastest)
        if hasattr(inst, "current"):
            try:
                _ = inst.current
                return lambda: inst.current
            except Exception: pass

        # Priority 2: measure_current() method
        if hasattr(inst, "measure_current") and callable(inst.measure_current):
            return inst.measure_current

        # Priority 3: Raw read
        if hasattr(inst, "read"):
            return lambda: float(inst.read())
            
        return None

    def _read_current(self) -> float:
        """Fallback read method."""
        inst = self.instrument
        if inst is None:
            raise RuntimeError("Instrument is not initialized")
        if hasattr(inst, "measure_current") and callable(getattr(inst, "measure_current")):
            try: return float(inst.measure_current())
            except Exception: pass
        if hasattr(inst, "current"):
            try: return float(getattr(inst, "current"))
            except Exception: pass
        if hasattr(inst, "read") and callable(getattr(inst, "read")):
            try: return float(inst.read())
            except Exception: pass
        raise RuntimeError("Could not read current from instrument")
    
    """
    Implements a multi-channel J-V sweep procedure for solar cells.
    """
    DATA_COLUMNS = [
        "Channel",
        "Voltage (V)",
        "Current (A)",
        "Time (s)",
        "Status",
    ]

    ANALYSIS_LABELS_UNITS = ANALYSIS_LABELS_UNITS

    # Parameters
    mux=Parameter("MUX Object", default=None)
    user_name = Parameter("User Name", default="")
    start_voltage = FloatParameter("Start Voltage (V)", default=1.2)
    stop_voltage = FloatParameter("Stop Voltage (V)", default=-0.200)
    step_size = FloatParameter("Step Size (V)", default=-0.010)
    compliance_current = FloatParameter("Compliance Current (A)", default=0.18)
    gpib_address = Parameter("GPIB Address", default="GPIB::1")
    channel1 = BooleanParameter("Channel 1", default=True)
    channel2 = BooleanParameter("Channel 2", default=True)
    channel3 = BooleanParameter("Channel 3", default=True)
    channel4 = BooleanParameter("Channel 4", default=True)
    channel5 = BooleanParameter("Channel 5", default=True)
    channel6 = BooleanParameter("Channel 6", default=True)
    nplc = FloatParameter("NPLC", default=1)
    
    # UNCOMMENTED THIS PARAMETER (It was commented out in your file)
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
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.mux = mux
        self.instrument = instrument
        try:
            self._sim = bool(self.simulation)
        except Exception:
            self._sim = False
        self._voltages = []
        self._currents = []
        self.results = None 
        self.analysis_results = {}
        self._cached_read_method = None # Performance Cache

    def startup(self) -> None:
        """Initialize instruments and record the start time."""
        logger.info("Startup: initializing instruments")

        if self.instrument is None and getattr(self, "manager", None) is not None:
            self.instrument = getattr(self.manager, "keithley", None)
        if self.mux is None and getattr(self, "manager", None) is not None:
            self.mux = getattr(self.manager, "mux", None)

        if self.instrument is None and not self._sim:
            try:
                from solarjv_analyzer.instruments.instrument_manager import get_keithley
                addr = (getattr(self, "gpib_address", None) or "").strip() or CONFIG.GPIB_ADDRESS
                self.instrument = get_keithley(address=addr)
            except Exception as e:
                logger.error(f"Could not connect to Keithley: {e}")
                raise

        # PERFORMANCE: Resolve the read method once here
        self._cached_read_method = self._resolve_read_method()

        self._start_time = time.time()

        if self.instrument is not None and not self._sim:
            try:
                self.instrument.apply_voltage()
                self.instrument.compliance_current = self.compliance_current
                self.instrument.source_voltage = self.start_voltage
                self.instrument.enable_source()
                try:
                    if hasattr(self.instrument, "measure_current"):
                        self.instrument.measure_current(nplc=float(self.nplc))
                except Exception: pass
                
                try:
                    if hasattr(self.instrument, "nplc"):
                        self.instrument.nplc = float(self.nplc)
                    
                    # PERFORMANCE FIX: Set Hardware Delay to 0
                    # We rely on Python sleep in execute() for the dwell time.
                    # This prevents "Double Delay".
                    if hasattr(self.instrument, "delay"):
                        self.instrument.delay = 0.0
                except Exception: pass
                
                if self.measurement_range != "Auto" and hasattr(self.instrument, "measurement_range"):
                    try:
                        self.instrument.measurement_range = self.measurement_range
                    except Exception: pass
                if hasattr(self.instrument, "four_wire"):
                    self.instrument.four_wire = (self.sense_mode == "4-wire")
            except Exception as e:
                logger.error(f"Failed to configure Keithley: {e}")
                raise

    def execute(self) -> None:
        """Perform the J-V sweep."""
        logger.info("Beginning J-V sweep")
        ch = int(self.active_channel)
        voltages: np.ndarray = self._generate_voltages()
        total_steps: int = len(voltages)
        step: int = 0

        self._voltages = []
        self._currents = []

        if self.results is None and getattr(self, "manager", None) is not None:
            self.results = getattr(self.manager, "results", None)

        logger.info(f"Measuring Channel {ch}")
        if self.mux is not None:
            try: self.mux.select_channel(ch)
            except Exception as e: logger.warning(f"MUX select failed: {e}")
        
        sleep(self.pre_sweep_delay)
        
        # PERFORMANCE: Get cached read method
        read_func = self._cached_read_method
        dwell = float(getattr(self, "delay_between_points", 0.0) or 0.0)

        for v in voltages:
            if self.should_stop():
                logger.warning(f"Aborting sweep early at Channel {ch}, Voltage {v:.3f} V")
                if self.mux is not None:
                    try: self.mux.deselect_channel(ch)
                    except Exception as e: logger.warning(f"MUX deselect failed: {e}")
                return

            # 1. Source
            self.instrument.source_voltage = float(v)
            
            # 2. Dwell (Python Sleep)
            if dwell > 0:
                sleep(dwell)
            
            # 3. Measure (Optimized)
            if self._sim:
                current = (0.1 * v) + np.random.normal(0, 1e-4)
            elif read_func:
                current = float(read_func())
            else:
                current = self._read_current() # Fallback

            self._voltages.append(float(v))
            self._currents.append(float(current))
            elapsed: float = time.time() - self._start_time
            
            self.emit('results', {
                "Channel": ch,
                "Voltage (V)": v,
                "Current (A)": current,
                "Time (s)": elapsed,
                "Status": "OK",
            })
            step += 1
            progress: float = 100.0 * step / max(1, total_steps)
            self.emit("progress", min(progress, 100.0))
            
        try:
            metrics = compute_jv_metrics(
                v_raw=self._voltages,
                i_raw=self._currents,
                area_cm2=float(self.device_area),
                incident_power_mw_per_cm2=float(self.incident_power),
            )
            self._last_metrics = {"Channel": ch, **metrics}
            self.analysis_results[ch] = metrics
        except Exception as e:
            logger.warning(f"Metric computation failed for channel {ch}: {e}")
            self._last_metrics = None
            
        if self.mux is not None:
            try: self.mux.deselect_channel(ch)
            except Exception as e: logger.warning(f"MUX deselect failed: {e}")
            
        self.emit("progress", 100.0)
        logger.info("J-V sweep complete")

        try:
            results_obj = getattr(self, "results", None)
            data_path = None
            for name in ("data_filename", "data_path", "filename", "datafile", "data_file"):
                p = getattr(results_obj, name, None)
                if isinstance(p, str) and p:
                    data_path = p
                    break
            
            if data_path is None and hasattr(results_obj, "_data_file") and getattr(results_obj, "_data_file"):
                try: data_path = results_obj._data_file.name
                except: pass
            
            try:
                if hasattr(results_obj, "_data_file") and results_obj._data_file:
                    results_obj._data_file.flush()
            except: pass
            
            if getattr(self, "_last_metrics", None) and data_path:
                logger.info(f"Appending [[ANALYSIS]] block to {data_path}")
                with open(data_path, "a", encoding="utf-8") as f:
                    f.write("\n[[ANALYSIS]]\n")
                    f.write(f"Channel\t{ch}\n")
                    for label, unit in self.ANALYSIS_LABELS_UNITS:
                        val = self._last_metrics.get(label, 0.0)
                        f.write(f"{label}\t{val}\t{unit}\n")
                    f.write("[[/ANALYSIS]]\n")
        except Exception as e:
            logger.warning(f"Failed to append [[ANALYSIS]] block: {e}")

    def shutdown(self) -> None:
        """Clean up."""
        logger.info("Shutting down J-V procedure")
        try: super().shutdown()
        except: pass
        try:
            if self.instrument is not None and not self._sim:
                if hasattr(self.instrument, "shutdown"):
                    self.instrument.shutdown()
                elif hasattr(self.instrument, "disable_source"):
                    try: self.instrument.disable_source()
                    except: pass
        except: pass