import logging
import traceback
from typing import List, Dict, Optional
from math import isclose
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import FloatParameter, BooleanParameter, Parameter, IntegerParameter
import numpy as np
from time import sleep
from solarjv_analyzer.config import CONFIG
import time
from solarjv_analyzer.analysis.analysis import compute_jv_metrics, ANALYSIS_LABELS_UNITS
import os

logger = logging.getLogger(__name__)

class JVProcedure(Procedure):
    """
    Implements a multi-channel J-V sweep procedure for solar cells.
    OPTIMIZED VERSION - Much faster sweep speeds with GUI-respecting configuration.
    """
    signals = ['results', 'progress', 'log', 'status', 'analysis']
    
    DATA_COLUMNS = [
        "Channel",
        "Voltage (V)",
        "Current (A)",
        "Time (s)",
        "Status",
    ]

    ANALYSIS_LABELS_UNITS = ANALYSIS_LABELS_UNITS

    # --- PARAMETERS ---
    mux = Parameter("MUX Object", default=None)
    user_name = Parameter("User Name", default="")
    
    start_voltage = FloatParameter("Start Voltage", units="V", default=1.2)
    stop_voltage = FloatParameter("Stop Voltage", units="V", default=-0.2)
    step_size = FloatParameter("Step Size", units="V", default=-0.01)
    compliance_current = FloatParameter("Compliance Current", units="A", default=0.18)
    
    gpib_address = Parameter("GPIB Address", default="GPIB::1")
    
    channel1 = BooleanParameter("Channel 1", default=True)
    channel2 = BooleanParameter("Channel 2", default=True)
    channel3 = BooleanParameter("Channel 3", default=True)
    channel4 = BooleanParameter("Channel 4", default=True)
    channel5 = BooleanParameter("Channel 5", default=True)
    channel6 = BooleanParameter("Channel 6", default=True)
    
    nplc = FloatParameter("NPLC", default=0.1)
    delay_between_points = FloatParameter("Dwell Time", units="s", default=0.01)
    pre_sweep_delay = FloatParameter("Pre-Sweep Delay", units="s", default=0.0)
    
    measurement_range = Parameter("Measurement Range", default="Auto")
    sense_mode = Parameter("Sense Mode", default="2-wire")
    device_area = FloatParameter("Device Area", units="cm2", default=0.089)
    incident_power = FloatParameter("Incident Power", units="mW/cm2", default=100)
    contact_threshold = FloatParameter("Contact Threshold", units="A", default=0.001)
    lateral_factor = FloatParameter("4-Probe Lateral Factor", default=1.0)
    probe_spacing = FloatParameter("4-Probe Spacing", units="um", default=2290)
    sample_thickness = FloatParameter("Sample Thickness", units="um", default=500)
    active_channel = Parameter("Active Channel", default="1")
    simulation = BooleanParameter("Simulation Mode", default=False)
    
    check_errors_between_points = BooleanParameter("Check Errors Between Points", default=False)
    
    # Number of retry attempts for voltage setting
    voltage_retry_attempts = IntegerParameter("Voltage Retry Attempts", default=3)

    # Range mapping for GUI selection to Amps
    RANGE_MAP = {
        "1A": 1.0,
        "100 mA": 0.1,    # Note: GUI uses "100 mA" with space
        "10 mA": 0.01,
        "1 mA": 0.001,
        "100 uA": 0.0001,
    }

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
        self._last_v = 0
        self._direct_visa = None

    def _generate_voltages(self) -> np.ndarray:
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = float(self.step_size)
        if step == 0: 
            step = 0.1
        step = abs(step)
        if stop < start: 
            step = -step
        
        vals = np.arange(start, stop + (0.5 * step), step)
        if len(vals) > 0 and not isclose(vals[-1], stop, rel_tol=1e-9, abs_tol=1e-12):
            if (step > 0 and vals[-1] > stop) or (step < 0 and vals[-1] < stop):
                vals[-1] = stop
        return vals

    def _fast_read_current(self) -> float:
        """
        OPTIMIZED: Bypass PyMeasure overhead for faster readings.
        Properly parses Keithley's comma-separated response.
        """
        inst = self.instrument
        if inst is None:
            raise RuntimeError("Instrument is not initialized")
        
        # Try direct VISA communication first (fastest)
        try:
            if self._direct_visa is None and hasattr(inst, 'adapter') and hasattr(inst.adapter, 'connection'):
                self._direct_visa = inst.adapter.connection
            
            if self._direct_visa is not None:
                # Direct VISA query - much faster than going through PyMeasure layers
                response = self._direct_visa.query(":READ?")
                # FIX: Keithley returns "current,voltage,status,time" - parse correctly
                # Example: "0.002345,0.000000,0,0"
                parts = response.strip().split(',')
                if parts:
                    return float(parts[0])  # Return the current value
                else:
                    raise ValueError(f"Empty response from instrument: {response}")
        except Exception as e:
            logger.debug(f"Direct VISA read failed, falling back: {e}")
            # Fall back to other methods if direct fails
            pass
            
        # Try measure_current method (moderate speed)
        if hasattr(inst, "measure_current") and callable(getattr(inst, "measure_current")):
            try: 
                return float(inst.measure_current())
            except Exception: 
                pass
            
        # Try property access (slowest, but works)
        if hasattr(inst, "current"):
            try: 
                return float(getattr(inst, "current"))
            except Exception: 
                pass
            
        # Last resort - raw read
        if hasattr(inst, "read") and callable(getattr(inst, "read")):
            try: 
                raw = inst.read()
                # Try to parse comma-separated if present
                if ',' in raw:
                    return float(raw.split(',')[0])
                return float(raw)
            except Exception: 
                pass
            
        raise RuntimeError("Could not read current from instrument")

    def _set_voltage_with_retry(self, voltage: float, retries: int = 3) -> bool:
        """
        Safely set voltage with retry logic.
        Returns True if successful, False if all retries failed.
        """
        for attempt in range(retries):
            try:
                self.instrument.source_voltage = float(voltage)
                return True
            except Exception as e:
                logger.warning(f"Failed to set voltage {voltage}V (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    sleep(0.01)  # Short delay before retry
                else:
                    # All retries failed
                    logger.error(f"Failed to set voltage {voltage}V after {retries} attempts")
                    return False
        return False

    def startup(self) -> None:
        """OPTIMIZED: Instrument initialization that respects ALL GUI selections."""
        logger.info("Startup: initializing instruments with GUI settings")
        try:
            if self.instrument is None and getattr(self, "manager", None) is not None:
                self.instrument = getattr(self.manager, "keithley", None)
            if self.mux is None and getattr(self, "manager", None) is not None:
                self.mux = getattr(self.manager, "mux", None)

            if self.instrument is None and not self._sim:
                from solarjv_analyzer.instruments.instrument_manager import get_keithley
                addr = (getattr(self, "gpib_address", None) or "").strip() or CONFIG.GPIB_ADDRESS
                self.instrument = get_keithley(address=addr)

            self._start_time = time.time()

            if self.instrument is not None and not self._sim:
                # ------------------------------------------------------------
                # CRITICAL: Apply ALL GUI settings in the correct order
                # ------------------------------------------------------------
                
                # 1. Set source mode to voltage FIRST
                logger.info(f"Setting source mode to voltage")
                if hasattr(self.instrument, "source_mode"):
                    try:
                        self.instrument.source_mode = 'voltage'
                    except Exception as e:
                        logger.warning(f"Could not set source mode: {e}")
                
                # 2. Set compliance current (from GUI)
                logger.info(f"Setting compliance current to {self.compliance_current} A")
                if hasattr(self.instrument, "compliance_current"):
                    try:
                        self.instrument.compliance_current = self.compliance_current
                    except Exception as e:
                        logger.warning(f"Could not set compliance current: {e}")
                
                # 3. Apply sense mode (2-wire / 4-wire) from GUI
                logger.info(f"Setting sense mode to {self.sense_mode}")
                if hasattr(self.instrument, "write"):
                    if self.sense_mode == "4-wire":
                        self.instrument.write(":SYST:RSEN ON")
                        logger.info("4-wire sensing enabled")
                    else:
                        self.instrument.write(":SYST:RSEN OFF")
                        logger.info("2-wire sensing enabled")
                
                # 4. Apply measurement range from GUI
                logger.info(f"Setting measurement range to {self.measurement_range}")
                if hasattr(self.instrument, "write"):
                    if self.measurement_range == "Auto":
                        self.instrument.write(":SENS:CURR:RANG:AUTO ON")
                        logger.info("Auto range enabled")
                    else:
                        self.instrument.write(":SENS:CURR:RANG:AUTO OFF")
                        range_val = self.RANGE_MAP.get(self.measurement_range)
                        if range_val:
                            self.instrument.write(f":SENS:CURR:RANG {range_val}")
                            logger.info(f"Fixed range set to {range_val} A")
                        else:
                            logger.warning(f"Unknown range value: {self.measurement_range}")
                
                # 5. Apply NPLC from GUI
                logger.info(f"Setting NPLC to {self.nplc}")
                if hasattr(self.instrument, "write"):
                    self.instrument.write(f":SENS:CURR:NPLC {float(self.nplc)}")
                
                # 6. Set measurement format (only current for speed)
                if hasattr(self.instrument, "write"):
                    self.instrument.write(":FORM:ELEM CURR")
                
                # 7. Disable concurrent measurements (only measure current)
                if hasattr(self.instrument, "write"):
                    self.instrument.write(":SENS:FUNC:CONC OFF")
                    self.instrument.write(":SENS:FUNC 'CURR:DC'")
                
                # 8. NOW set the initial voltage (using actual start_voltage value)
                logger.info(f"Setting initial voltage to {self.start_voltage}V")
                if hasattr(self.instrument, "source_voltage"):
                    try:
                        self.instrument.source_voltage = float(self.start_voltage)
                    except Exception as e:
                        logger.error(f"Could not set initial source voltage: {e}")
                        raise
                
                # 9. Finally enable output
                if hasattr(self.instrument, "enable_source"):
                    try:
                        self.instrument.enable_source()
                        logger.info("Output enabled")
                    except Exception as e:
                        logger.warning(f"Could not enable source: {e}")
                
                # --- SPEED OPTIMIZATIONS (These don't affect measurement integrity) ---
                try:
                    # Disable delays and auto-zero for speed
                    if hasattr(self.instrument, "delay"):
                        self.instrument.delay = 0.0
                        
                    if hasattr(self.instrument, "write"):
                        self.instrument.write(":SYST:AZER OFF")
                        self.instrument.write(":DISP:ENAB ON")
                        self.instrument.write("*CLS")  # Clear errors
                        
                except Exception as e:
                    logger.warning(f"Could not apply speed optimizations: {e}")
                # ----------------------------------------------------------------------
                
                # Log final configuration for verification
                logger.info(f"Instrument configured - Range: {self.measurement_range}, Sense: {self.sense_mode}, NPLC: {self.nplc}")
                    
        except Exception as e:
            logger.error(f"Startup Failed: {e}", exc_info=True)
            raise

    def execute(self) -> None:
        """OPTIMIZED: Perform the J-V sweep with proper error handling."""
        import os
        import time
        import traceback
        import numpy as np
        from time import sleep
        
        try:
            logger.info("Beginning J-V sweep")
            ch = int(self.active_channel)
            voltages = self._generate_voltages()
            total_steps = len(voltages)
            step = 0
            retry_attempts = int(getattr(self, "voltage_retry_attempts", 3))

            self._voltages = []
            self._currents = []

            if self.results is None and getattr(self, "manager", None) is not None:
                self.results = getattr(self.manager, "results", None)

            logger.info(f"Measuring Channel {ch}")
            
            # MUX selection
            if self.mux is not None:
                try: 
                    self.mux.select_channel(ch)
                    sleep(0.02)
                except Exception as e: 
                    logger.warning(f"MUX select failed: {e}")

            sleep(float(self.pre_sweep_delay))
            
            raw_dwell = getattr(self, "delay_between_points", 0.0)
            dwell = float(raw_dwell if raw_dwell is not None else 0.0)
            
            for i, v in enumerate(voltages):
                if self.should_stop():
                    logger.warning(f"Aborting sweep early at Channel {ch}")
                    return

                # Use retry logic for voltage setting - NO 0.0V fallback
                voltage_set = self._set_voltage_with_retry(v, retry_attempts)
                if not voltage_set:
                    # Critical failure - abort the sweep
                    error_msg = f"Failed to set voltage {v}V after {retry_attempts} attempts. Aborting sweep."
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                self._last_v = v
                
                # Software Dwell
                if dwell > 0.001:
                    sleep(dwell)
                
                # Measure Current
                if self._sim:
                    current = (0.1 * v) + np.random.normal(0, 1e-4)
                else:
                    try:
                        current = self._fast_read_current()
                    except Exception as e:
                        logger.error(f"Critical: Current read failed at {v}V: {e}")
                        current = float('nan')

                self._voltages.append(float(v))
                self._currents.append(float(current))
                elapsed: float = time.time() - self._start_time
                
                self.emit('results', {
                    "Channel": ch,
                    "Voltage (V)": v,
                    "Current (A)": current,
                    "Time (s)": elapsed,
                    "Status": "OK" if not np.isnan(current) else "ERROR",
                })
                step += 1
                self.emit("progress", 100.0 * step / max(1, total_steps))
            
            self._finalize_analysis(ch)
            
        except Exception as e:
            logger.error(f"CRITICAL EXECUTION ERROR: {e}")
            logger.error(traceback.format_exc()) 
            raise  # Re-raise to properly abort the experiment

    def _finalize_analysis(self, ch):
        """Analysis with minimal overhead."""
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
            logger.warning(f"Metric computation failed: {e}")
            self._last_metrics = None
            
        if self.mux is not None:
            try: 
                self.mux.deselect_channel(ch)
                sleep(0.02)
            except Exception: 
                pass
            
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
        """Clean shutdown with hardware reset."""
        logger.info("Shutting down J-V procedure")
        try: super().shutdown()
        except: pass
        try:
            if self.instrument is not None and not self._sim:
                # Set output to 0V before disabling (safe value)
                if hasattr(self.instrument, "source_voltage"):
                    try:
                        self.instrument.source_voltage = 0.0
                        sleep(0.01)  # Allow time for voltage to settle
                    except:
                        pass
                
                # Disable output
                if hasattr(self.instrument, "disable_source"):
                    try:
                        self.instrument.disable_source()
                    except:
                        pass
                
                # Re-enable auto-zero for future runs
                if hasattr(self.instrument, "write"):
                    try: 
                        self.instrument.write(":SYST:AZER ON")
                        self.instrument.write(":DISP:ENAB ON")
                    except: pass
        except: pass