import logging
from typing import List, Dict, Optional
from math import isclose
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import FloatParameter, BooleanParameter, Parameter
import numpy as np
from time import sleep
from solarjv_analyzer.config import CONFIG
import time

# NECESSARY CHANGE: Import the main calculator from your new analysis package
from solarjv_analyzer.analysis.main_calculator import run_full_analysis

logger = logging.getLogger(__name__)

class JVProcedure(Procedure):
    """
    Implements a multi-channel J–V sweep procedure for solar cells.
    This version delegates all post-measurement analysis to an external package.
    """
    signals = ['results', 'progress', 'log', 'status']
    DATA_COLUMNS = [
        "Channel", "Voltage (V)", "Current (A)", "Time (s)", "Status",
    ]
    ANALYSIS_LABELS_UNITS = [
        ("EFF - Efficency", "%"), ("FF- fill factor", "%"),
        ("Voc - open circuit volatge", "mV"), ("Jsc - short circ. current density", "mA/cm2"),
        ("Vmax", "mV"), ("Jmax", "mA/cm2"),
        ("Isc - short circ. current", "A"), ("Rsc - short circ. resistence", "Ohm"),
        ('Roc  open ""', "Ohm"), ("A - Area", "cm2"), ("Incd. Pwr", "mW/cm2"),
    ]

    # All parameters remain the same as your original file
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
    flip_to_labview = BooleanParameter("Flip current to LabVIEW sign", default=True)

    def __init__(self, *args, manager: Optional[object] = None,
                 mux: Optional[object] = None,
                 instrument: Optional[object] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.manager = manager
        self.mux = mux
        self.instrument = instrument
        self._sim = bool(self.simulation)
        self._voltages = []
        self._currents = []
        self.results = None
        self.analysis_results: Dict[int, Dict[str, float]] = {}

    def startup(self) -> None:
        """Initializes instruments and records the start time."""
        logger.info("Startup: initializing instruments")
        if self.instrument is None and self.manager:
            self.instrument = getattr(self.manager, "keithley", None)
        if self.mux is None and self.manager:
            self.mux = getattr(self.manager, "mux", None)
        
        self._start_time = time.time()
        
        if self.instrument and not self._sim:
            try:
                self.instrument.apply_voltage()
                self.instrument.compliance_current = self.compliance_current
                self.instrument.source_voltage = self.start_voltage
                self.instrument.enable_source()
            except Exception as e:
                logger.error(f"Failed to configure Keithley: {e}")
                raise

    def execute(self) -> None:
        """Performs the J–V sweep and calls the external analysis package."""
        logger.info("Beginning J–V sweep")
        ch = int(self.active_channel)
        voltages: np.ndarray = self._generate_voltages()
        total_steps: int = len(voltages)

        self._voltages, self._currents = [], []

        logger.info(f"Measuring Channel {ch}")
        if self.mux:
            self.mux.select_channel(ch)
        
        sleep(self.pre_sweep_delay)

        for i, v in enumerate(voltages):
            if self.should_stop():
                logger.warning(f"Aborting sweep at Channel {ch}, V={v:.3f} V")
                break
            
            self.instrument.source_voltage = float(v)
            sleep(self.delay_between_points)
            current = self._read_current()

            self._voltages.append(float(v))
            self._currents.append(float(current))
            elapsed = time.time() - self._start_time

            self.emit('results', {
                "Channel": ch, "Voltage (V)": v, "Current (A)": current,
                "Time (s)": elapsed, "Status": "OK",
            })
            self.emit("progress", 100.0 * (i + 1) / total_steps)

        # --- NECESSARY CHANGE: Call to External Analysis Package ---
        try:
            metrics = run_full_analysis(
                voltage_v=self._voltages,
                current_a=self._currents,
                device_area_cm2=float(self.device_area),
                incident_power_mw_per_cm2=float(self.incident_power),
                flip_current=bool(self.flip_to_labview),
            )
            self._last_metrics = {"Channel": ch, **metrics} if metrics else None
            self.analysis_results[ch] = metrics if metrics else {}
        except Exception as e:
            logger.warning(f"Metric computation failed for channel {ch}: {e}")
            self._last_metrics = None
        # --- End of Change ---

        if self.mux:
            self.mux.deselect_channel(ch)
        
        self.emit("progress", 100.0)
        logger.info("J–V sweep complete")

        # This logic for appending the analysis block to the CSV is preserved
        if hasattr(self, "results") and self.results and self._last_metrics:
            try:
                with open(self.results.data_filename, "a", encoding="utf-8") as f:
                    f.write("\n[[ANALYSIS]]\n")
                    for label, unit in self.ANALYSIS_LABELS_UNITS:
                        val = self._last_metrics.get(label, 0.0)
                        f.write(f"{label}\t{val}\t{unit}\n")
                    f.write("[[/ANALYSIS]]\n")
            except Exception as e:
                logger.warning(f"Failed to append [[ANALYSIS]] block: {e}")

    def shutdown(self) -> None:
        """Cleans up the instrument output."""
        logger.info("Shutting down J–V procedure")
        if self.instrument and not self._sim:
            if hasattr(self.instrument, "shutdown"):
                self.instrument.shutdown()

    def _generate_voltages(self) -> np.ndarray:
        start = float(self.start_voltage)
        stop = float(self.stop_voltage)
        step = float(self.step_size)
        if step == 0: step = 0.1
        step = abs(step)
        if stop < start: step = -step
        vals = np.arange(start, stop + (0.5 * step), step)
        if len(vals) and not isclose(vals[-1], stop):
            vals[-1] = stop
        return vals

    def _read_current(self) -> float:
        inst = self.instrument
        if inst is None: raise RuntimeError("Instrument is not initialized")
        if hasattr(inst, "measure_current"): return float(inst.measure_current())
        if hasattr(inst, "current"): return float(getattr(inst, "current"))
        raise RuntimeError("Could not read current from instrument")
