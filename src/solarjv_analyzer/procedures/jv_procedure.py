import logging
from typing import List, Dict

#
from solarjv_analyzer.instruments.instrument_manager import get_keithley
from pymeasure.experiment import Procedure
from pymeasure.experiment.parameters import FloatParameter, BooleanParameter, Parameter
import numpy as np
from time import sleep
from solarjv_analyzer.config import CONFIG
import time

logger = logging.getLogger(__name__)

class JVProcedure(Procedure):
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

    def startup(self) -> None:
        """
        Initialize instruments and record the start time.
        """
        logger.info("Startup: initializing Keithley instrument")
        from solarjv_analyzer.instruments.mux_controller import MuxController, SimulatedMux
        if self.simulation:
            self.mux = SimulatedMux()
        else:
            self.mux = MuxController(port=CONFIG.MUX_PORT)
            self.mux.connect()
        #self.instrument = get_keithley(self.gpib_address, simulate=CONFIG.SIMULATION_MODE)
        self.instrument = get_keithley()
        self._start_time = time.time()
        # Configure the source-meter for voltage sourcing
        self.instrument.apply_voltage()
        self.instrument.compliance_current = self.compliance_current
        self.instrument.source_voltage = self.start_voltage
        self.instrument.enable_source()
        self.instrument.nplc = self.nplc
        self.instrument.delay = self.delay_between_points
        if self.measurement_range != "Auto":
            self.instrument.measurement_range = self.measurement_range
        if self.sense_mode == "4-wire":
            self.instrument.four_wire = True
        else:
            self.instrument.four_wire = False

    def execute(self) -> None:
        """
        Perform the J–V sweep across the selected channel,
        emitting results and progress updates.
        """
        logger.info("Beginning J–V sweep")
        ch = int(self.active_channel)
        voltages: np.ndarray = np.arange(
            self.start_voltage,
            self.stop_voltage + self.step_size,
            self.step_size
        )
        total_steps: int = len(voltages)
        step: int = 0

        logger.info(f"Measuring Channel {ch}")
        self.mux.select_channel(ch)
        sleep(self.pre_sweep_delay)
        for v in voltages:
            if self.should_stop():
                logger.warning(
                    f"Aborting sweep early at Channel {ch}, Voltage {v:.3f} V"
                )
                return
            self.instrument.source_voltage = v
            sleep(self.delay_between_points)
            current: float = self.instrument.measure_current()
            elapsed: float = time.time() - self._start_time
            status: str = "OK"
            self.emit('results', {
                "Channel": ch,
                "Voltage (V)": v,
                "Current (A)": current,
                "Time (s)": elapsed,
                "Status": status,
            })
            progress: float = 100.0 * step / total_steps
            self.emit("progress", progress)
            logger.debug(
                f"Step {step}/{total_steps}: "
                f"Ch={ch}, V={v:.3f} V, I={current:.6e} A, "
                f"Elapsed={elapsed:.2f}s, Progress={progress:.1f}%"
            )
            step += 1
        self.mux.deselect_channel(ch)
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
            self.instrument.close()
        except Exception:
            pass
        # Placeholder for appending metadata to the data file:
        #   [[PARAMS]]
        #   DeviceArea=...
        #   Sample thickness=...
        # Actual metadata insertion will be implemented later.