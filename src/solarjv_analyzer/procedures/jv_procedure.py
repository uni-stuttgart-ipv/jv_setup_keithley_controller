import logging
from typing import List, Dict

from solarjv_analyzer.instruments.mux_controller import get_mux
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

    def startup(self) -> None:
        """
        Initialize instruments and record the start time.
        """
        logger.info("Startup: initializing Keithley instrument")
        #self.mux = get_mux(CONFIG.MUX_PORT)
        # Use the GPIB address provided by the GUI parameter
        #self.instrument = get_keithley(self.gpib_address, simulate=CONFIG.SIMULATION_MODE)
        self.instrument = get_keithley()
        self._start_time = time.time()
        # Configure the source-meter for voltage sourcing
        #self.instrument.apply_voltage(compliance_current=self.compliance_current)
        self.instrument.apply_voltage()
        self.instrument.enable_source()

    def execute(self) -> None:
        """
        Perform the J–V sweep across the selected channels,
        emitting results and progress updates.
        """
        logger.info("Beginning J–V sweep")
        voltages: np.ndarray = np.arange(
            self.start_voltage,
            self.stop_voltage + self.step_size,
            self.step_size
        )
        # Determine which channels are enabled
        channels: List[int] = [
            ch for ch in range(1, 7)
            if getattr(self, f"channel{ch}")
        ]
        total_steps: int = len(voltages) * len(channels) if channels else 1
        step: int = 0

        for ch in channels:
            logger.info(f"Measuring Channel {ch}")
            self.instrument.select_channel(ch)
            #self.mux.select_channel(ch)
            for v in voltages:
                if self.should_stop():
                    logger.warning(
                        f"Aborting sweep early at Channel {ch}, Voltage {v:.3f} V"
                    )
                    return
                # Source the set voltage and wait for settling
                self.instrument.source_voltage = v
                sleep(0.1)
                current: float = self.instrument.measure_current()
                elapsed: float = time.time() - self._start_time
                status: str = "OK"
                # Emit the measurement data and progress
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
        try:
            self.mux.close()
        except Exception:
            pass
        # Placeholder for appending metadata to the data file:
        #   [[PARAMS]]
        #   DeviceArea=...
        #   Sample thickness=...
        # Actual metadata insertion will be implemented later.