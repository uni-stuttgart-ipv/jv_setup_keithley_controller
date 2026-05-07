import logging
from typing import Union
from .mux_controller import MuxController
from solarjv_analyzer import config

log = logging.getLogger(__name__)

def get_keithley(address: str):
    """
    Factory function to connect to a real Keithley 2400, falling back to
    a simulated version on failure.
    """
    try:
        from pymeasure.adapters import VISAAdapter
        from pymeasure.instruments.keithley import Keithley2400
        
        adapter = VISAAdapter(address)
        instrument = Keithley2400(adapter)
        instrument.reset()
        instrument.apply_voltage(compliance_current=0.1)
        instrument.measure_current()
        _ = instrument.id
        log.info(f"Connected real Keithley2400 at {address}")
        return instrument
    except Exception as e:
        # Import the simulated class from its new location
        from .simulated.simulated_keithley import SimulatedKeithley2400
        log.warning(f"Failed to connect to real Keithley2400: {e}. Using simulated version.")
        return SimulatedKeithley2400()

class InstrumentManager:
    """
    Manages the lifecycle (connection, disconnection) of all instruments
    used in the application, handling both real and simulated hardware.
    """
    def __init__(self):
        self.mux = None
        self.keithley = None

    def connect_mux(self, simulation=False):
        """Connects to the MUX, choosing between real or simulated."""
        if self.mux:
            return
        if simulation:
            # Import the simulated class from its new location
            from .simulated.simulated_mux import SimulatedMux
            self.mux = SimulatedMux()
            self.mux.connect()
        else:
            self.mux = MuxController(port=config.MUX_PORT)
            self.mux.connect()

    def connect_keithley(self, simulation=False):
        """Connects to the Keithley SMU, choosing between real or simulated."""
        if self.keithley:
            return
        if simulation:
            # Import and instantiate the simulated class from its new location
            from .simulated.simulated_keithley import SimulatedKeithley2400
            self.keithley = SimulatedKeithley2400()
            log.info("Connected SimulatedKeithley2400.")
        else:
            self.keithley = get_keithley(address=config.GPIB_ADDRESS)

    def disconnect_mux(self):
        """Disconnects the MUX if it is connected."""
        if self.mux:
            self.mux.close()
            self.mux = None

    def disconnect_keithley(self):
        """Disconnects the Keithley SMU if it is connected."""
        if self.keithley and hasattr(self.keithley, "shutdown"):
            self.keithley.shutdown()
        self.keithley = None