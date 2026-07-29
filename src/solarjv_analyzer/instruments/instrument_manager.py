"""
Instrument manager – live hardware only.

The `simulation` parameter is kept for compatibility with existing calls
but is ignored. All connections are to real instruments.
"""

import logging

from .mux_controller import MuxController
from solarjv_analyzer import config

log = logging.getLogger(__name__)


def get_keithley(address: str):
    """
    Connect to a real Keithley 2400.

    Raises:
        Exception: if the connection or initialisation fails.
    """
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


class InstrumentManager:
    """
    Manages the lifecycle (connection, disconnection) of real instruments only.
    """

    def __init__(self):
        self.mux = None
        self.keithley = None

    def connect_mux(self, simulation=False):
        """
        Connect to the real multiplexer.

        Args:
            simulation: Ignored (kept for compatibility). Always uses real hardware.
        """
        if self.mux:
            return
        # No fallback; always real.
        self.mux = MuxController(port=config.MUX_PORT)
        self.mux.connect()

    def connect_keithley(self, simulation=False):
        """
        Connect to the real Keithley 2400.

        Args:
            simulation: Ignored (kept for compatibility). Always uses real hardware.
        """
        if self.keithley:
            return
        # No fallback; always real.
        self.keithley = get_keithley(address=config.GPIB_ADDRESS)

    def disconnect_mux(self):
        """Disconnect the multiplexer and release the serial port."""
        if self.mux:
            try:
                self.mux.close()
            except Exception as e:
                log.error(f"Error disconnecting MUX: {e}")
            self.mux = None

    def disconnect_keithley(self):
        """Disconnect the Keithley and release the VISA resource."""
        if self.keithley:
            try:
                if hasattr(self.keithley, "shutdown"):
                    self.keithley.shutdown()
                # Close the VISA adapter to free the port
                if hasattr(self.keithley, "close"):
                    self.keithley.close()
                elif hasattr(self.keithley, "adapter") and hasattr(self.keithley.adapter, "close"):
                    self.keithley.adapter.close()
            except Exception as e:
                log.error(f"Error disconnecting Keithley: {e}")
            self.keithley = None