import logging
from typing import Union
from .mux_controller import MuxController
from solarjv_analyzer import config

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())
class FakeKeithley2400:
    def __init__(self):
        self._voltage = 0.0
        self._channel = 1  # Default channel

    def select_channel(self, ch):
        self._channel = ch

    def apply_voltage(self): pass
    def enable_source(self): pass

    def measure_current(self):
        return self.current

    @property
    def id(self):
        return "FAKE-KEITHLEY-2400"

    def shutdown(self):
        print("FakeKeithley2400: shutdown called.")

    @property
    def current(self):
        import numpy as np
        noise = np.random.normal(0, 0.001)
        scaling = 1 + (self._channel - 1) * 0.1
        return scaling * (0.1 * self._voltage) + noise

    @property
    def source_voltage(self): return self._voltage
    @source_voltage.setter
    def source_voltage(self, value): self._voltage = value

def get_keithley(address: str = "GPIB::24") -> Union[FakeKeithley2400, "Keithley2400"]:
    try:
        from pymeasure.adapters import VISAAdapter
        from pymeasure.instruments.keithley import Keithley2400
        import logging
        log = logging.getLogger(__name__)

        # Give VISAAdapter the resource *string* (not a pyvisa session)
        adapter = VISAAdapter(address, preprocess_reply=lambda s: s.strip())

        # If connecting over serial (ASRL...), apply Keithley 2400 RS‑232 defaults
        if address.upper().startswith("ASRL"):
            try:
                from pyvisa.constants import StopBits, Parity, FlowControl
                res = adapter.connection  # underlying pyvisa Resource
                res.baud_rate = 9600
                res.data_bits = 8
                res.stop_bits = StopBits.one
                res.parity = Parity.none
                res.flow_control = FlowControl.none
                res.write_termination = "\n"
                res.read_termination = "\n"
                res.timeout = 10000  # ms
            except Exception as e:
                log.warning(f"Opened {address} but failed to apply serial settings: {e}")

        instrument = Keithley2400(adapter)
        # Basic bring-up
        instrument.reset()
        try:
            instrument.beep(2400, 0.15)  # audible confirmation, if supported
        except Exception:
            pass
        instrument.apply_voltage(compliance_current=0.1)
        instrument.measure_current()
        _ = instrument.id  # verify comms
        log.info(f"Connected real Keithley2400 at {address}")
        return instrument
    except Exception as e:
        log.warning(f"Failed to connect to real Keithley2400 at {address}: {e}. Using FakeKeithley2400.")
        return FakeKeithley2400()

class InstrumentManager:
    def __init__(self):
        self.mux = None
        self.keithley = None

    def connect_mux(self, simulation=False):
        if self.mux:
            print("MUX already connected.")
            return
        if simulation:
            from .mux_controller import SimulatedMux
            self.mux = SimulatedMux()
            print("Simulated MUX is connected.")
        else:
            self.mux = MuxController(port=config.MUX_PORT)
            self.mux.connect()
            print(f"MUX connected on {config.MUX_PORT}")

    def disconnect_mux(self):
        if self.mux:
            self.mux.close()
            self.mux = None

    def disconnect_keithley(self):
        if self.keithley is None:
            return
        try:
            if hasattr(self.keithley, "shutdown"):
                try:
                    self.keithley.shutdown()
                except Exception:
                    pass
            # Ensure the underlying VISA resource is closed
            if hasattr(self.keithley, "adapter") and hasattr(self.keithley.adapter, "connection"):
                try:
                    self.keithley.adapter.connection.close()
                except Exception:
                    pass
        finally:
            self.keithley = None

    def connect_keithley(self, simulation=False):
        if simulation:
            self.keithley = FakeKeithley2400()
            return self.keithley
        self.keithley = get_keithley(address=config.GPIB_ADDRESS)
        return self.keithley
