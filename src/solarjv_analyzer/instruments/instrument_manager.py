import logging
from typing import Union

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
        import pyvisa
        from pymeasure.adapters import VISAAdapter
        from pymeasure.instruments.keithley import Keithley2400

        rm = pyvisa.ResourceManager()
        adapter = VISAAdapter(rm.open_resource(address))
        instrument = Keithley2400(adapter)
        # reset and configure source and measurement
        instrument.reset()
        instrument.apply_voltage(compliance_current=0.1)
        instrument.measure_current()
        # test communication by reading ID
        _ = instrument.id
        log.info(f"Connected real Keithley2400 at {address}")
        return instrument
    except Exception as e:
        log.warning(f"Failed to connect to real Keithley2400 at {address}: {e}. Using FakeKeithley2400.")
        return FakeKeithley2400()
