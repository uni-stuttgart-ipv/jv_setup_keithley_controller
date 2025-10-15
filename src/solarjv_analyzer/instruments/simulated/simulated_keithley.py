import numpy as np

class SimulatedKeithley2400:
    """
    A simulated version of the Keithley 2400 SourceMeter for testing
    the application without physical hardware.
    
    This version includes internal channel simulation to modify its output,
    matching the logic from the original file.
    """
    def __init__(self):
        self._voltage = 0.0
        self._channel = 1  # Default channel

    def select_channel(self, ch):
        """Simulates selecting a channel to modify the output signal."""
        self._channel = ch

    def apply_voltage(self):
        """Simulates setting the instrument to apply voltage."""
        pass

    def enable_source(self):
        """Simulates enabling the instrument's output."""
        pass

    @property
    def id(self):
        """Returns a simulated instrument ID."""
        return "SIMULATED-KEITHLEY-2400"

    def shutdown(self):
        """Simulates shutting down the instrument."""
        print("SimulatedKeithley2400: shutdown called.")

    def measure_current(self):
        """Returns a simulated current value based on the set voltage."""
        return self.current

    @property
    def current(self):
        """
        Calculates a simulated current. The output is scaled based on the
        selected channel to simulate measurements on different devices.
        """
        noise = np.random.normal(0, 0.001)
        # Scaling factor changes based on the channel to produce different curves
        scaling = 1 + (self._channel - 1) * 0.1
        return scaling * (0.1 * self._voltage) + noise

    @property
    def source_voltage(self):
        """Gets the currently set source voltage."""
        return self._voltage

    @source_voltage.setter
    def source_voltage(self, value):
        """Sets the source voltage."""
        self._voltage = value