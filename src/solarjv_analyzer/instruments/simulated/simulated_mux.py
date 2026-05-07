class SimulatedMux:
    """
    A simulated version of the MuxController for testing the application
    without physical hardware.
    
    It prints its actions to the console instead of communicating over a
    serial port.
    """
    def connect(self):
        """Simulates connecting to the MUX."""
        print("Simulated MUX is connected.")

    def select_channel(self, channel):
        """Simulates selecting a channel."""
        print(f"[SimulatedMux] Pretend selecting channel {channel}")

    def deselect_channel(self, channel):
        """Simulates deselecting a channel."""
        print(f"[SimulatedMux] Pretend deselecting channel {channel}")

    def close(self):
        """Simulates closing the connection."""
        print("[SimulatedMux] Pretend closing connection")