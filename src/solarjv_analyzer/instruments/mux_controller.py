import serial
import time

class MuxController:
    """
    A controller for the multiplexer device, using the proven hex protocol.
    """

    def __init__(self, port, baudrate=115200, timeout=1):
        """
        Initialize the multiplexer connection.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        """
        Open the serial port.
        """
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout
        )
        print(f"✅ Mux connected on {self.port}")

    def select_channel(self, channel: int):
        """
        Select (turn ON) the specified channel (0–5).
        """
        print(f"Selecting channel {channel}")
        command = f"AA010{channel}000000BB"
        self._send_hex(command)

    def deselect_channel(self, channel: int):
        """
        Deselect (turn OFF) the specified channel.
        """
        print(f"Deselecting channel {channel}")
        command = f"AA010{channel}000100BB"
        self._send_hex(command)

    def _send_hex(self, hex_string):
        """
        Convert the hex string to bytes and write it.
        """
        if not self.ser:
            raise RuntimeError("MUX serial port not connected.")
        data = bytes.fromhex(hex_string)
        self.ser.write(data)
        time.sleep(0.5)

    def close(self):
        """
        Close the serial port.
        """
        if self.ser:
            self.ser.close()
            print("✅ Mux connection closed.")

class SimulatedMux:
    def select_channel(self, channel):
        print(f"[SimulatedMux] Pretend selecting channel {channel}")

    def deselect_channel(self, channel):
        print(f"[SimulatedMux] Pretend deselecting channel {channel}")

    def close(self):
        print("[SimulatedMux] Pretend closing connection")