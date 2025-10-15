import serial
import time

class MuxController:
    """
    A controller for the real multiplexer device, using a hex protocol
    over a serial connection.
    """

    def __init__(self, port, baudrate=115200, timeout=1):
        """
        Initializes the multiplexer configuration.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        """
        Opens the serial port connection to the device.
        """
        if self.ser and self.ser.is_open:
            print(f"MUX already connected on {self.port}")
            return
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            print(f"✅ Mux connected on {self.port}")
        except serial.SerialException as e:
            print(f"❌ Failed to connect to MUX on {self.port}: {e}")
            raise

    def select_channel(self, channel: int):
        """
        Selects (turns ON) the specified channel (1–6).
        """
        print(f"Selecting channel {channel}")
        pixel = channel - 1
        command = f"AA010{pixel}000000BB"
        self._send_hex(command)

    def deselect_channel(self, channel: int):
        """
        Deselects (turns OFF) the specified channel.
        """
        print(f"Deselecting channel {channel}")
        pixel = channel - 1
        command = f"AA010{pixel}000100BB"
        self._send_hex(command)

    def _send_hex(self, hex_string: str):
        """
        Converts the hex string to bytes and writes it to the serial port.
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("MUX serial port not connected.")
        data = bytes.fromhex(hex_string)
        self.ser.write(data)
        time.sleep(0.5)

    def close(self):
        """
        Closes the serial port connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("✅ Mux connection closed.")