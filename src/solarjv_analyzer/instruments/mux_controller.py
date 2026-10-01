import logging
import serial
import time

logger = logging.getLogger(__name__)

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
            logger.debug(f"MUX already connected on {self.port}")
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
            logger.info(f"MUX connected on {self.port}")
        except serial.SerialException as e:
            logger.error(f"Failed to connect to MUX on {self.port}: {e}")
            raise

    def select_channel(self, channel: int):
        """
        Selects (turns ON) the specified channel (1–6).
        """
        logger.debug(f"Selecting channel {channel}")
        pixel = channel - 1
        command = f"AA010{pixel}000000BB"
        self._send_hex(command)

    def deselect_channel(self, channel: int):
        """
        Deselects (turns OFF) the specified channel.
        """
        logger.debug(f"Deselecting channel {channel}")
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
        try:
            self.ser.write(data)
        except serial.SerialException:
            # Windows returns ERROR_ACCESS_DENIED (WinError 5) from WriteFile
            # when the USB adapter behind an OPEN handle has gone away — a
            # pulled cable, a hub power event, USB selective suspend. pyserial
            # leaves `is_open` True after such a failure, so without closing
            # here the manager keeps believing the MUX is connected: the status
            # light stays green, the monitor never reconnects because the port
            # is "already open", and every later run fails identically until
            # the application is restarted.
            #
            # Closing (rather than dropping the object) is deliberate:
            # `is_mux_alive()` treats a missing `ser` attribute as alive, for
            # the simulated doubles, but reads `is_open` when one is present.
            logger.error(
                "MUX write failed on %s — the adapter appears to have "
                "disconnected. Closing the port so it can be reopened.",
                self.port,
            )
            try:
                self.ser.close()
            except Exception:                         # noqa: BLE001
                pass
            raise
        time.sleep(0.5)

    def close(self):
        """
        Closes the serial port connection.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("MUX connection closed.")