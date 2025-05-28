import logging
import serial
from serial.serialutil import SerialException

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

class MuxController:
    def __init__(self, port: str = "COM3", baud: int = 9600):
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=1)
            log.info(f"Opened MUX on {port}@{baud}")
        except SerialException as e:
            log.error(f"SerialException: Could not open MUX on {port}: {e}")
            raise

    def select_channel(self, channel: int) -> None:
        # translate channel to your device’s hex command
        cmd = f"{channel:02X}\n".encode()
        self.ser.write(cmd)
        log.debug(f"MUX -> select {channel}")

    def close(self) -> None:
        self.ser.close()
        log.info("MUX connection closed")


# Simulated Mux for fallback
class SimulatedMux:
    def __init__(self, port: str = "", baud: int = 0):
        log.warning(f"Simulated MUX init for port {port}@{baud}")
    def select_channel(self, channel: int) -> None:
        log.info(f"Simulated MUX -> select {channel}")
    def close(self) -> None:
        log.info("Simulated MUX closed")

def get_mux(port: str = "COM3") -> MuxController:
    try:
        mux = MuxController(port)
        return mux
    except Exception as e:
        log.error(f"Could not open MUX on {port}: {e}. Falling back to SimulatedMux.")
        return SimulatedMux(port)