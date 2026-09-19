r"""Is the instrument's COM port actually there?

Both instruments reach the PC as USB-to-serial adapters — the MUX through a
CH340, the Keithley through a Prolific PL2303 into its RS-232 port — so
"connected" really means "that adapter is still enumerated by Windows".

This matters because the obvious checks lie. `pyserial`'s `is_open` stays True
after the USB device is yanked (it only learns otherwise on the next failed
I/O), and a PyVISA session handle stays valid the same way. A status light
driven by those would sit there green with the cable in someone's hand.

Enumerating ports is a pure read — no port is opened and no device is written
to — so it is safe to poll, and safe to run while a measurement is in progress.
"""

import logging
import re

logger = logging.getLogger(__name__)

_ASRL_RE = re.compile(r"ASRL(?:COM)?(\d+)::INSTR", re.IGNORECASE)


def list_ports() -> list:
    """Every serial port Windows currently enumerates.

    Returns a list of `ListPortInfo`; empty if pyserial cannot enumerate (which
    is treated as "unknown", never as "nothing is connected").
    """
    try:
        from serial.tools import list_ports as _lp
        return list(_lp.comports())
    except Exception as exc:
        logger.debug(f"Could not enumerate serial ports: {exc}")
        return []


def port_names() -> set:
    """Just the device names, upper-cased: {'COM3', 'COM4', ...}."""
    return {p.device.upper() for p in list_ports() if getattr(p, "device", None)}


def visa_to_port(address: str) -> str:
    """`ASRL3::INSTR` -> `COM3`. Empty for GPIB/USB/TCPIP addresses.

    The Keithley is reached over RS-232 through a USB adapter, so its VISA
    resource name carries the COM number — but only for ASRL resources. Any
    other transport has no COM port to look for, and callers must not treat
    that as "missing".
    """
    if not address:
        return ""
    match = _ASRL_RE.search(address.strip())
    return f"COM{match.group(1)}" if match else ""


def is_port_present(name: str, available: set = None) -> bool:
    """True if `name` ('COM4') is currently enumerated.

    An empty name means "not addressed by COM port" — reported as present, so
    a GPIB or USB-TMC Keithley is never shown as unplugged by this check.
    """
    if not name:
        return True
    available = port_names() if available is None else available
    return name.upper() in available


def describe(name: str) -> str:
    """Human description of a port ('USB-SERIAL CH340'), or '' if absent."""
    if not name:
        return ""
    for port in list_ports():
        if getattr(port, "device", "").upper() == name.upper():
            return getattr(port, "description", "") or ""
    return ""
