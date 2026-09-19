"""
Instrument manager – live hardware only.

The `simulation` parameter is kept for compatibility with existing calls
but is ignored. All connections are to real instruments.

Session ownership
-----------------
Exactly ONE ``InstrumentManager`` should exist per process: ``main.py``
creates it and passes the same object to the calibration window and the
main window. Copying an instrument *object* from one manager into another
used to leave two owners for one VISA session — whichever disconnected
first invalidated the handle the other was still holding, and the survivor
had no way to notice (see ``visa_session_open`` below).
"""

import logging

from .mux_controller import MuxController

log = logging.getLogger(__name__)


def visa_session_open(instrument) -> bool:
    """Return True when `instrument`'s VISA session is still usable.

    PyVISA invalidates a resource's ``session`` attribute on ``close()`` —
    reading it afterwards raises ``InvalidSession``. Probing it is free (no
    I/O, no instrument state change), unlike querying ``*IDN?``, so this is
    safe to call from the GUI thread and from status-light refreshes.

    A non-VISA instrument (test double, simulated stub) has no adapter
    connection to probe and is reported as usable.
    """
    if instrument is None:
        return False
    connection = getattr(getattr(instrument, "adapter", None), "connection", None)
    if connection is None:
        return True
    try:
        _ = connection.session
    except Exception:
        return False
    return True


def get_keithley(address: str):
    """
    Connect to a real Keithley 2400.

    Raises:
        Exception: if the connection or initialisation fails.
    """
    from pymeasure.adapters import VISAAdapter
    from pymeasure.instruments.keithley import Keithley2400

    adapter = VISAAdapter(address)
    try:
        instrument = Keithley2400(adapter)
        instrument.reset()
        instrument.apply_voltage(compliance_current=0.1)
        instrument.measure_current()
        _ = instrument.id
    except Exception:
        # Do not leak the VISA session (and, on serial transports, the COM
        # port) when initialisation fails part-way — a leaked session makes
        # the next connection attempt fail with "resource busy".
        try:
            adapter.close()
        except Exception:
            pass
        raise
    log.info(f"Connected real Keithley2400 at {address}")
    return instrument


class InstrumentManager:
    """
    Manages the lifecycle (connection, disconnection) of real instruments only.
    """

    def __init__(self):
        self.mux = None
        self.keithley = None

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------
    def _ports_in_use(self) -> set:
        """COM ports this manager currently holds open.

        Handed to the resolver so it never probes them: opening a busy port
        fails, which would be read as "not the Keithley" and could move a live
        connection onto a different port.
        """
        from .port_resolver import _visa_to_com

        in_use = set()
        port = getattr(getattr(self.mux, "ser", None), "port", None)
        if port:
            in_use.add(str(port).upper())
        if self.keithley is not None:
            connection = getattr(getattr(self.keithley, "adapter", None),
                                 "connection", None)
            name = getattr(connection, "resource_name", "") or ""
            com = _visa_to_com(name)
            if com:
                in_use.add(com.upper())
        return in_use

    def is_keithley_alive(self) -> bool:
        """True when a Keithley is held AND its VISA session is still open.

        Use this — never ``keithley is not None`` — for status lights and
        readiness gates: a closed session leaves the attribute set, so the
        plain None check reports "connected" for an unusable instrument.
        """
        return visa_session_open(self.keithley)

    def is_mux_alive(self) -> bool:
        """True when a MUX is held and its serial port is still open."""
        if self.mux is None:
            return False
        serial_port = getattr(self.mux, "ser", None)
        if serial_port is None:
            return True
        return bool(getattr(serial_port, "is_open", True))

    def _drop_stale_keithley(self) -> None:
        """Forget a Keithley whose session is already closed.

        Deliberately does NOT go through ``disconnect_keithley()``: that
        would attempt ``shutdown()`` writes on the dead session and bury the
        real cause under "Invalid session handle" errors.
        """
        if self.keithley is not None and not visa_session_open(self.keithley):
            log.warning("Keithley handle is stale (VISA session closed) — reconnecting.")
            self.keithley = None

    def _drop_stale_mux(self) -> None:
        """Forget a MUX whose serial port is already closed."""
        if self.mux is not None and not self.is_mux_alive():
            log.warning("MUX handle is stale (serial port closed) — reconnecting.")
            self.mux = None

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------
    def connect_mux(self, simulation=False):
        """
        Connect to the real multiplexer.

        Args:
            simulation: Ignored (kept for compatibility). Always uses real hardware.
        """
        self._drop_stale_mux()
        if self.mux:
            return
        # The port is resolved at startup, not read from config.py: config.py
        # ships inside the packaged app and COM numbers change on their own.
        # `active_mux_port()` falls back to config when nothing was resolved.
        from .port_resolver import active_mux_port, refresh_if_unconfirmed
        # If startup could only fall back to config.py, try again now — the
        # adapter may simply have been plugged in since. Enumeration only; it
        # never opens a port, so this is safe on the GUI thread.
        refresh_if_unconfirmed(busy_ports=self._ports_in_use())
        self.mux = MuxController(port=active_mux_port())
        self.mux.connect()

    def connect_keithley(self, simulation=False):
        """
        Connect to the real Keithley 2400.

        Args:
            simulation: Ignored (kept for compatibility). Always uses real hardware.
        """
        self._drop_stale_keithley()
        if self.keithley:
            return
        # Resolved at startup — see connect_mux().
        from .port_resolver import active_keithley_resource, refresh_if_unconfirmed
        refresh_if_unconfirmed(busy_ports=self._ports_in_use())   # see connect_mux()
        self.keithley = get_keithley(address=active_keithley_resource())

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
            if not visa_session_open(self.keithley):
                # Already closed (e.g. by another teardown path) — dropping
                # the reference is all that is left to do. Attempting
                # shutdown() here only logs a misleading InvalidSession error.
                log.info("Keithley session already closed — releasing handle.")
                self.keithley = None
                return
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
