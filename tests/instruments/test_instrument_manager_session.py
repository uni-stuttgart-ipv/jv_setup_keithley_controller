# tests/instruments/test_instrument_manager_session.py
"""
Session-liveness contract for InstrumentManager.

A closed VISA session leaves the `keithley` attribute set. Before this was
handled, `connect_keithley()`'s `if self.keithley: return` short-circuited on
such a dead handle and every sweep failed with
`pyvisa.errors.InvalidSession`, while the status light stayed green.
"""
import pytest

from solarjv_analyzer.instruments.instrument_manager import (
    InstrumentManager, visa_session_open,
)


class _FakeConnection:
    def __init__(self):
        self._open = True

    @property
    def session(self):
        if not self._open:
            raise RuntimeError("Invalid session handle. The resource might be closed.")
        return 1

    def close(self):
        self._open = False


class _FakeAdapter:
    def __init__(self):
        self.connection = _FakeConnection()

    def close(self):
        self.connection.close()


class _FakeKeithley:
    def __init__(self):
        self.adapter = _FakeAdapter()
        self.shutdown_calls = 0
        self.writes = []

    def write(self, cmd):
        _ = self.adapter.connection.session
        self.writes.append(cmd)

    def shutdown(self):
        self.shutdown_calls += 1
        _ = self.adapter.connection.session   # a real shutdown talks to it

    def close(self):
        self.adapter.close()


class _FakeSerial:
    def __init__(self):
        self.is_open = True

    def close(self):
        self.is_open = False


class _FakeMux:
    def __init__(self):
        self.ser = _FakeSerial()
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        self.ser.is_open = True

    def close(self):
        self.ser.close()


# ---------------------------------------------------------------------------
# visa_session_open
# ---------------------------------------------------------------------------

def test_session_open_detects_close():
    k = _FakeKeithley()
    assert visa_session_open(k) is True
    k.close()
    assert visa_session_open(k) is False


def test_session_open_on_none_and_on_non_visa_object():
    assert visa_session_open(None) is False
    # A test double / simulated stub with no adapter is treated as usable.
    assert visa_session_open(object()) is True


# ---------------------------------------------------------------------------
# Liveness reporting
# ---------------------------------------------------------------------------

def test_is_keithley_alive_tracks_the_session_not_the_attribute():
    mgr = InstrumentManager()
    assert mgr.is_keithley_alive() is False

    k = _FakeKeithley()
    mgr.keithley = k
    assert mgr.is_keithley_alive() is True

    k.close()
    assert mgr.keithley is not None, "precondition: the handle is still set"
    assert mgr.is_keithley_alive() is False, (
        "a closed session must not be reported as connected"
    )


def test_is_mux_alive_tracks_the_serial_port():
    mgr = InstrumentManager()
    assert mgr.is_mux_alive() is False
    mgr.mux = _FakeMux()
    assert mgr.is_mux_alive() is True
    mgr.mux.close()
    assert mgr.is_mux_alive() is False


# ---------------------------------------------------------------------------
# Self-healing connect
# ---------------------------------------------------------------------------

def test_connect_keithley_reconnects_a_stale_handle(monkeypatch):
    mgr = InstrumentManager()
    dead = _FakeKeithley()
    dead.close()
    mgr.keithley = dead

    fresh = _FakeKeithley()
    monkeypatch.setattr(
        "solarjv_analyzer.instruments.instrument_manager.get_keithley",
        lambda address: fresh,
    )

    mgr.connect_keithley()

    assert mgr.keithley is fresh, "stale handle was not replaced"
    assert mgr.is_keithley_alive()
    assert dead.shutdown_calls == 0, (
        "the dead handle must be dropped, not shut down — talking to it only "
        "logs a misleading InvalidSession error"
    )


def test_connect_keithley_keeps_a_live_handle(monkeypatch):
    mgr = InstrumentManager()
    live = _FakeKeithley()
    mgr.keithley = live

    def _should_not_connect(address):
        raise AssertionError("reconnected despite a healthy session")

    monkeypatch.setattr(
        "solarjv_analyzer.instruments.instrument_manager.get_keithley",
        _should_not_connect,
    )
    mgr.connect_keithley()
    assert mgr.keithley is live


def test_connect_mux_reconnects_a_closed_port(monkeypatch):
    mgr = InstrumentManager()
    dead = _FakeMux()
    dead.close()
    mgr.mux = dead

    fresh = _FakeMux()

    monkeypatch.setattr(
        "solarjv_analyzer.instruments.instrument_manager.MuxController",
        lambda port: fresh,
    )
    mgr.connect_mux()
    assert mgr.mux is fresh, "closed port was not replaced"
    assert fresh.connect_calls == 1
    assert mgr.is_mux_alive()


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------

def test_disconnect_keithley_on_a_closed_session_just_releases_the_handle():
    """Double-teardown must not raise or log InvalidSession noise."""
    mgr = InstrumentManager()
    k = _FakeKeithley()
    mgr.keithley = k
    k.close()          # someone else already closed it

    mgr.disconnect_keithley()

    assert mgr.keithley is None
    assert k.shutdown_calls == 0


def test_disconnect_keithley_shuts_down_a_live_session():
    mgr = InstrumentManager()
    k = _FakeKeithley()
    mgr.keithley = k

    mgr.disconnect_keithley()

    assert k.shutdown_calls == 1
    assert visa_session_open(k) is False
    assert mgr.keithley is None


def test_get_keithley_closes_the_adapter_when_init_fails(monkeypatch):
    """A half-open session would hold the COM port against the next retry."""
    import solarjv_analyzer.instruments.instrument_manager as im

    closed = []

    class _Adapter:
        def close(self):
            closed.append(True)

    class _FakeVISAAdapter:
        def __new__(cls, address):
            return _Adapter()

    class _Boom:
        def __init__(self, adapter):
            raise OSError("VI_ERROR_RSRC_BUSY")

    fake_adapters = type("m", (), {"VISAAdapter": _FakeVISAAdapter})
    fake_keithley_mod = type("m", (), {"Keithley2400": _Boom})
    monkeypatch.setitem(__import__("sys").modules, "pymeasure.adapters", fake_adapters)
    monkeypatch.setitem(
        __import__("sys").modules, "pymeasure.instruments.keithley", fake_keithley_mod
    )

    with pytest.raises(OSError):
        im.get_keithley("ASRL3::INSTR")

    assert closed, "the VISA adapter leaked after a failed initialisation"
