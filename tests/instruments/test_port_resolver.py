"""Startup port resolution.

`config.py` ships inside the packaged app, so a COM number that changes makes
the app unusable until someone rebuilds it. The resolver decides both ports at
startup instead, and remembers what it found.

The two instruments are separable because they use different adapter chips —
CH340 for the MUX, Prolific PL2303 for the Keithley — but the evidence is not
symmetric: the Keithley can be *proved* with `*IDN?`, while the MUX protocol
has no reply at all (audit S7), so its identification is an inference from the
USB descriptor. These tests pin the consequences of that asymmetry.
"""
import json
import os

import pytest

from solarjv_analyzer.instruments import port_resolver as pr


class FakePort:
    """Stands in for pyserial's ListPortInfo."""

    def __init__(self, device, vid=None, pid=None, serial_number=None,
                 location=None, description="", manufacturer=""):
        self.device = device
        self.vid = vid
        self.pid = pid
        self.serial_number = serial_number
        self.location = location
        self.description = description
        self.manufacturer = manufacturer
        self.hwid = f"USB VID:PID={vid or 0:04X}:{pid or 0:04X}"


def mux_port(device="COM5", **kwargs):
    kwargs.setdefault("vid", 0x1A86)
    kwargs.setdefault("pid", 0x7523)
    kwargs.setdefault("description", "USB-SERIAL CH340")
    return FakePort(device, **kwargs)


def keithley_port(device="COM3", **kwargs):
    kwargs.setdefault("vid", 0x067B)
    kwargs.setdefault("pid", 0x23A3)
    kwargs.setdefault("description", "Prolific PL2303GT USB Serial COM Port")
    return FakePort(device, **kwargs)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    """No real ports, no real settings file, no cached resolution."""
    pr.reset_for_tests()
    monkeypatch.setattr(pr, "settings_path",
                        lambda: str(tmp_path / ".solarjv" / "config.json"))
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda device, timeout=None: (False, ""))
    yield
    pr.reset_for_tests()


def resolve(ports, settings=None, probe=True, **kwargs):
    kwargs.setdefault("config_mux", "COM4")
    kwargs.setdefault("config_keithley", "ASRL3::INSTR")
    return pr.resolve(ports=ports, settings=settings or {}, probe=probe, **kwargs)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device, resource", [
    ("COM3", "ASRL3::INSTR"), ("COM12", "ASRL12::INSTR"), ("nonsense", ""),
])
def test_com_to_visa(device, resource):
    assert pr.com_to_visa(device) == resource


def test_signatures_separate_the_two_chips():
    assert pr.matches_signature(mux_port(), pr.MUX_VIDS, pr.MUX_KEYWORDS)
    assert not pr.matches_signature(
        keithley_port(), pr.MUX_VIDS, pr.MUX_KEYWORDS)
    assert pr.matches_signature(
        keithley_port(), pr.KEITHLEY_VIDS, pr.KEITHLEY_KEYWORDS)


def test_signature_matches_on_description_when_the_vid_is_unknown():
    """Chip revisions change the PID and sometimes the VID; the Windows
    description does not."""
    odd = FakePort("COM9", vid=0x9999, description="USB-SERIAL CH340")
    assert pr.matches_signature(odd, pr.MUX_VIDS, pr.MUX_KEYWORDS)


# ---------------------------------------------------------------------------
# MUX resolution
# ---------------------------------------------------------------------------

def test_the_mux_is_found_by_its_chip(monkeypatch):
    result = resolve([mux_port("COM5"), keithley_port("COM3")])
    assert (result.mux_port, result.mux_source) == ("COM5", "detected")


def test_a_moved_mux_is_followed_by_serial_number():
    """The whole point: replugged into another socket, found at its new COM."""
    settings = {"mux_identity": {"vid": 0x1A86, "pid": 0x7523,
                                 "serial": "ABC123"}}
    ports = [mux_port("COM11", serial_number="ABC123")]
    result = resolve(ports, settings)
    assert (result.mux_port, result.mux_source) == ("COM11", "remembered")


def test_two_ch340s_are_never_guessed_between():
    """The MUX protocol has no identify command, so there is nothing to ask.

    Driving the wrong board would switch relays on unknown hardware — refusing
    is the only safe answer.
    """
    result = resolve([mux_port("COM5"), mux_port("COM6"), keithley_port()])
    assert result.mux_port == ""
    assert any("more than one CH340" in p for p in result.problems)


def test_two_ch340s_are_fine_once_one_is_remembered():
    settings = {"mux_identity": {"vid": 0x1A86, "pid": 0x7523, "serial": "B"}}
    ports = [mux_port("COM5", serial_number="A"),
             mux_port("COM6", serial_number="B")]
    assert resolve(ports, settings).mux_port == "COM6"


def test_a_pinned_port_is_never_silently_replaced():
    """If the operator pinned a port and it is gone, that is an error — not an
    invitation to use a different device."""
    settings = {"mux_pinned": True, "mux_port": "COM7"}
    result = resolve([mux_port("COM5")], settings)
    assert result.mux_port == ""
    assert any("pinned" in p for p in result.problems)


def test_config_is_the_fallback_when_nothing_is_recognised():
    plain = FakePort("COM4", description="Standard Serial over Bluetooth")
    result = resolve([plain])
    assert (result.mux_port, result.mux_source) == ("COM4", "config")


# ---------------------------------------------------------------------------
# Keithley resolution
# ---------------------------------------------------------------------------

def test_a_lone_prolific_adapter_is_probed_for_confirmation(monkeypatch):
    """The common case: the rack is powered on before the app is launched.

    A single PL2303 adapter is the Keithley by elimination, but we still ask
    `*IDN?` so the log records which instrument answered.
    """
    asked = []

    def fake_probe(device, timeout=None):
        asked.append(device)
        return (True, "KEITHLEY INSTRUMENTS INC.,MODEL 2400,1234,C32")

    monkeypatch.setattr(pr, "probe_is_keithley", fake_probe)
    result = resolve([mux_port("COM5"), keithley_port("COM3")])

    assert asked == ["COM3"]
    assert result.keithley_resource == "ASRL3::INSTR"
    assert result.keithley_source == "detected"
    assert result.problems == []


def test_a_lone_silent_prolific_adapter_still_resolves(monkeypatch):
    """The reply verifies; it does not gate.

    If the instrument is off or set to another baud rate the adapter stays
    silent — we still hand the address to the app, with a warning, rather than
    leaving the Keithley unresolved.
    """
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: (False, ""))
    result = resolve([mux_port("COM5"), keithley_port("COM3")])

    assert result.keithley_resource == "ASRL3::INSTR"
    assert result.keithley_source == "detected"
    assert any("did not answer" in problem for problem in result.problems)


def test_a_lone_prolific_adapter_resolves_without_a_probe_available(monkeypatch):
    """`probe=False` (the refresh path) must never open a port."""
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: pytest.fail(f"probed {d}"))
    result = resolve([mux_port("COM5"), keithley_port("COM3")], probe=False)

    assert result.keithley_resource == "ASRL3::INSTR"
    assert result.keithley_source == "detected"


def test_probing_disambiguates_two_prolific_adapters(monkeypatch):
    answered = []

    def fake_probe(device, timeout=None):
        answered.append(device)
        return (device == "COM6", "KEITHLEY INSTRUMENTS INC.,MODEL 2400,1234,C32")

    monkeypatch.setattr(pr, "probe_is_keithley", fake_probe)
    result = resolve([keithley_port("COM3"), keithley_port("COM6")])

    assert result.keithley_resource == "ASRL6::INSTR"
    assert set(answered) >= {"COM3", "COM6"}


def test_two_prolific_adapters_and_a_powered_off_instrument(monkeypatch):
    """Nothing answers: say so, rather than picking one at random."""
    monkeypatch.setattr(pr, "probe_is_keithley", lambda d, timeout=None: (False, ""))
    result = resolve([keithley_port("COM3"), keithley_port("COM6")])

    assert result.keithley_resource in ("", "ASRL3::INSTR")
    assert any("more than one PL2303" in p for p in result.problems)


def test_the_mux_port_is_never_probed(monkeypatch):
    """`*IDN?` is meaningless to a board expecting binary hex frames, and
    could be read as one."""
    probed = []

    def fake_probe(device, timeout=None):
        probed.append(device)
        return False, ""

    monkeypatch.setattr(pr, "probe_is_keithley", fake_probe)
    resolve([mux_port("COM5"), FakePort("COM8", description="Some adapter")])

    assert "COM5" not in probed, "probed the MUX"


def test_prolific_ports_are_probed_before_anything_else(monkeypatch):
    """The PL2303 adapters are the candidates, so they are asked first; other
    adapters are only reached if none of them answers."""
    order = []

    def fake_probe(device, timeout=None):
        order.append(device)
        return False, ""

    monkeypatch.setattr(pr, "probe_is_keithley", fake_probe)
    resolve([FakePort("COM8", description="Some other adapter"),
             keithley_port("COM3"), keithley_port("COM6")])

    assert order[:2] == ["COM3", "COM6"], f"probe order was {order}"
    assert "COM8" in order, "the fallback scan should still reach other ports"


def test_a_non_serial_resource_is_used_as_given(monkeypatch):
    """A GPIB or USB-TMC Keithley has no COM port to search for."""
    probed = []
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: probed.append(d) or (False, ""))
    result = resolve([mux_port()], config_keithley="GPIB0::24::INSTR")

    assert result.keithley_resource == "GPIB0::24::INSTR"
    assert probed == [], "probed serial ports for a GPIB instrument"


def test_probing_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: pytest.fail("probed"))
    resolve([mux_port(), keithley_port()], probe=False)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_what_was_learnt_is_remembered_and_reused(monkeypatch):
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: (d == "COM3", "KEITHLEY 2400"))
    first = resolve([mux_port("COM5", serial_number="S1"), keithley_port("COM3")])
    pr.apply(first)

    saved = pr.load_hardware_settings()
    assert saved["mux_port"] == "COM5"
    assert saved["mux_identity"]["serial"] == "S1"
    assert saved["keithley_resource"] == "ASRL3::INSTR"

    # ...and a later start finds the same device at a new COM number without
    # probing anything.
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: pytest.fail("should not probe"))
    again = pr.resolve(ports=[mux_port("COM9", serial_number="S1")],
                       settings=saved, probe=True,
                       config_mux="COM4", config_keithley="ASRL3::INSTR")
    assert (again.mux_port, again.mux_source) == ("COM9", "remembered")


def test_settings_live_beside_the_other_user_preferences(tmp_path):
    """Namespaced under `hardware` so it cannot collide with `base_directory`
    or the store's `store_extra_copy` in the same file."""
    os.makedirs(os.path.dirname(pr.settings_path()), exist_ok=True)
    with open(pr.settings_path(), "w", encoding="utf-8") as handle:
        json.dump({"base_directory": "D:/Work", "store_extra_copy": "E:/Copy"},
                  handle)

    pr.save_hardware_settings({"mux_port": "COM5"})

    with open(pr.settings_path(), encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["base_directory"] == "D:/Work"
    assert data["store_extra_copy"] == "E:/Copy"
    assert data["hardware"]["mux_port"] == "COM5"


# ---------------------------------------------------------------------------
# the active values
# ---------------------------------------------------------------------------

def test_active_values_fall_back_to_config_until_resolution_happens():
    from solarjv_analyzer import config
    assert pr.active_mux_port() == config.MUX_PORT
    assert pr.active_keithley_resource() == config.GPIB_ADDRESS


def test_active_values_follow_the_resolution():
    result = resolve([mux_port("COM5"), keithley_port("COM3")], probe=False)
    result.keithley_resource = "ASRL7::INSTR"
    pr.apply(result, remember=False)

    assert pr.active_mux_port() == "COM5"
    assert pr.active_keithley_resource() == "ASRL7::INSTR"


# ---------------------------------------------------------------------------
# Powered-on-late: the calibration checklist tells operators to switch the
# instruments on AFTER launching the app, so startup resolution routinely runs
# against hardware that cannot answer yet.
# ---------------------------------------------------------------------------

def test_a_fallback_result_is_not_treated_as_confirmed():
    result = resolve([FakePort("COM4", description="Generic")], probe=False)
    pr.apply(result, remember=False)
    assert result.mux_source == "config"
    assert pr.mux_confirmed() is False


def test_detected_and_remembered_results_are_confirmed():
    result = resolve([mux_port("COM5"), keithley_port("COM3")], probe=False)
    pr.apply(result, remember=False)
    assert pr.mux_confirmed() is True


def test_refresh_retries_once_the_instrument_is_powered_on(monkeypatch):
    """The scenario this exists for: nothing answers at startup, the operator
    powers the rack on, and the next connect attempt finds it."""
    # Startup: the Keithley is off, so *IDN? gets no reply anywhere.
    monkeypatch.setattr(pr, "_enumerate", lambda: [mux_port("COM5")])
    monkeypatch.setattr(pr, "probe_is_keithley", lambda d, timeout=None: (False, ""))
    monkeypatch.setattr(pr, "load_hardware_settings", lambda: {})
    pr.apply(pr.resolve(probe=True), remember=False)
    assert pr.keithley_confirmed() is False

    # Operator powers it on; the next connect attempt re-resolves.
    monkeypatch.setattr(pr, "_enumerate",
                        lambda: [mux_port("COM5"), keithley_port("COM7")])
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: (d == "COM7", "KEITHLEY,MODEL 2400"))
    monkeypatch.setattr(pr, "save_hardware_settings", lambda data: None)

    pr.refresh_if_unconfirmed()

    assert pr.active_keithley_resource() == "ASRL7::INSTR"
    assert pr.keithley_confirmed() is True


def test_refresh_does_nothing_once_everything_is_confirmed(monkeypatch):
    """Idempotent: no probing, no port opening, on every later connect."""
    monkeypatch.setattr(pr, "_enumerate",
                        lambda: [mux_port("COM5"), keithley_port("COM3")])
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: (d == "COM3", "KEITHLEY 2400"))
    monkeypatch.setattr(pr, "load_hardware_settings", lambda: {})
    monkeypatch.setattr(pr, "save_hardware_settings", lambda data: None)
    pr.apply(pr.resolve(probe=True), remember=False)
    assert pr.keithley_confirmed() and pr.mux_confirmed()

    monkeypatch.setattr(pr, "_enumerate",
                        lambda: pytest.fail("re-enumerated when already confirmed"))
    assert pr.refresh_if_unconfirmed() is None


# ---------------------------------------------------------------------------
# Review findings, 2026-09-19. Each of these is a bug that was in the first
# version of the resolver.
# ---------------------------------------------------------------------------

def test_refresh_never_opens_a_port(monkeypatch):
    """`refresh_if_unconfirmed()` runs inside connect_*(), which is called on
    the GUI thread — from CalibrationWindow.__init__ among others. Opening a
    serial port there breaks the project's rule against serial I/O on the GUI
    thread, and a ghost COM port can block for seconds."""
    monkeypatch.setattr(pr, "_enumerate", lambda: [mux_port("COM5")])
    monkeypatch.setattr(pr, "load_hardware_settings", lambda: {})
    monkeypatch.setattr(pr, "save_hardware_settings", lambda data: None)
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: pytest.fail(f"opened {d}"))

    pr.apply(pr.resolve(ports=[], probe=False), remember=False)
    pr.refresh_if_unconfirmed()          # must not probe


def test_a_port_we_already_hold_is_never_probed(monkeypatch):
    """Opening a busy port fails, which would read as "not the Keithley" and
    could move a live connection onto a different port."""
    probed = []
    monkeypatch.setattr(pr, "probe_is_keithley",
                        lambda d, timeout=None: probed.append(d) or (False, ""))

    resolve([FakePort("COM3", description="Unknown adapter"),
             FakePort("COM8", description="Another adapter")],
            busy_ports={"COM3"})

    assert "COM3" not in probed


def test_a_refresh_cannot_downgrade_a_good_resolution(monkeypatch):
    """The MUX was identified at startup and has since been unplugged. A later
    refresh must not replace a known-good port with nothing — and must not
    persist that."""
    monkeypatch.setattr(pr, "save_hardware_settings", lambda data: None)
    good = resolve([mux_port("COM5"), keithley_port("COM3")], probe=False)
    pr.apply(good, remember=False)
    assert pr.active_mux_port() == "COM5"

    monkeypatch.setattr(pr, "_enumerate", lambda: [])       # everything unplugged
    monkeypatch.setattr(pr, "load_hardware_settings", lambda: {})
    pr.apply(pr.resolve(probe=False), remember=False, upgrade_only=True)

    assert pr.active_mux_port() == "COM5", "a known-good port was thrown away"


def test_a_refresh_still_fills_an_empty_slot(monkeypatch):
    """The upgrade-only rule must not block the case it exists to serve."""
    monkeypatch.setattr(pr, "save_hardware_settings", lambda data: None)
    pr.apply(resolve([], probe=False, config_mux="COM99",
                     config_keithley="ASRL99::INSTR"), remember=False)
    assert pr.active_mux_port() in ("", "COM4")

    monkeypatch.setattr(pr, "_enumerate", lambda: [mux_port("COM5")])
    monkeypatch.setattr(pr, "load_hardware_settings", lambda: {})
    pr.apply(pr.resolve(probe=False), remember=False, upgrade_only=True)

    assert pr.active_mux_port() == "COM5"


def test_the_mux_keyword_list_is_not_over_broad():
    """"usb-serial" was in this list. Plenty of unrelated adapters describe
    themselves that way, and a false match makes the MUX look *ambiguous*,
    which refuses to resolve — a worse outcome than the mis-identification the
    ambiguity check exists to prevent."""
    innocent = FakePort("COM8", vid=0x0403, pid=0x6001,
                        description="USB Serial Port (FTDI)")
    assert not pr.matches_signature(innocent, pr.MUX_VIDS, pr.MUX_KEYWORDS)

    result = resolve([mux_port("COM5"), innocent])
    assert result.mux_port == "COM5"
    assert not [p for p in result.problems if "CH340" in p or "MUX" in p], \
        "an unrelated adapter made the MUX look ambiguous"
