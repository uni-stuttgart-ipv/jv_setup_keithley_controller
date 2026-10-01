r"""Work out which COM port each instrument is on, at startup.

`config.py` ships inside the packaged application, so once Briefcase has built
the installer there is no file for a lab user to edit — and COM numbers change
on their own: a different USB socket, a driver update, or another device
claiming the number first. This module removes that failure mode by resolving
the ports every time the app starts, and remembering what it found.

The two instruments are distinguishable because they use **different adapter
chips**:

| Instrument | Chip            | Typical VID | Identified by                    |
|------------|-----------------|-------------|----------------------------------|
| MUX        | CH340 (QinHeng) | 0x1A86      | USB identity only                |
| Keithley   | Prolific PL2303 | 0x067B      | USB identity **and** ``*IDN?``   |

The asymmetry matters. The Keithley can be *proved* — open the port, ask
``*IDN?``, look for ``KEITHLEY``/``2400``. The MUX cannot: its protocol writes
a frame and never reads a reply (audit finding S7), so there is nothing to ask
it. MUX identification is therefore an inference from the USB descriptor, and
the code refuses to guess when two candidates look alike rather than risk
driving the wrong device.

Resolution never probes a port that looks like the MUX. ``*IDN?`` is meaningless
to a board expecting binary hex frames and could be read as one.
"""

import json
import logging
import os
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# --- device signatures ----------------------------------------------------
# Matched on vendor ID OR a descriptor keyword: the PID varies across chip
# revisions (CH340/CH341, PL2303 HX/GT/TA/…) and hardcoding every one of them
# ages badly, while the vendor ID and the Windows description do not.
MUX_VIDS = {0x1A86}                       # QinHeng Electronics (CH340/CH341)
# Deliberately narrow. "usb-serial" was here and is far too loose — plenty of
# unrelated adapters describe themselves that way, and a false match makes the
# MUX look *ambiguous*, which refuses to resolve. Failing to connect is a worse
# outcome than the mis-identification the ambiguity check exists to prevent.
MUX_KEYWORDS = ("ch340", "ch341", "qinheng")
KEITHLEY_VIDS = {0x067B}                  # Prolific
KEITHLEY_KEYWORDS = ("pl2303", "prolific")

PROBE_TIMEOUT_S = 0.4
PROBE_BAUD = 9600                         # what VISAAdapter uses for ASRL
IDN_MARKERS = ("keithley", "2400")

_COM_RE = re.compile(r"^COM(\d+)$", re.IGNORECASE)

# Cached resolution for the life of the process.
_active = {"mux_port": "", "keithley_resource": "", "resolved": False,
           "keithley_confirmed": False, "mux_confirmed": False,
           "keithley_override": ""}

# Sources that mean "we know this is the right device", as opposed to "this is
# the best guess available".
CONFIRMED_SOURCES = ("pinned", "remembered", "detected")


# --- settings -------------------------------------------------------------
def settings_path() -> str:
    """`~/.solarjv/config.json` — the per-user file the rest of the app uses.

    Outside the packaged bundle, so it survives a reinstall and can be edited
    by hand when someone needs to force a port.
    """
    return os.path.join(os.path.expanduser("~"), ".solarjv", "config.json")


def _load_all() -> dict:
    try:
        with open(settings_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def load_hardware_settings() -> dict:
    """The `hardware` section, or {}. Namespaced so it cannot collide with
    `base_directory` or the store's `store_extra_copy`."""
    section = _load_all().get("hardware")
    return section if isinstance(section, dict) else {}


def save_hardware_settings(hardware: dict) -> None:
    data = _load_all()
    data["hardware"] = hardware
    try:
        os.makedirs(os.path.dirname(settings_path()), exist_ok=True)
        with open(settings_path(), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    except OSError as exc:
        logger.warning(f"Could not save the hardware settings: {exc}")


# --- port identity --------------------------------------------------------
def com_to_visa(device: str) -> str:
    """`COM3` -> `ASRL3::INSTR`."""
    match = _COM_RE.match((device or "").strip())
    return f"ASRL{match.group(1)}::INSTR" if match else ""


def _text_of(port) -> str:
    return " ".join(
        str(getattr(port, attr, "") or "")
        for attr in ("description", "manufacturer", "product", "hwid")
    ).lower()


def matches_signature(port, vids: set, keywords: tuple) -> bool:
    if getattr(port, "vid", None) in vids:
        return True
    text = _text_of(port)
    return any(keyword in text for keyword in keywords)


def identity_of(port) -> dict:
    return {
        "vid": getattr(port, "vid", None),
        "pid": getattr(port, "pid", None),
        "serial": getattr(port, "serial_number", None),
        "location": getattr(port, "location", None),
        "description": getattr(port, "description", None),
    }


def identity_matches(saved: dict, port) -> bool:
    """Is `port` the device we remembered?

    Strongest available evidence wins: a serial number pins the identity to one
    physical cable, so it survives being moved to any socket. Without one —
    common for CH340 — the USB location pins it to one socket, which is weaker
    but still better than the COM number. Description alone is the last resort.
    """
    if not saved:
        return False
    current = identity_of(port)
    if saved.get("vid") is not None and saved.get("vid") != current.get("vid"):
        return False
    if saved.get("pid") is not None and saved.get("pid") != current.get("pid"):
        return False
    if saved.get("serial") and current.get("serial"):
        return saved["serial"] == current["serial"]
    if saved.get("location") and current.get("location"):
        return saved["location"] == current["location"]
    if saved.get("description") and current.get("description"):
        return saved["description"] == current["description"]
    # Same vid/pid and nothing finer to go on.
    return saved.get("vid") is not None


# --- the Keithley probe ---------------------------------------------------
def probe_is_keithley(device: str, timeout: float = PROBE_TIMEOUT_S) -> tuple:
    """Ask a port `*IDN?` and see whether a Keithley 2400 answers.

    Returns (is_keithley, raw_reply). Read-only: `*IDN?` is an identity query
    and nothing else is ever sent. Both terminators are appended so the
    instrument answers whichever its front panel is set to.
    """
    try:
        import serial
    except Exception as exc:                         # pragma: no cover
        logger.debug(f"pyserial unavailable, cannot probe: {exc}")
        return False, ""
    try:
        with serial.Serial(device, PROBE_BAUD, timeout=timeout,
                           write_timeout=timeout) as handle:
            handle.reset_input_buffer()
            handle.write(b"*IDN?\r\n")
            handle.flush()
            time.sleep(min(timeout, 0.2))
            reply = handle.read(200).decode("latin-1", errors="replace").strip()
        # The app opens this same port through VISA moments later. Windows
        # releases a COM port on CloseHandle, but not always before the next
        # open reaches the driver — a tenth of a second here is cheaper than
        # an "access denied" at startup.
        time.sleep(0.1)
    except Exception as exc:
        logger.debug(f"Probe of {device} failed: {exc}")
        return False, ""
    low = reply.lower()
    return (all(marker in low for marker in IDN_MARKERS), reply)


# --- resolution -----------------------------------------------------------
class Resolution:
    """What startup decided, and how."""

    def __init__(self):
        self.mux_port = ""
        self.mux_source = "unresolved"
        self.keithley_resource = ""
        self.keithley_source = "unresolved"
        self.problems = []
        self.seen = []
        self.mux_identity = {}
        self.keithley_identity = {}

    @property
    def ok(self) -> bool:
        return bool(self.mux_port) and bool(self.keithley_resource)

    def summary(self) -> str:
        return (f"MUX={self.mux_port or '?'} ({self.mux_source}); "
                f"Keithley={self.keithley_resource or '?'} ({self.keithley_source})")


def _enumerate():
    try:
        from serial.tools import list_ports
        return list(list_ports.comports())
    except Exception as exc:
        logger.warning(f"Could not enumerate serial ports: {exc}")
        return []


def resolve(ports=None, settings=None, probe=True, config_mux=None,
            config_keithley=None, busy_ports=None) -> Resolution:
    """Decide both ports. Pure apart from the optional `*IDN?` probe.

    `busy_ports` names ports we already hold open — they are never probed.
    Opening one would fail (it is busy), which would be read as "not the
    Keithley" and could quietly move a live connection to a different port.
    """
    from solarjv_analyzer import config as app_config

    ports = _enumerate() if ports is None else list(ports)
    settings = load_hardware_settings() if settings is None else dict(settings)
    config_mux = app_config.MUX_PORT if config_mux is None else config_mux
    config_keithley = (app_config.GPIB_ADDRESS if config_keithley is None
                       else config_keithley)

    result = Resolution()
    result.seen = [
        f"{p.device} vid:pid={(getattr(p, 'vid', 0) or 0):04X}:"
        f"{(getattr(p, 'pid', 0) or 0):04X} {getattr(p, 'description', '')}"
        for p in ports
    ]
    by_name = {p.device.upper(): p for p in ports}

    # ---- MUX ------------------------------------------------------------
    pinned = settings.get("mux_port") if settings.get("mux_pinned") else None
    if pinned:
        # A pinned port is an instruction, not a hint. If it is missing, that
        # is an error to report — NOT an invitation to drive whatever other
        # CH340 happens to be plugged in. Resolution stops here either way.
        if pinned.upper() in by_name:
            result.mux_port, result.mux_source = pinned, "pinned"
        else:
            result.mux_source = "pinned but absent"
            result.problems.append(
                f"the pinned MUX port {pinned} is not connected")
    else:
        remembered = settings.get("mux_identity") or {}
        hits = [p for p in ports if identity_matches(remembered, p)]
        if len(hits) == 1:
            result.mux_port, result.mux_source = hits[0].device, "remembered"
        else:
            candidates = [p for p in ports
                          if matches_signature(p, MUX_VIDS, MUX_KEYWORDS)]
            if len(candidates) == 1:
                result.mux_port, result.mux_source = candidates[0].device, "detected"
            elif len(candidates) > 1:
                # The MUX protocol has no identify command, so there is no way
                # to tell two CH340s apart. Driving the wrong one would switch
                # relays on an unknown board — refuse rather than guess.
                result.problems.append(
                    "more than one CH340 adapter is connected "
                    f"({', '.join(p.device for p in candidates)}) — cannot tell "
                    "which is the MUX")
        if not result.mux_port and config_mux and config_mux.upper() in by_name:
            result.mux_port, result.mux_source = config_mux, "config"
    if result.mux_port and result.mux_port.upper() in by_name:
        result.mux_identity = identity_of(by_name[result.mux_port.upper()])

    # ---- Keithley -------------------------------------------------------
    pinned_k = (settings.get("keithley_resource")
                if settings.get("keithley_pinned") else None)
    if pinned_k:
        # Used as given, unlike the MUX: a VISA resource may legitimately name
        # a transport with no COM port to check (GPIB, USB-TMC). If it is wrong
        # the connect fails with a clear error rather than silently using some
        # other instrument.
        result.keithley_resource, result.keithley_source = pinned_k, "pinned"

    if not result.keithley_resource and config_keithley \
            and not _visa_to_com(config_keithley):
        # A GPIB or USB-TMC resource has no COM port to search for, so there is
        # nothing to detect and nothing to probe — use it as given.
        result.keithley_resource = config_keithley
        result.keithley_source = "config (non-serial)"

    if not result.keithley_resource:
        remembered = settings.get("keithley_identity") or {}
        hits = [p for p in ports if identity_matches(remembered, p)]
        if len(hits) == 1:
            result.keithley_resource = com_to_visa(hits[0].device)
            result.keithley_source = "remembered"
            result.keithley_identity = identity_of(hits[0])

    # Never probe the MUX: `*IDN?` means nothing to a board expecting binary
    # hex frames and could be read as one.
    excluded = {(result.mux_port or "").upper()} | {
        str(d).upper() for d in (busy_ports or ())}
    eligible = [p for p in ports if p.device.upper() not in excluded]
    likely = [p for p in eligible
              if matches_signature(p, KEITHLEY_VIDS, KEITHLEY_KEYWORDS)]

    if not result.keithley_resource and len(likely) == 1:
        # One PL2303 adapter and nothing else claiming to be one, so the
        # adapter identity already decides it. The rack is powered on before
        # the app is launched, so `*IDN?` should answer — ask it, because a
        # positive identification is worth having and a silent adapter is a
        # real signal (wrong cable, instrument off, or a baud-rate mismatch,
        # which looks like a software fault and is not one).
        #
        # The reply verifies; it does not gate. Refusing to resolve a lone
        # adapter because the instrument did not answer would turn a warning
        # into an outage.
        device = likely[0].device
        if probe:
            answered, reply = probe_is_keithley(device)
            if answered:
                logger.info(f"Keithley confirmed on {device}: {reply}")
            else:
                result.problems.append(
                    f"the PL2303 adapter on {device} did not answer *IDN? — "
                    "using it anyway; check the instrument is switched on and "
                    f"its RS-232 baud rate is {PROBE_BAUD}")
        result.keithley_resource = com_to_visa(device)
        result.keithley_source = "detected"
        result.keithley_identity = identity_of(likely[0])

    if not result.keithley_resource and probe:
        # Probing is now only a disambiguator, and only ever runs at startup.
        likely_names = {p.device for p in likely}
        # Compare by device name: ListPortInfo does not define equality in a
        # way `in` can be relied on.
        others = [p for p in eligible
                  if p.device not in likely_names
                  and not matches_signature(p, MUX_VIDS, MUX_KEYWORDS)]
        for port in likely + others:
            is_keithley, reply = probe_is_keithley(port.device)
            if is_keithley:
                result.keithley_resource = com_to_visa(port.device)
                result.keithley_source = "detected"
                result.keithley_identity = identity_of(port)
                logger.info(f"Keithley identified on {port.device}: {reply}")
                break
        if not result.keithley_resource and len(likely) > 1:
            result.problems.append(
                "more than one PL2303 adapter is connected "
                f"({', '.join(p.device for p in likely)}) and none answered "
                "*IDN? — is the Keithley switched on?")

    if not result.keithley_resource and config_keithley:
        com = _visa_to_com(config_keithley)
        if not com or com.upper() in by_name:
            result.keithley_resource = config_keithley
            result.keithley_source = "config"

    if not result.mux_port:
        result.problems.append("no MUX port could be resolved")
    if not result.keithley_resource:
        result.problems.append("no Keithley resource could be resolved")
    return result


def _visa_to_com(resource: str) -> str:
    match = re.search(r"ASRL(?:COM)?(\d+)::INSTR", resource or "", re.IGNORECASE)
    return f"COM{match.group(1)}" if match else ""


def apply(result: Resolution, remember: bool = True,
          upgrade_only: bool = False) -> None:
    """Make a resolution the process-wide answer, and persist what was learnt.

    `upgrade_only` keeps a re-resolution from *downgrading* a good answer: if
    the MUX was identified at startup and has since been unplugged, a later
    refresh must not replace a known-good port with nothing and then persist
    that. Only a confirmed result, or a result filling a gap, is accepted.
    """
    def _accept(new_value, new_source, current_value, currently_confirmed):
        if not upgrade_only:
            return True
        if not current_value:
            return bool(new_value)
        if currently_confirmed:
            return new_source in CONFIRMED_SOURCES and bool(new_value)
        return bool(new_value)

    if _accept(result.mux_port, result.mux_source,
               _active["mux_port"], _active["mux_confirmed"]):
        _active["mux_port"] = result.mux_port
        _active["mux_confirmed"] = result.mux_source in CONFIRMED_SOURCES
    if _accept(result.keithley_resource, result.keithley_source,
               _active["keithley_resource"], _active["keithley_confirmed"]):
        _active["keithley_resource"] = result.keithley_resource
        _active["keithley_confirmed"] = result.keithley_source in CONFIRMED_SOURCES
    _active["resolved"] = True
    if not remember:
        return
    settings = load_hardware_settings()
    settings.setdefault("mux_pinned", False)
    settings.setdefault("keithley_pinned", False)
    if result.mux_port:
        settings["mux_port"] = result.mux_port
        if result.mux_identity:
            settings["mux_identity"] = result.mux_identity
    if result.keithley_resource:
        settings["keithley_resource"] = result.keithley_resource
        if result.keithley_identity:
            settings["keithley_identity"] = result.keithley_identity
    settings["resolved_at"] = datetime.now().isoformat(timespec="seconds")
    save_hardware_settings(settings)


def resolve_at_startup() -> Resolution:
    """Resolve both ports once, log the outcome, and cache it."""
    result = resolve()
    if result.seen:
        logger.info("Serial ports seen: %s", "; ".join(result.seen))
    for problem in result.problems:
        logger.warning("Port resolution: %s", problem)
    logger.info("Port resolution: %s", result.summary())
    apply(result)
    return result


def keithley_confirmed() -> bool:
    """True when the Keithley resource came from evidence, not a guess."""
    return bool(_active["keithley_confirmed"])


def mux_confirmed() -> bool:
    return bool(_active["mux_confirmed"])


def refresh_if_unconfirmed(busy_ports=None) -> Resolution:
    """Re-resolve when the last attempt only produced a fallback.

    The calibration checklist tells operators to power the instruments on
    *after* launching the app, so a Keithley that was still off at startup
    cannot answer `*IDN?` and resolution falls back to whatever `config.py`
    says — which is exactly the stale value this feature exists to stop
    depending on. Re-resolving on the next connect attempt (the "Hardware
    Disconnected — Retry" path, or the start of a run) picks it up as soon as
    it is powered.

    Cheap and idempotent: once both are confirmed this does nothing at all.
    """
    if _active["resolved"] and _active["mux_confirmed"] and _active["keithley_confirmed"]:
        return None
    if _active.get("keithley_override") and _active.get("mux_confirmed"):
        # The operator has named the Keithley outright; nothing to look for.
        return None
    logger.info("Re-resolving instrument ports (previous result was a fallback)")
    # NO PROBING HERE. This runs inside connect_keithley()/connect_mux(), which
    # are called on the GUI thread — from CalibrationWindow.__init__ among
    # others. Opening serial ports there would violate the project's rule
    # against serial I/O on the GUI thread and can block for seconds on a ghost
    # COM port. Enumeration alone is a registry read, and is enough: the
    # adapters identify themselves without being opened.
    result = resolve(probe=False, busy_ports=busy_ports)
    for problem in result.problems:
        logger.warning("Port resolution: %s", problem)
    logger.info("Port resolution: %s", result.summary())
    apply(result, upgrade_only=True)
    return result


def active_mux_port() -> str:
    """The MUX port to use. Falls back to config if nothing was resolved."""
    if _active["resolved"] and _active["mux_port"]:
        return _active["mux_port"]
    from solarjv_analyzer import config
    return config.MUX_PORT


def set_session_keithley_resource(resource: str) -> str:
    """Override the Keithley address for THIS SESSION only.

    The manual backup for when auto-detection picks the wrong adapter, or
    there is nothing to detect (a GPIB or USB-TMC instrument). Deliberately
    **not** written to `~/.solarjv/config.json`: an address typed to get
    through one afternoon should not quietly become the permanent answer on
    every future launch, where it would be indistinguishable from a real
    detection and would outlive the situation that justified it.

    Pass "" to drop the override and go back to what resolution found.
    Returns the address now in force.
    """
    resource = (resource or "").strip()
    _active["keithley_override"] = resource
    if resource:
        logger.info(f"Keithley address overridden for this session: {resource}")
    else:
        logger.info("Keithley address override cleared")
    return active_keithley_resource()


def session_keithley_override() -> str:
    """The address the operator typed, or "" if none."""
    return _active.get("keithley_override", "")


def active_keithley_resource() -> str:
    """The Keithley VISA resource to use. Falls back to config.

    A session override beats resolution, which beats config — the operator
    typing an address is the strongest statement of intent available.
    """
    if _active.get("keithley_override"):
        return _active["keithley_override"]
    if _active["resolved"] and _active["keithley_resource"]:
        return _active["keithley_resource"]
    from solarjv_analyzer import config
    return config.GPIB_ADDRESS


def reset_for_tests() -> None:
    _active.update(mux_port="", keithley_resource="", resolved=False,
                   keithley_override="")
