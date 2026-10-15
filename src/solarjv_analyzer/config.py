# src/solarjv_analyzer/config.py

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class _Config:
    # Serial port (or other identifier) for your 6-channel MUX
    MUX_PORT: str = "COM4"
    # VISA resource string for your Keithley SourceMeter
    GPIB_ADDRESS = "ASRL3::INSTR"
    # If True, use simulated instruments instead of real hardware
    SIMULATION_MODE: bool = False
    # Base directory for all result files (relative or absolute)
    import os

    # Base directory for all user data
    RESULTS_ROOT = os.path.join(Path.home(), "SolarJV_Data")  # or "~/Reports"
    # Format string for date-based subfolders or filenames (day-month-year)
    DATE_FORMAT: str = "%d-%m-%Y"
    # Default prefix for result filenames
    FILENAME_PREFIX: str = "JV"
    # Default behavior: save each channel in separate files (True) or all in one (False)
    SAVE_SEPARATE_FILES: bool = True
    # Number of channels on the multiplexer
    CHANNEL_COUNT: int = 6
    
    TIMESTAMP_FORMAT = "%Y-%m-%d_%H:%M:%S"

    # ---- Protected store (S drive) ------------------------------------
    # Finished reports are copied here, laid out as
    #   <STORE_ROOT>\<windows user>\<yyyy-mm-dd>\<Calibration|JV|SPO>\
    # The share allows create and read but refuses delete, so the app never
    # writes working files here — see STAGING_ROOT. Override at runtime with
    # SOLARJV_STORE_ROOT (used for rehearsals and by the test suite).
    STORE_ROOT: str = r"S:\Data\JV"

    # All temp files and finished reports are written here first; the
    # publisher copies the finished ones to the store and then deletes the
    # local copy. Must be somewhere deletion is allowed.
    STAGING_ROOT: str = os.path.join(
        os.environ.get("LOCALAPPDATA")
        or os.path.join(os.path.expanduser("~"), ".local", "share"),
        "SolarJV", "staging",
    )

    # Master switch. SOLARJV_STORE_ENABLED=0 turns the feature off entirely
    # and the app behaves exactly as it did before it existed.
    STORE_ENABLED: bool = True

    # Seconds a file must go unmodified before it counts as finished.
    #
    # 0.0, because nothing depends on it any more. Every producer now writes
    # through `store.finalize.published_file()` or an excluded in-progress
    # name, so a file carrying its final name has already been closed by its
    # writer — a fact, where this was a guess that cost 15 s on every report.
    # Kept as a setting rather than deleted: it is the fallback if a future
    # writer is ever added that does not use the helper, and raising it is the
    # one-line way to make the sweeper cautious again. `LOG_QUIET_SECONDS` in
    # the sweeper still governs session logs, which have no completion event.
    STORE_QUIET_SECONDS: float = 0.0

    # How often the sweeper looks for finished files, in milliseconds.
    #
    # The QFileSystemWatcher in the sidecar is what makes publishing prompt;
    # this timer is the safety net behind it, for events the OS coalesces or
    # drops. It is deliberately slower than the 20 s poll it replaced, because
    # it is no longer the thing anybody waits on.
    STORE_SWEEP_INTERVAL_MS: int = 60000

# instantiate a singleton for easy import
CONFIG = _Config()

# Module-level constants for direct import
MUX_PORT = CONFIG.MUX_PORT
GPIB_ADDRESS = CONFIG.GPIB_ADDRESS
SIMULATION_MODE = CONFIG.SIMULATION_MODE
RESULTS_ROOT = CONFIG.RESULTS_ROOT
DATE_FORMAT = CONFIG.DATE_FORMAT
FILENAME_PREFIX = CONFIG.FILENAME_PREFIX
SAVE_SEPARATE_FILES = CONFIG.SAVE_SEPARATE_FILES
CHANNEL_COUNT = CONFIG.CHANNEL_COUNT
TIMESTAMP_FORMAT= CONFIG.TIMESTAMP_FORMAT
STORE_ROOT = CONFIG.STORE_ROOT
STAGING_ROOT = CONFIG.STAGING_ROOT
STORE_ENABLED = CONFIG.STORE_ENABLED
STORE_QUIET_SECONDS = CONFIG.STORE_QUIET_SECONDS
STORE_SWEEP_INTERVAL_MS = CONFIG.STORE_SWEEP_INTERVAL_MS
