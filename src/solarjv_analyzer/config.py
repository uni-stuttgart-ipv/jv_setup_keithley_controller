# src/solarjv_analyzer/config.py

from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class _Config:
    # Serial port (or other identifier) for your 6-channel MUX
    MUX_PORT: str = "/dev/ttyUSB0"
    # VISA resource string for your Keithley SourceMeter
    GPIB_ADDRESS: str = "GPIB::24"
    # If True, use simulated instruments instead of real hardware
    SIMULATION_MODE: bool = False
    # Base directory for all result files (relative or absolute)
    RESULTS_ROOT: Path = Path.home() / "Reports"
    # Format string for date-based subfolders or filenames (day-month-year)
    DATE_FORMAT: str = "%d-%m-%Y"
    # Default prefix for result filenames
    FILENAME_PREFIX: str = "JV"
    # Default behavior: save each channel in separate files (True) or all in one (False)
    SAVE_SEPARATE_FILES: bool = True
    # Number of channels on the multiplexer
    CHANNEL_COUNT: int = 6
    
    TIMESTAMP_FORMAT = "%Y-%m-%d_%H:%M:%S"

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