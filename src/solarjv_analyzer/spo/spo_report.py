"""
SPO (Set-Point Operation) Report Writer

Two artifacts are produced for every SPO run:

1. A raw, crash-safe CSV of the time-series data. Every row is written and
   flushed to disk (with fsync) the moment it is measured, so a crash never
   loses previously-recorded data (`init` / `write_row`).
2. A final, human-readable formatted report containing the experimental
   parameters, computed stability metrics, and the full time series. This
   is generated once the run completes (or is aborted) from the in-memory
   rows already safely persisted in the raw CSV (`finalize`).
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)


class SpoReport:
    """Writes the raw SPO time-series CSV and the final formatted report."""

    RAW_HEADER = ["Time (s)", "Voltage (V)", "Current (A)", "Power (W)"]

    def __init__(self, filepath: str):
        """
        Args:
            filepath: Full path to the raw, live-written CSV file.
        """
        self.filepath = filepath
        self.parameters = {}
        self._file = None
        self._writer = None
        self._rows = []  # kept in memory to build the final report

    def init(self, parameters: dict):
        """
        Open the raw CSV file and write the parameter header, then the
        time-series column header.

        Args:
            parameters: Mapping of parameter name -> (value, unit).
        """
        self.parameters = parameters
        self._file = open(self.filepath, "w", newline="", encoding="utf-8")
        for name, (value, unit) in parameters.items():
            unit_str = f" {unit}" if unit else ""
            self._file.write(f"# {name}: {value}{unit_str}\n")
        self._file.write("\n")

        self._writer = csv.writer(self._file)
        self._writer.writerow(self.RAW_HEADER)
        self._flush()

    def write_row(self, time_s: float, voltage_v: float, current_a: float, power_w: float):
        """Append one measurement sample and flush it to disk immediately."""
        if self._writer is None:
            raise RuntimeError("SpoReport.write_row() called before init().")

        row = [round(float(time_s), 4), float(voltage_v), float(current_a), float(power_w)]
        self._writer.writerow(row)
        self._flush()
        self._rows.append(row)

    def _flush(self):
        """Flush the OS buffer and fsync so data survives a crash."""
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            # Not all filesystems / platforms support fsync; flush() alone
            # already greatly reduces the risk of data loss.
            pass

    def close(self):
        """Close the raw CSV file handle, if open."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def finalize(self, metrics: dict, metrics_units: dict = None, output_path: str = None) -> str:
        """
        Write the final formatted report (parameters + metrics + full time
        series). Safe to call after an abort, since it only depends on the
        rows already flushed to the raw CSV.

        Args:
            metrics: Dict of computed SPO metrics (see spo_analysis).
            metrics_units: Optional dict mapping metric key -> unit string.
            output_path: Optional explicit output path; defaults to
                `<raw_basename>_report.csv` next to the raw file.

        Returns:
            str: Path to the written formatted report.
        """
        self.close()
        metrics_units = metrics_units or {}
        target = output_path or self._default_report_path()

        try:
            with open(target, "w", newline="", encoding="utf-8") as f:
                f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                f.write("Parameter,Value,Unit\n")
                for name, (value, unit) in self.parameters.items():
                    f.write(f"{name},{value},{unit}\n")
                f.write("\n")

                f.write("[[ SPO METRICS ]]\n")
                f.write("Metric,Value,Unit\n")
                for label, value in metrics.items():
                    unit = metrics_units.get(label, "")
                    f.write(f"{label},{value},{unit}\n")
                f.write("\n")

                f.write("[[ TIME SERIES DATA ]]\n")
                f.write("Time (s),Voltage (V),Current (A),Power (mW)\n")
                for time_s, voltage_v, current_a, power_w in self._rows:
                    f.write(f"{time_s},{voltage_v},{current_a},{power_w * 1000.0}\n")

            logger.info(f"SPO formatted report saved: {target}")
        except Exception as e:
            logger.error(f"Failed to write SPO formatted report: {e}")
            raise

        return target

    def _default_report_path(self) -> str:
        """Derive the formatted report path from the raw CSV path."""
        base, ext = os.path.splitext(self.filepath)
        if base.endswith("_raw"):
            base = base[: -len("_raw")]
        return f"{base}_report{ext or '.csv'}"
