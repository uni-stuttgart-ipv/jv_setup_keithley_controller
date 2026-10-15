"""The raw MPP journal and the formatted report.

Same two-file shape as SPO: a journal flushed row by row while the run is in
progress, then one formatted report written at the end from the rows held in
memory. The journal exists so a crash three hours into a tracking run still
leaves the measurements on disk.

Both halves go through `store.finalize`, so the store sweeper can tell a live
file from a finished one by name alone:

  * the journal carries `JOURNAL_SUFFIX` while it is being appended to. The
    sweeper skips it, and if a run dies the janitor promotes it so the data
    still reaches the store;
  * the report is written with `published_file()`, so it only takes its final
    name once it is closed and complete.

Without that, a `data_interval` of 20 s or 30 s would leave the journal idle
long enough for the sweeper to publish and delete it mid-run.
"""

import csv
import logging
import os

from solarjv_analyzer.store import paths
from solarjv_analyzer.store.finalize import promote, published_file

logger = logging.getLogger(__name__)

CHUNK_HEADERS = ["Time (s)", "Voltage (V)", "Current (A)", "Power (W)", "Phase"]


def _header(name: str, unit: str) -> str:
    return f"{name} ({unit})" if unit else name


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


class MppReport:
    """Writes the raw MPP time series, then the final formatted report."""

    RAW_HEADER = CHUNK_HEADERS

    def __init__(self, filepath: str):
        """
        Args:
            filepath: the LOGICAL path of the raw time series. The journal is
                written beside it under JOURNAL_SUFFIX and only takes this
                name if it has to outlive the run.
        """
        self.filepath = filepath
        self._raw_path = filepath + paths.JOURNAL_SUFFIX
        self.parameters = {}
        self._file = None
        self._writer = None
        self._rows = []            # kept in memory to build the final report

    @property
    def raw_path(self) -> str:
        """Where the journal is being written right now."""
        return self._raw_path

    # -- the live journal -------------------------------------------------
    def init(self, parameters: dict):
        """Open the journal and write the parameter and column headers."""
        self.parameters = parameters
        os.makedirs(os.path.dirname(self._raw_path) or ".", exist_ok=True)
        self._file = open(self._raw_path, "w", newline="", encoding="utf-8")
        for name, (value, unit) in parameters.items():
            unit_str = f" {unit}" if unit else ""
            self._file.write(f"# {name}: {value}{unit_str}\n")
        self._file.write("\n")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.RAW_HEADER)
        self._flush()

    def write_row(self, time_s: float, voltage_v: float, current_a: float,
                  power_w: float, phase: str):
        """Append one sample and push it at the disk immediately."""
        if self._writer is None:
            raise RuntimeError("MppReport.init() must be called before write_row()")
        row = (time_s, voltage_v, current_a, power_w, phase)
        self._rows.append(row)
        self._writer.writerow([_cell(v) if isinstance(v, float) else v for v in row])
        self._flush()

    def _flush(self):
        if self._file is None:
            return
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except (OSError, ValueError):
            pass            # not every filesystem supports it; flush is enough

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._writer = None

    # -- the formatted report ---------------------------------------------
    def finalize(self, metrics: dict, metrics_units: dict = None,
                 output_path: str = None, discard_raw: bool = True,
                 notes_text: str = "", save_notes: bool = False) -> str:
        """Write the report, then deal with the journal.

        Safe after an abort: it only uses rows already recorded.

        Returns:
            str: the path written.
        """
        self.close()
        metrics_units = metrics_units or {}
        target = output_path or self._default_report_path()

        try:
            with published_file(target, newline="", encoding="utf-8") as f:
                # Column-wise, matching the J-V and SPO reports exactly, so
                # `Open` parses all three with one code path.
                f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                names, values = [], []
                for name, (value, unit) in self.parameters.items():
                    names.append(_header(name, unit))
                    values.append(_cell(value))
                f.write(",".join(names) + "\n")
                f.write(",".join(values) + "\n")
                f.write("\n")

                f.write("[[ ANALYSIS SUMMARY ]]\n")
                labels = [_header(k, metrics_units.get(k, "")) for k in metrics]
                f.write("Channel," + ",".join(labels) + "\n")
                channel = self.parameters.get("Channel", ("", ""))[0]
                f.write(f"{_cell(channel)}," +
                        ",".join(_cell(v) for v in metrics.values()) + "\n")
                f.write("\n")

                if save_notes and notes_text.strip():
                    f.write("[[ NOTES ]]\n")
                    f.write(notes_text.strip())
                    f.write("\n\n")

                f.write("[[ TIME SERIES DATA ]]\n")
                f.write("Time (s),Voltage (V),Current (A),Power (mW),Phase\n")
                for time_s, voltage_v, current_a, power_w, phase in self._rows:
                    f.write(f"{time_s},{voltage_v},{current_a},"
                            f"{power_w * 1000.0},{phase}\n")

            logger.info(f"MPP formatted report saved: {target}")
        except Exception as e:
            logger.error(f"Failed to write MPP formatted report: {e}")
            # The journal is now the only copy of this run. Give it its real
            # name so the store publishes it, instead of leaving it for the
            # janitor. Both callers of finalize() log and swallow.
            self._promote_raw()
            raise

        if discard_raw:
            self._discard_raw(target)
        else:
            self._promote_raw()

        return target

    def _report_is_complete(self, path: str) -> bool:
        """Re-read the report and check it holds every row measured.

        Verifying by reading rather than trusting the write: deleting the
        journal on the strength of an unchecked write is how a truncated
        report becomes unrecoverable data loss.
        """
        label = "[[ TIME SERIES DATA ]]"
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError:
            return False
        if label not in lines:
            return False
        # The LAST occurrence: a note could legitimately contain the marker.
        marker = len(lines) - 1 - lines[::-1].index(label)
        written = [line for line in lines[marker + 2:] if line.strip()]
        return len(written) >= len(self._rows)

    def _promote_raw(self):
        """Give the journal its real name, so it is a publishable file."""
        if not self._raw_path or not os.path.exists(self._raw_path):
            return
        try:
            promote(self._raw_path, self.filepath)
            logger.info(
                f"Kept the raw MPP time series as "
                f"{os.path.basename(self.filepath)}; it will reach the store.")
        except OSError as exc:
            logger.warning(f"Could not promote the raw MPP journal: {exc}")

    def _discard_raw(self, report_path: str):
        """Remove the journal now the report has replaced it."""
        if not self._raw_path:
            return
        if not self._report_is_complete(report_path):
            logger.warning(
                f"Keeping the raw MPP journal "
                f"{os.path.basename(self._raw_path)}: the report could not be "
                "verified.")
            self._promote_raw()
            return
        try:
            os.remove(self._raw_path)
            logger.info("Raw MPP journal discarded after a verified report.")
        except OSError as exc:
            # A write-once share refuses deletes, and Windows refuses them
            # while a handle is open. The report is written and safe either
            # way, so this is a warning, not a failed run.
            logger.warning(f"Could not remove the raw MPP journal: {exc}")

    def _default_report_path(self) -> str:
        base, ext = os.path.splitext(self.filepath)
        if base.endswith("_raw"):
            base = base[: -len("_raw")]
        return f"{base}_report{ext or '.csv'}"
