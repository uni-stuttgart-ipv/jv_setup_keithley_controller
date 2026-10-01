"""
SPO (Set-Point Operation) Report Writer

A run writes two files, but only ONE of them survives it:

1. A raw, crash-safe CSV of the time-series data. Every row is written and
   flushed to disk (with fsync) the moment it is measured, so a crash never
   loses previously-recorded data (`init` / `write_row`). This is a journal,
   not an output.
2. A final, human-readable formatted report containing the experimental
   parameters, computed stability metrics, and the full time series. This
   is generated once the run completes (or is aborted) from the in-memory
   rows already safely persisted in the raw CSV (`finalize`).

`finalize()` then **discards the journal**, but only after re-reading the
report and confirming it holds every row that was measured — the same shape
as the JV path, which deletes its per-channel `_temp` CSVs once the merged
report is written. One run, one file.

What that leaves behind is exactly right: the journal only persists when
`finalize()` never ran, i.e. when the application died mid-hold, which is the
one case where the journal is the only record of the measurement. The store
sweeper publishes files that have been quiet for a while, and the journal's
mtime updates on every sample, so it can never be published mid-run — but a
journal orphaned by a crash goes quiet and does reach the S drive.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)


def _header(name: str, unit: str) -> str:
    """`Mean Power (mW)` — the J-V report's header convention."""
    name = str(name).replace(",", " ")
    unit = str(unit or "").strip()
    return f"{name} ({unit})" if unit else name


def _cell(value) -> str:
    """One CSV cell: never emit a stray comma into a column-wise row."""
    text = "" if value is None else str(value)
    return text.replace(",", " ")


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


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

    def finalize(self, metrics: dict, metrics_units: dict = None,
                 output_path: str = None, discard_raw: bool = True,
                 notes_text: str = "", save_notes: bool = False) -> str:
        """
        Write the final formatted report (parameters + metrics + full time
        series), then discard the raw journal. Safe to call after an abort,
        since it only depends on the rows already flushed to the raw CSV.

        Args:
            metrics: Dict of computed SPO metrics (see spo_analysis).
            metrics_units: Optional dict mapping metric key -> unit string.
            output_path: Optional explicit output path; defaults to
                `<raw_basename>_report.csv` next to the raw file.
            discard_raw: Delete the raw CSV once the report is verified to
                contain every measured row. Pass False to keep both.
            notes_text: Operator notes, written verbatim in a `[[ NOTES ]]`
                section placed exactly where the J-V report puts it — between
                the metrics and the measurement data — so a combined JV+SPO
                run produces two reports carrying the same notes in the same
                place.
            save_notes: Whether the operator asked for the notes to be saved.

        Returns:
            str: Path to the written formatted report.
        """
        self.close()
        metrics_units = metrics_units or {}
        target = output_path or self._default_report_path()

        try:
            with open(target, "w", newline="", encoding="utf-8") as f:
                # COLUMN-WISE, to match the J-V report exactly: one header
                # row of names (units folded into the header) and one row of
                # values. The old row-wise "Parameter,Value,Unit" layout meant
                # the two report types could not be opened, diffed or
                # concatenated the same way, and a spreadsheet had to be
                # transposed by hand before anything could be plotted.
                f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                names, values = [], []
                for name, (value, unit) in self.parameters.items():
                    names.append(_header(name, unit))
                    values.append(_cell(value))
                f.write(",".join(names) + "\n")
                f.write(",".join(values) + "\n")
                f.write("\n")

                # Same shape as the J-V "[[ ANALYSIS SUMMARY ]]" block, down to
                # the leading Channel column, so both reports' analysis rows
                # line up when stacked.
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
                f.write("Time (s),Voltage (V),Current (A),Power (mW)\n")
                for time_s, voltage_v, current_a, power_w in self._rows:
                    f.write(f"{time_s},{voltage_v},{current_a},{power_w * 1000.0}\n")

            logger.info(f"SPO formatted report saved: {target}")
        except Exception as e:
            logger.error(f"Failed to write SPO formatted report: {e}")
            raise

        if discard_raw:
            self._discard_raw(target)

        return target

    def _report_is_complete(self, path: str) -> bool:
        """Re-read the report and check it holds every row we measured.

        Verifying by reading rather than by trusting the write is the point:
        the report is written with one `open(..., "w")`, and a failure part
        way through it (disk full, or the cp1252 encoding fault in audit A4)
        leaves a truncated file that looks plausible. Deleting the journal on
        the strength of that would be unrecoverable data loss.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            logger.error(f"Could not read back the SPO report {path}: {exc}")
            return False

        # Search from the END. Operator notes are written verbatim into a
        # section above this one, so a note that happens to contain the marker
        # text would otherwise be mistaken for the start of the data. Measured
        # rows are numeric and can never collide with it.
        label = "[[ TIME SERIES DATA ]]"
        if label not in lines:
            logger.error(f"The SPO report {path} has no time-series section.")
            return False
        marker = len(lines) - 1 - lines[::-1].index(label)

        # marker + 1 is the column header; the data starts after it.
        written = [line for line in lines[marker + 2:] if line.strip()]
        if len(written) != len(self._rows):
            logger.error(
                f"The SPO report {path} holds {len(written)} rows but "
                f"{len(self._rows)} were measured.")
            return False
        return True

    def _discard_raw(self, report_path: str):
        """Remove the raw journal now that the report has replaced it."""
        raw = self.filepath
        if not raw:
            return
        if _same_path(raw, report_path):
            # The report was written over the raw path — there is no separate
            # journal to remove, and removing it would delete the report.
            return
        if not self._report_is_complete(report_path):
            logger.warning(
                f"Keeping the raw SPO CSV {os.path.basename(raw)}: the report "
                "could not be verified.")
            return
        try:
            os.remove(raw)
            logger.info(
                f"Raw SPO CSV discarded after a verified report: "
                f"{os.path.basename(raw)}")
        except OSError as exc:
            # A write-once share refuses deletes, and Windows refuses them
            # while any handle is still open. Neither is a failure of the run:
            # the report is written and safe, so say so and move on.
            logger.warning(
                f"Could not remove the raw SPO CSV {os.path.basename(raw)}: "
                f"{exc}")

    def _default_report_path(self) -> str:
        """Derive the formatted report path from the raw CSV path."""
        base, ext = os.path.splitext(self.filepath)
        if base.endswith("_raw"):
            base = base[: -len("_raw")]
        return f"{base}_report{ext or '.csv'}"
