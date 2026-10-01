"""The raw SPO CSV is a journal, not an output.

`finalize()` writes the report and then deletes the raw CSV — but only after
re-reading the report and confirming it holds every measured row. Deleting on
the strength of a write that was never checked is how a truncated report
becomes unrecoverable data loss.
"""
import os

import pytest

from solarjv_analyzer.spo.spo_report import SpoReport


PARAMS = {
    "Hold Voltage": (0.85, "V"),
    "Hold Duration": (30.0, "s"),
    "Channel": (3, ""),
}
METRICS = {"mean_power_mw": 12.5, "drift_percent": -1.2}


def _run(tmp_path, rows=5, name="spo_ch3_2026-09-19T10-00-00_raw.csv"):
    raw = tmp_path / name
    report = SpoReport(str(raw))
    report.init(PARAMS)
    for i in range(rows):
        report.write_row(i * 1.0, 0.85, 0.0147, 0.0125)
    return report, raw


def test_the_raw_csv_is_gone_once_the_report_is_written(tmp_path):
    report, raw = _run(tmp_path)
    assert raw.exists(), "the journal should exist during the run"

    target = report.finalize(METRICS)

    assert os.path.exists(target)
    assert not raw.exists(), "the raw journal outlived the report"


def test_the_report_keeps_every_row_the_journal_had(tmp_path):
    report, raw = _run(tmp_path, rows=7)
    target = report.finalize(METRICS)

    lines = open(target, encoding="utf-8").read().splitlines()
    marker = lines.index("[[ TIME SERIES DATA ]]")
    data = [line for line in lines[marker + 2:] if line.strip()]
    assert len(data) == 7


def test_a_truncated_report_does_not_take_the_journal_with_it(tmp_path, monkeypatch):
    """If the report cannot be verified, the journal is the only copy of the
    measurement and must survive."""
    report, raw = _run(tmp_path)
    monkeypatch.setattr(report, "_report_is_complete", lambda path: False)

    target = report.finalize(METRICS)

    assert os.path.exists(target)
    assert raw.exists(), "the journal was deleted despite an unverified report"


def test_verification_rejects_a_truncated_report(tmp_path):
    """The failure this guards against: one open(..., "w") that dies part way
    through (disk full, or the cp1252 encoding fault in audit A4) leaves a
    file that looks plausible but is short."""
    report, raw = _run(tmp_path, rows=6)
    target = report.finalize(METRICS, discard_raw=False)

    lines = open(target, encoding="utf-8").read().splitlines()
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines[:-2]) + "\n")     # lose the last 2 rows

    assert not report._report_is_complete(target)


def test_verification_rejects_a_report_with_no_time_series(tmp_path):
    report, raw = _run(tmp_path)
    broken = tmp_path / "broken_report.csv"
    broken.write_text("[[ SPO METRICS ]]\nMetric,Value,Unit\n", encoding="utf-8")

    assert not report._report_is_complete(str(broken))


def test_the_journal_survives_when_finalize_is_never_called(tmp_path):
    """A crash mid-hold leaves no report, so the journal IS the measurement."""
    report, raw = _run(tmp_path)
    report.close()

    assert raw.exists()


def test_writing_the_report_over_the_raw_path_deletes_nothing(tmp_path):
    """Combined mode passes an explicit output_path. If it ever collided with
    the raw path, discarding the raw would discard the report."""
    report, raw = _run(tmp_path)

    target = report.finalize(METRICS, output_path=str(raw))

    assert os.path.exists(target)
    assert raw.exists()


def test_an_undeletable_journal_is_not_an_error(tmp_path, monkeypatch):
    """A write-once share refuses deletes. The report is written and safe, so
    that must be a warning, not a failed run."""
    report, raw = _run(tmp_path)

    def refuse(path):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "remove", refuse)
    target = report.finalize(METRICS)          # must not raise

    assert os.path.exists(target)


def test_discard_can_be_switched_off(tmp_path):
    report, raw = _run(tmp_path)
    report.finalize(METRICS, discard_raw=False)
    assert raw.exists()
