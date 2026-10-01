"""Tests for spo/spo_report.py — crash-safe CSV writing and report finalize."""

import os
import tempfile

import pytest
from solarjv_analyzer.spo.spo_report import SpoReport


@pytest.fixture
def report():
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    r = SpoReport(path)
    yield r, path
    try:
        os.remove(path)
    except OSError:
        pass


class TestSpoReport:
    def test_init_writes_parameter_header(self, report):
        r, path = report
        params = {"Hold Voltage": (0.6, "V"), "Channel": (1, "")}
        r.init(params)
        r.close()
        with open(path) as f:
            content = f.read()
        assert "# Hold Voltage: 0.6 V" in content
        assert "# Channel: 1" in content
        assert "Time (s),Voltage (V),Current (A),Power (W)" in content

    def test_write_row_flushes_immediately(self, report):
        r, path = report
        r.init({"V": (0.5, "V")})
        r.write_row(0.0, 0.5, -0.01, -0.005)
        r.close()
        with open(path) as f:
            content = f.read()
        assert "0.0,0.5,-0.01,-0.005" in content or "0.0000,0.5,-0.01,-0.005" in content

    def test_write_row_before_init_raises(self, report):
        r, _ = report
        with pytest.raises(RuntimeError, match="init"):
            r.write_row(0.0, 0.5, -0.01, -0.005)

    def test_finalize_writes_all_blocks(self, report):
        r, path = report
        r.init({"Hold Voltage": (0.6, "V"), "Duration": (300.0, "s")})
        r.write_row(0.0, 0.6, -0.015, -0.009)
        r.write_row(1.0, 0.6, -0.014, -0.0084)

        metrics = {"mean_power_mw": 8.7, "drift_percent": 1.2}
        units = {"mean_power_mw": "mW", "drift_percent": "%"}
        report_path = r.finalize(metrics, units)

        assert os.path.exists(report_path)
        with open(report_path) as f:
            content = f.read()
        assert "[[ EXPERIMENTAL PARAMETERS ]]" in content
        assert "[[ SPO METRICS ]]" in content
        assert "[[ TIME SERIES DATA ]]" in content
        assert "8.7" in content
        assert "1.2" in content

        try:
            os.remove(report_path)
        except OSError:
            pass

    def test_close_then_finalize_still_works(self, report):
        """finalize() closes the raw file first, then writes the report from memory."""
        r, path = report
        r.init({"V": (0.5, "V")})
        r.write_row(0.0, 0.5, -0.01, -0.005)
        r.close()
        report_path = r.finalize({"m": 1.0}, {"m": "x"})
        assert os.path.exists(report_path)
        try:
            os.remove(report_path)
        except OSError:
            pass
