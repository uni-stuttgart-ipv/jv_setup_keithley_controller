"""Tests for analysis_panel.py — matrix table, public API, state management."""

from PyQt5.QtWidgets import QApplication
import pytest

from solarjv_analyzer.gui.widgets.analysis_panel import AnalysisPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


LABELS = [
    ("EFF", "%"), ("FF", "%"), ("Voc", "mV"), ("Jsc", "mA/cm2"),
    ("Vmpp", "mV"), ("Jmpp", "mA/cm2"), ("Pmpp", "mW"), ("Isc", "A"),
    ("Rsh", "Ohm"), ("Rs", "Ohm"), ("A", "cm2"), ("Incd. Pwr", "mW/cm2"),
]


class TestAnalysisPanel:
    @pytest.fixture
    def panel(self, qapp):
        return AnalysisPanel()

    # -- reset_channels ----------------------------------------------------

    def test_reset_channels_creates_rows(self, panel):
        panel.reset_channels([1, 2], LABELS)
        # 2 channels × 2 directions = 4 rows
        assert panel._table.rowCount() == 4
        assert panel._table.columnCount() == 13  # 1 label + 12 metrics
        assert "Ch 1" in panel._table.item(0, 0).text()
        assert "Ch 2" in panel._table.item(2, 0).text()

    def test_reset_channels_empty_shows_placeholder(self, panel):
        panel.reset_channels([], LABELS)
        assert panel._stack.currentIndex() == 0  # placeholder

    def test_reset_channels_single_channel(self, panel):
        panel.reset_channels([3], LABELS)
        assert panel._table.rowCount() == 2  # Forward + Reverse

    # -- analysis ----------------------------------------------------------

    def test_analysis_updates_cell(self, panel):
        panel.reset_channels([1], LABELS)
        panel.analysis({"Channel": 1, "Direction": "Forward", "EFF": 22.57})
        item = panel._table.item(0, 1)  # Col 1 = EFF
        assert "22.57" in item.text()

    def test_analysis_unknown_channel_silently_ignored(self, panel):
        panel.reset_channels([1], LABELS)
        # Should not raise
        panel.analysis({"Channel": 99, "Direction": "Forward", "EFF": 10.0})

    # -- clear_all ---------------------------------------------------------

    def test_clear_all_resets_to_em_dash(self, panel):
        panel.reset_channels([1], LABELS)
        panel.analysis({"Channel": 1, "Direction": "Forward", "EFF": 22.57})
        panel.clear_all()
        item = panel._table.item(0, 1)
        assert item.text() == "—"

    # -- set_single_sweep_mode ---------------------------------------------

    def test_single_sweep_mode_hides_reverse(self, panel):
        panel.reset_channels([1], LABELS)
        panel.set_single_sweep_mode(True)
        # Row 0 = Forward (visible), Row 1 = Reverse (hidden)
        assert not panel._table.isRowHidden(0)
        assert panel._table.isRowHidden(1)

    def test_single_sweep_mode_shows_all(self, panel):
        panel.reset_channels([1], LABELS)
        panel.set_single_sweep_mode(True)
        panel.set_single_sweep_mode(False)
        assert not panel._table.isRowHidden(0)
        assert not panel._table.isRowHidden(1)

    # -- set_active_channel ------------------------------------------------

    def test_set_active_channel_selects_row(self, panel):
        panel.reset_channels([1, 2], LABELS)
        panel.set_active_channel(2, "Forward")
        # Row 2 should be selected (Ch 1: rows 0-1, Ch 2: rows 2-3)
        selected = panel._table.selectionModel().selectedRows()
        assert len(selected) == 1
        assert selected[0].row() == 2

    def test_set_active_channel_missing_does_nothing(self, panel):
        panel.reset_channels([1], LABELS)
        # Should not raise
        panel.set_active_channel(99, "Forward")

    # -- _format_value -----------------------------------------------------

    def test_format_value_two_decimal(self, panel):
        assert panel._format_value(22.5678) == "22.57"

    def test_format_value_small_value_four_decimal(self, panel):
        assert panel._format_value(0.0596) == "0.0596"

    def test_format_value_extreme_scientific(self, panel):
        assert "e" in panel._format_value(1.234e-5)

    def test_format_value_infinity(self, panel):
        assert panel._format_value(float("inf")) == "inf"
