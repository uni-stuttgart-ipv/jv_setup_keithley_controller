"""Tests for parameter_tab.py — get_parameters, architecture toggle, units."""

from PyQt5.QtWidgets import QApplication
import pytest

from solarjv_analyzer.gui.widgets.parameter_tab import ParameterTab


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class TestParameterTab:
    @pytest.fixture
    def tab(self, qapp):
        return ParameterTab()

    def test_default_architecture_nip(self, tab):
        assert tab.get_parameters()["architecture"] == "n-i-p"

    def test_architecture_toggle_to_pin(self, tab):
        tab.architecture_toggle.setChecked(True)
        assert tab.get_parameters()["architecture"] == "p-i-n"

    def test_get_parameters_keys(self, tab):
        p = tab.get_parameters()
        for key in ("start_voltage", "stop_voltage", "step_size",
                     "sweep_rate", "compliance_current", "device_area",
                     "architecture"):
            assert key in p, f"Missing key: {key}"

    def test_voltage_unit_conversion(self, tab):
        """mV inputs are converted to V."""
        tab.start_voltage.setText("1200")
        tab.start_unit.setCurrentText("mV")
        tab.stop_voltage.setText("-200")
        tab.stop_unit.setCurrentText("mV")
        p = tab.get_parameters()
        assert p["start_voltage"] == pytest.approx(1.2)
        assert p["stop_voltage"] == pytest.approx(-0.2)

    def test_compliance_current_unit_conversion(self, tab):
        tab.compliance_current.setText("100")
        tab.comp_unit.setCurrentText("mA")
        p = tab.get_parameters()
        assert p["compliance_current"] == pytest.approx(0.1)

    def test_channel_selection_defaults(self, tab):
        """By default all 6 channels are selected."""
        channels = tab.get_selected_channels()
        assert channels == [1, 2, 3, 4, 5, 6]
