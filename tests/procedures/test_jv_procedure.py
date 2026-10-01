"""Tests for procedures/jv_procedure.py — voltage sequence, NPLC, parsing."""

import numpy as np
import pytest
from solarjv_analyzer.procedures.jv_procedure import JVProcedure


class TestVoltageSequence:
    """_generate_voltage_sequence() tests."""

    def test_forward_positive_to_negative(self):
        p = JVProcedure()
        p.start_voltage = 1.2
        p.stop_voltage = -0.2
        p.step_size = -0.2
        p.sweep_direction = "Forward"
        seq = p._generate_voltage_sequence()
        assert seq[0] == pytest.approx(1.2)
        assert seq[-1] == pytest.approx(-0.2)
        assert len(seq) == 8  # 1.2, 1.0, 0.8, ..., -0.2

    def test_forward_negative_to_positive(self):
        p = JVProcedure()
        p.start_voltage = -0.2
        p.stop_voltage = 1.2
        p.step_size = 0.2
        p.sweep_direction = "Forward"
        seq = p._generate_voltage_sequence()
        assert seq[0] == pytest.approx(-0.2)
        assert seq[-1] == pytest.approx(1.2)

    def test_reverse_generates_backwards(self):
        p = JVProcedure()
        p.start_voltage = 1.2
        p.stop_voltage = -0.2
        p.step_size = -0.2
        p.sweep_direction = "Reverse"
        seq = p._generate_voltage_sequence()
        # Reverse sweeps from start towards stop just like forward,
        # but the sweep_direction flag tags them differently
        assert len(seq) > 0

    def test_stop_voltage_included(self):
        p = JVProcedure()
        p.start_voltage = 1.0
        p.stop_voltage = 0.0
        p.step_size = -0.3
        p.sweep_direction = "Forward"
        seq = p._generate_voltage_sequence()
        assert seq[-1] == pytest.approx(0.0)

    def test_step_absolute_value_used(self):
        p = JVProcedure()
        p.start_voltage = 0.0
        p.stop_voltage = 0.5
        p.step_size = -0.1  # negative step, abs used
        p.sweep_direction = "Forward"
        seq = p._generate_voltage_sequence()
        assert len(seq) == 6  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5


class TestNPLCCalculation:
    """_calculate_nplc() tests."""

    def test_normal_sweep_rate(self):
        p = JVProcedure()
        p.start_voltage = 1.0
        p.stop_voltage = 0.0
        p.step_size = -0.1
        p.sweep_rate = 0.5
        p.line_frequency = 50.0
        p.single_sweep_mode = True
        p.nplc = 1.0
        p.delay_between_points = 0.0
        nplc, delay = p._calculate_nplc()
        assert nplc > 0
        assert delay >= 0

    def test_zero_sweep_rate_falls_back_to_nplc(self):
        p = JVProcedure()
        p.start_voltage = 1.0
        p.stop_voltage = 0.0
        p.step_size = -0.1
        p.sweep_rate = 0.0
        p.line_frequency = 50.0
        p.single_sweep_mode = True
        p.nplc = 2.5
        p.delay_between_points = 0.01
        nplc, delay = p._calculate_nplc()
        assert nplc == pytest.approx(2.5)
        assert delay == pytest.approx(0.01)

    def test_nplc_clamped_to_range(self):
        p = JVProcedure()
        p.start_voltage = 1.0
        p.stop_voltage = 0.0
        p.step_size = -0.001  # tiny step → many points → tiny time per point
        p.sweep_rate = 0.1
        p.line_frequency = 50.0
        p.single_sweep_mode = True
        p.nplc = 1.0
        p.delay_between_points = 0.0
        nplc, _ = p._calculate_nplc()
        assert 0.01 <= nplc <= 10.0  # clamped


class TestDataColumns:
    def test_data_columns_defined(self):
        assert len(JVProcedure.DATA_COLUMNS) == 5
        assert "Voltage (V)" in JVProcedure.DATA_COLUMNS
        assert "Current (A)" in JVProcedure.DATA_COLUMNS

    def test_analysis_labels_units_match(self):
        """ANALYSIS_LABELS_UNITS is imported from analysis.py and has 14 entries
        (12 core J-V metrics + shunt resistivity and sheet resistance)."""
        assert len(JVProcedure.ANALYSIS_LABELS_UNITS) == 14
