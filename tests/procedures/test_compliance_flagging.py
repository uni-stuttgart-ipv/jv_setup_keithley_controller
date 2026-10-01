# tests/procedures/test_compliance_flagging.py
"""
First-principles tests for compliance-limited point detection.

Physical basis: when the device wants more current than the programmed
compliance limit, the SourceMeter clamps the reading AT the limit — the
recorded value is the instrument's ceiling, not the device's J-V curve.
Evidence from a real calibration report (calibration_2026-05-06): a sweep
started at 0.7 V with 0.1 A compliance recorded J = 0.09999789, 0.09999777,
0.09999777, 0.09999776 for the first four points (the clamp, to within
0.003%), then fell onto the true diode curve at 0.66 V. Such points must be
flagged and excluded from metric computation.
"""
import numpy as np

from solarjv_analyzer.procedures.jv_procedure import JVProcedure


class TestComplianceFiltering:
    def test_real_report_clamped_points_flagged(self):
        """The exact values from the real calibration report: 4 clamped
        points at 0.1 A compliance must be flagged; the rest kept."""
        v = np.array([0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64])
        i = np.array([0.09999789, 0.09999777, 0.09999777, 0.09999776,
                      0.09315519, 0.08100446, 0.06908946])
        cv, ci, n_flagged = JVProcedure.filter_compliance_points(v, i, 0.1)
        assert n_flagged == 4
        assert len(cv) == 3
        assert cv[0] == 0.66  # first genuine measurement survives
        assert np.all(np.abs(ci) < 0.099)

    def test_clean_sweep_untouched(self):
        """A sweep that never reaches compliance loses no points."""
        v = np.linspace(-0.2, 0.6, 81)
        i = -(0.0596 - 0.07 * np.maximum(v - 0.5, 0))  # well below 0.1 A
        cv, ci, n_flagged = JVProcedure.filter_compliance_points(v, i, 0.1)
        assert n_flagged == 0
        assert len(cv) == len(v)

    def test_q2_polarity_flagged_symmetrically(self):
        """Compliance acts on |I| — a Q2 cell clamped at +0.1 A is flagged
        exactly like a Q4 cell clamped at -0.1 A."""
        i_q4 = np.array([-0.09999, -0.05, -0.01])
        i_q2 = -i_q4
        v = np.array([0.7, 0.5, 0.3])
        _, _, n4 = JVProcedure.filter_compliance_points(v, i_q4, 0.1)
        _, _, n2 = JVProcedure.filter_compliance_points(-v, i_q2, 0.1)
        assert n4 == n2 == 1

    def test_all_points_clamped_returns_empty(self):
        """A fully clamped sweep has zero valid measurements."""
        v = np.array([0.7, 0.69])
        i = np.array([0.0999999, 0.0999998])
        cv, ci, n_flagged = JVProcedure.filter_compliance_points(v, i, 0.1)
        assert n_flagged == 2
        assert len(cv) == 0
