"""Tests for app_controller.py — CSV parsing, formatting, architecture."""

from PyQt5.QtWidgets import QApplication
import pytest

from solarjv_analyzer.gui.app_controller import AppController


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------------------
# _parse_temp_file
# ---------------------------------------------------------------------------

class TestParseTempFile:
    def test_parses_analysis_block(self, tmp_path):
        """A minimal temp file with an [[ANALYSIS]] block."""
        path = tmp_path / "temp.csv"
        path.write_text(
            "# User Name: tester\n"
            "# Device Area: 0.089\n"
            "Channel,Voltage (V),Current (A),Time (s),Status\n"
            "1,0.5,-0.01,0.0,OK\n"
            "1,0.6,-0.005,0.1,OK\n"
            "[[ANALYSIS]]\n"
            "EFF\t22.5\n"
            "FF\t75.0\n"
            "[[/ANALYSIS]]\n"
        )
        # Minimal mock: we only need the view and its attributes that
        # _parse_temp_file touches (none — it's a pure method).
        # But we need to construct AppController with a view.
        # Instead, test the standalone helper directly.
        from solarjv_analyzer.gui.app_controller import AppController
        # _parse_temp_file is an instance method — call through a dummy
        import io
        import types

        # Create a minimal controller-like object just for _parse_temp_file
        ctrl = AppController.__new__(AppController)
        df, analysis, params = ctrl._parse_temp_file(str(path))
        assert len(df) == 2
        assert analysis.get("EFF") == 22.5
        assert analysis.get("FF") == 75.0
        assert len(params) >= 2  # User Name + Device Area

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        ctrl = AppController.__new__(AppController)
        df, analysis, params = ctrl._parse_temp_file(str(path))
        assert df.empty
        assert analysis == {}
        assert params == []


# ---------------------------------------------------------------------------
# _find_first_architecture
# ---------------------------------------------------------------------------

class TestFindFirstArchitecture:
    def test_no_experiments_returns_empty(self):
        """When browser is empty, return ''."""
        # We can't easily construct a full AppController with browser,
        # but we can test the method's logic on a mock.
        pass  # Requires full window — deferred to integration tests


# ---------------------------------------------------------------------------
# Architecture backward compatibility
# ---------------------------------------------------------------------------

class TestArchitectureBackwardCompat:
    def test_default_architecture_on_new_procedure(self):
        from solarjv_analyzer.procedures.jv_procedure import JVProcedure
        p = JVProcedure()
        assert p.architecture == "n-i-p"

    def test_architecture_stored_in_params(self):
        from solarjv_analyzer.procedures.jv_procedure import JVProcedure
        p = JVProcedure(architecture="p-i-n")
        assert p.architecture == "p-i-n"
