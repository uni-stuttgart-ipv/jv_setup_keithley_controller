"""Tests for file_panel.py — filename validation and parameter retrieval."""

from PyQt5.QtWidgets import QApplication
import pytest

from solarjv_analyzer.gui.widgets.file_panel import FilePanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class TestFilePanel:
    @pytest.fixture
    def panel(self, qapp):
        return FilePanel()

    def test_initial_empty_filename(self, panel):
        assert panel.filename_input.text() == ""

    def test_has_valid_filename_empty(self, panel):
        assert not panel.has_valid_filename()

    def test_has_valid_filename_whitespace(self, panel):
        panel.filename_input.setText("   ")
        assert not panel.has_valid_filename()

    def test_has_valid_filename_valid(self, panel):
        panel.filename_input.setText("TestRun")
        assert panel.has_valid_filename()

    def test_get_parameters(self, panel):
        panel.filename_input.setText("MyOutput.csv")
        params = panel.get_parameters()
        assert "filename" in params
        assert "directory" in params
        assert "single_file" in params
        assert params["filename"] == "MyOutput.csv"
        assert isinstance(params["single_file"], bool)

    def test_validation_hint_visible(self, panel):
        assert hasattr(panel, "filename_hint")
        # Hint is initially visible (empty filename)
        assert not panel.filename_hint.isHidden()
