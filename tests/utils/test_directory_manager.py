"""Tests for utils/directory_manager.py — path construction, mode switching."""

import os

import pytest
from solarjv_analyzer.utils.directory_manager import DirectoryManager


@pytest.fixture(autouse=True)
def fresh_singleton(monkeypatch):
    """Reset the DirectoryManager singleton between tests."""
    DirectoryManager._instance = None
    yield
    DirectoryManager._instance = None


class TestDirectoryManager:
    def test_singleton_returns_same_instance(self):
        a = DirectoryManager(username="u1")
        b = DirectoryManager(username="u2")
        assert a is b

    @staticmethod
    def _patch_results_root(monkeypatch, path):
        """RESULTS_ROOT is imported from config inside the module."""
        monkeypatch.setattr(
            "solarjv_analyzer.config.RESULTS_ROOT", path
        )

    def test_get_current_directory_main_mode(self, tmp_path, monkeypatch):
        root = str(tmp_path / "data")
        self._patch_results_root(monkeypatch, root)
        dm = DirectoryManager(username="testuser", mode="Main")
        d = dm.get_current_directory(create=True)
        assert "testuser" in d
        assert "Main" in d
        assert os.path.exists(d)

    def test_get_current_directory_calibration_mode(self, tmp_path, monkeypatch):
        root = str(tmp_path / "data")
        self._patch_results_root(monkeypatch, root)
        dm = DirectoryManager(username="testuser", mode="Calibration")
        d = dm.get_current_directory(create=True)
        assert "Calibration" in d
        assert os.path.exists(d)

    def test_mode_switch_changes_path(self, tmp_path, monkeypatch):
        root = str(tmp_path / "data")
        self._patch_results_root(monkeypatch, root)
        dm = DirectoryManager(username="testuser", mode="Main")
        main_dir = dm.get_current_directory(create=True)
        dm.set_mode("Calibration")
        cal_dir = dm.get_current_directory(create=True)
        assert main_dir != cal_dir

    def test_get_timestamp_filename(self):
        dm = DirectoryManager(username="testuser", mode="Main")
        fname = dm.get_timestamp_filename(prefix="measurement", extension=".csv")
        assert fname.startswith("measurement_")
        assert fname.endswith(".csv")

    def test_save_and_load_preference_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "solarjv_analyzer.utils.directory_manager.DirectoryManager._get_config_path",
            lambda _: str(tmp_path / "pref.json"),
        )
        dm = DirectoryManager(username="u", mode="Main")
        dm.save_preference("/my/custom/path")
        loaded = dm.load_preference()
        assert loaded == "/my/custom/path"
