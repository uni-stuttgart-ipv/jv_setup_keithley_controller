"""Tests for config.py — frozen dataclass singleton and module constants."""

import pytest


class TestConfig:
    def test_singleton_is_frozen(self):
        from solarjv_analyzer.config import _Config
        c = _Config()
        with pytest.raises(Exception):
            c.MUX_PORT = "CHANGED"

    def test_module_constants_exported(self):
        import solarjv_analyzer.config as cfg
        for name in (
            "MUX_PORT", "GPIB_ADDRESS", "SIMULATION_MODE", "RESULTS_ROOT",
            "DATE_FORMAT", "FILENAME_PREFIX", "SAVE_SEPARATE_FILES",
            "CHANNEL_COUNT", "TIMESTAMP_FORMAT",
        ):
            assert hasattr(cfg, name), f"Missing constant: {name}"

    def test_channel_count_is_six(self):
        from solarjv_analyzer.config import CHANNEL_COUNT
        assert CHANNEL_COUNT == 6
