"""The mechanism the whole feature rests on.

`store.attach()` redirects `DirectoryManager`'s base root to staging, and that
single call is what moves every producer's output — JV via the file panel, SPO
and calibration via the directory manager — without one line changing in
`app_controller`, `spo_procedure` or `calibration_window`.

If this test fails, the feature is silently writing measurement data straight
into the protected store (where working files cannot be deleted) or into the
old local tree (where nothing gets published). Both are bad enough to be worth
a full-window test.
"""
import logging
import os
import time

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture
def clean_logging():
    """Window log handlers outlive the window and crash later log calls."""
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for h in list(root.handlers):
        if h not in saved:
            root.removeHandler(h)
    for h in saved:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def store_env(tmp_path, monkeypatch):
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT",
                        str(tmp_path / "old-local-tree"))
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(tmp_path / "staging"))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "1")
    from solarjv_analyzer.store import identity, sweeper
    monkeypatch.setattr(identity, "_cached", "st000000")
    monkeypatch.setattr(sweeper, "extra_target", lambda: "")
    monkeypatch.setattr(sweeper, "_logs_dir", lambda: "")
    yield tmp_path
    DirectoryManager._instance = None


def test_attach_redirects_every_producer_into_staging(
    app, store_env, clean_logging
):
    from solarjv_analyzer import store
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    window = JVAnalyzerWindow("yaman3397")
    try:
        before = window.file_panel.get_parameters()["directory"]
        assert "old-local-tree" in before, before

        assert store.attach(window) is True

        # ...and it survives the app's own refresh, which fires on mode switches.
        window._update_save_directory()
        after = window.file_panel.get_parameters()["directory"]
        assert after.startswith(str(store_env / "staging")), after

        # SPO and calibration resolve through the same root.
        assert window.dir_manager.get_base_root() == str(store_env / "staging")
    finally:
        store.detach(window)
        window.close()
        window.deleteLater()
        app.processEvents()


def test_a_finished_run_reaches_the_store_and_leaves_temps_alone(
    app, store_env, clean_logging
):
    """The full round trip, using the paths the application really builds."""
    from solarjv_analyzer import store
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    window = JVAnalyzerWindow("yaman3397")
    try:
        assert store.attach(window) is True
        run_dir = window.file_panel.get_parameters()["directory"]
        os.makedirs(run_dir, exist_ok=True)

        # what app_controller writes during a two-channel single-file run
        temps = []
        for channel in (1, 2):
            temp = os.path.join(run_dir, f"jv_ch{channel}_forward_temp.csv")
            with open(temp, "w", encoding="utf-8") as handle:
                handle.write("Channel,Voltage (V),Current (A)\n1,0.5,-0.01\n")
            temps.append(temp)
        report = os.path.join(run_dir, "jv_2026-09-09_15-30-00_ch1_2.csv")
        payload = ("[[ ANALYSIS SUMMARY ]]\nEFF,18.4,%\n"
                   "[[ MEASUREMENT DATA ]]\n"
                   "Channel,Voltage (V),Current (A)\n1,0.50,-0.0102\n"
                   "# 25 °C, 3 µA offset\n")          # non-ascii on purpose
        with open(report, "w", encoding="utf-8") as handle:
            handle.write(payload)
        old = time.time() - 60
        os.utime(report, (old, old))
        for temp in temps:
            os.utime(temp, (old, old))

        summary = store.sweep(include_logs=False)

        assert (summary["published"], summary["failed"]) == (1, 0)

        # The date folder comes from the run's own dd-mm-yyyy staging folder,
        # converted — not from "today" — so a run crossing midnight is intact.
        published = (store_env / "store" / "st000000" /
                     time.strftime("%Y-%m-%d") / "Main" /
                     "jv_2026-09-09_15-30-00_ch1_2.csv")
        assert published.exists(), sorted(
            os.walk(str(store_env / "store")))
        assert published.read_text(encoding="utf-8") == payload

        assert not os.path.exists(report), "staging copy should be gone"
        for temp in temps:
            assert os.path.exists(temp), "temps are the app's to delete, not ours"
    finally:
        store.detach(window)
        window.close()
        window.deleteLater()
        app.processEvents()


def test_disabled_by_env_changes_nothing(app, store_env, clean_logging, monkeypatch):
    from solarjv_analyzer import store
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")
    window = JVAnalyzerWindow("yaman3397")
    try:
        before = window.file_panel.get_parameters()["directory"]
        assert store.attach(window) is False
        window._update_save_directory()
        assert window.file_panel.get_parameters()["directory"] == before
        assert "old-local-tree" in before
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
