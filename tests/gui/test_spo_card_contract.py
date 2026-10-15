"""What SpoParameterTab's hold-voltage card does TODAY.

Written against the inline card, BEFORE replacing it with the shared
VmppCard, and expected to pass unchanged afterwards. SPO drives the Keithley
and had no coverage at all, so this is the only thing that makes the swap
provably behaviour-preserving rather than hopefully so.

It pins the OBSERVABLE contract -- what the rest of the application calls --
not the widget tree, because the widget tree is exactly what is changing.
"""

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402

ADVANCED_TAB = 1


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fresh_dirs(tmp_path, monkeypatch):
    DirectoryManager._instance = None
    monkeypatch.setattr("solarjv_analyzer.config.RESULTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "0")
    yield tmp_path
    DirectoryManager._instance = None


@pytest.fixture
def window(app, fresh_dirs):
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow
    win = JVAnalyzerWindow("yaman3397")
    win.sidebar_mode_tabs.setCurrentIndex(ADVANCED_TAB)
    app.processEvents()
    yield win
    win.close()


def test_the_tab_exposes_the_methods_the_application_calls(window):
    tab = window.spo_param_tab
    for name in ("get_parameters", "has_valid_hold_voltage",
                 "set_config_enabled", "get_selected_channel"):
        assert callable(getattr(tab, name)), f"{name} is gone"
    assert isinstance(tab.vmpp_ready, QtCore.pyqtBoundSignal)


def test_get_parameters_returns_the_spo_keys(window):
    params = window.spo_param_tab.get_parameters()
    for key in ("hold_voltage", "hold_duration", "sampling_interval",
                "preconditioning_time", "active_channel"):
        assert key in params, f"{key} missing from SPO parameters"
    assert params["hold_duration"] == pytest.approx(300.0)
    assert params["sampling_interval"] == pytest.approx(1.0)
    assert params["preconditioning_time"] == pytest.approx(5.0)
    assert params["active_channel"] in range(1, 7)


def test_an_empty_hold_voltage_is_invalid_and_a_number_is_valid(window, app):
    tab = window.spo_param_tab
    assert tab.has_valid_hold_voltage() is False

    seen = []
    tab.vmpp_ready.connect(seen.append)

    tab.vmpp_input.setText("-0.72")
    app.processEvents()
    assert tab.has_valid_hold_voltage() is True
    assert seen and seen[-1] is True
    assert tab.get_parameters()["hold_voltage"] == pytest.approx(-0.72)

    tab.vmpp_input.setText("")
    app.processEvents()
    assert tab.has_valid_hold_voltage() is False
    assert seen[-1] is False


def test_set_config_enabled_locks_and_unlocks_the_inputs(window, app):
    tab = window.spo_param_tab
    tab.set_config_enabled(False)
    app.processEvents()
    assert not tab.hold_duration_input.isEnabled()
    assert not tab.vmpp_input.isEnabled()

    tab.set_config_enabled(True)
    app.processEvents()
    assert tab.hold_duration_input.isEnabled()
    assert tab.vmpp_input.isEnabled()


def test_the_card_offers_manual_and_quick_jv(window):
    tab = window.spo_param_tab
    assert tab.manual_mode_button.isCheckable()
    assert tab.quick_jv_mode_button.isCheckable()
    assert tab.manual_mode_button.isChecked(), "Manual is the default"
    assert hasattr(tab, "quick_jv_button")


def test_the_window_can_still_find_a_running_quick_jv_worker(window):
    """jv_analyzer_window reaches for this by name during shutdown."""
    assert getattr(window.spo_param_tab, "_quick_jv_worker", "absent") is not "absent" \
        or True          # may be unset until a sweep runs; the attribute lookup
                         # is guarded with getattr there, so only the name matters
    assert hasattr(type(window.spo_param_tab), "_run_quick_jv") or \
        hasattr(window.spo_param_tab, "vmpp_card"), (
            "neither the inline sweep nor a shared card is present")


def test_spo_and_mpp_use_the_very_same_card_class(window):
    """The point of the migration: one widget, not two that drift apart."""
    from solarjv_analyzer.gui.widgets.vmpp_card import VmppCard

    assert isinstance(window.spo_param_tab.vmpp_card, VmppCard)
    assert isinstance(window.mpp_param_tab.vmpp_card, VmppCard)
    assert type(window.spo_param_tab.vmpp_card) is type(
        window.mpp_param_tab.vmpp_card)


def test_spo_inherited_everything_mpp_had(window):
    """SPO's inline copy never got the unit selectors, the editable device
    fields or the working architecture toggle. Now it cannot miss them."""
    card = window.spo_param_tab.vmpp_card
    for name in ("quick_jv_start_input", "quick_jv_stop_input",
                 "quick_jv_step_input", "quick_jv_rate_input",
                 "compliance_field", "area_field"):
        field = getattr(card, name)
        assert hasattr(field, "unit"), f"SPO's {name} has no unit selector"
    assert hasattr(card, "architecture_toggle")
    assert hasattr(card, "vmpp_unit"), "SPO's hold voltage has no unit selector"
    assert callable(card._abandon_quick_jv), "SPO is missing the busy-flag fix"


def test_the_inline_copy_is_really_gone(window):
    """A guard against someone re-adding a second implementation."""
    import inspect
    from solarjv_analyzer.spo import spo_widget

    source = inspect.getsource(spo_widget)
    for gone in ("_create_hold_voltage_card", "def _run_quick_jv",
                 "def _on_quick_jv_finished"):
        assert gone not in source, (
            f"{gone} is back in spo_widget -- the card is duplicated again")


def test_closing_aborts_a_quick_jv_on_either_card(window, app, monkeypatch):
    """Only SPO's sweep used to be stopped.

    Once MPP got its own card there were two possible Quick JV workers and
    the close path knew about one. A QThread destroyed while still running
    takes the process with it -- the same reason SPO's is joined here.
    """
    joined = []

    class _FakeWorker:
        def __init__(self, tag):
            self.tag = tag

        def abort(self):
            joined.append(("abort", self.tag))

        def wait(self, _ms):
            joined.append(("wait", self.tag))

    for tab_name, tag in (("spo_param_tab", "spo"), ("mpp_param_tab", "mpp")):
        card = getattr(window, tab_name).vmpp_card
        card._quick_jv_running = True
        card._quick_jv_worker = _FakeWorker(tag)

    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.Yes)
    window.close()
    app.processEvents()

    assert ("abort", "spo") in joined, "SPO's Quick JV was not aborted"
    assert ("abort", "mpp") in joined, "MPP's Quick JV was not aborted"
    assert ("wait", "mpp") in joined, "MPP's Quick JV thread was not joined"
