"""The standalone MPP tab: its fields, its gating, and its live display.

These run against the real window, so they cover the wiring as well as the
widgets — a field that exists but is not connected to anything passes a unit
test and fails an operator.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402
from solarjv_analyzer.mpp.mpp_widget import MAX_RUN_TIME_MIN  # noqa: E402

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
    win.deleteLater()
    app.processEvents()


# --- parameters -------------------------------------------------------------

def test_the_defaults_are_the_agreed_ones(window):
    params = window.mpp_param_tab.get_parameters()

    assert params["run_time"] == 5.0            # same as the SPO hold
    assert params["probe_step"] == 10.0         # mV
    assert params["probe_interval"] == 60.0
    assert params["data_interval"] == 10.0
    assert params["probe_points"] == 3
    assert params["averages_per_point"] == 10
    assert params["max_excursion"] == 300.0     # mV, the professor's figure
    assert params["active_channel"] == 1


def test_the_noise_threshold_is_off_by_default(window):
    """Switched on at 2% it freezes the tracker — see the MPP engine tests."""
    tab = window.mpp_param_tab

    assert not tab.threshold_check.isChecked()
    assert not tab.threshold_input.isEnabled(), (
        "the percentage must not be editable while the gate is off")
    assert tab.get_parameters()["use_noise_threshold"] is False

    tab.threshold_check.setChecked(True)
    assert tab.threshold_input.isEnabled()
    assert tab.get_parameters()["use_noise_threshold"] is True


def test_the_run_time_is_capped_at_three_hours(window):
    tab = window.mpp_param_tab
    tab.run_time_input.setText("600")           # ten hours

    assert tab.get_parameters()["run_time"] == MAX_RUN_TIME_MIN


def test_nonsense_falls_back_to_the_default(window):
    """An empty or unparsable field must not crash a run that is starting."""
    tab = window.mpp_param_tab
    tab.run_time_input.setText("")
    tab.probe_step_input.setText("abc")

    params = tab.get_parameters()
    assert params["run_time"] == 5.0
    assert params["probe_step"] == 10.0


def test_the_start_voltage_keeps_its_sign(window):
    """A p-i-n cell contacted the usual way here tracks around -1.1 V."""
    tab = window.mpp_param_tab
    tab.set_start_voltage(-1.1104)

    assert tab.has_valid_start_voltage()
    assert tab.get_parameters()["start_voltage"] == pytest.approx(-1.1104)


# --- gating -----------------------------------------------------------------

def test_start_is_refused_until_a_start_voltage_is_given(window, app):
    window._show_mode("mpp")
    app.processEvents()

    assert not window.mpp_param_tab.has_valid_start_voltage()
    assert not window.mpp_start_button.isEnabled()


def test_a_start_voltage_alone_is_not_enough(window, app, monkeypatch):
    """The filename gate still applies, as it does for J-V and SPO."""
    window._show_mode("mpp")
    monkeypatch.setattr(window, "_filename_valid", False)

    window.mpp_param_tab.set_start_voltage(-1.11)
    app.processEvents()

    assert not window.mpp_start_button.isEnabled()


def test_filename_and_start_voltage_together_enable_start(window, app, monkeypatch):
    window._show_mode("mpp")
    monkeypatch.setattr(window, "_filename_valid", True)
    monkeypatch.setattr(window.controller, "mpp_running", False, raising=False)

    window.mpp_param_tab.set_start_voltage(-1.11)
    app.processEvents()

    assert window.mpp_start_button.isEnabled()


def test_a_run_in_progress_keeps_start_disabled(window, app, monkeypatch):
    window._show_mode("mpp")
    monkeypatch.setattr(window, "_filename_valid", True)
    monkeypatch.setattr(window.controller, "mpp_running", True, raising=False)

    window.mpp_param_tab.set_start_voltage(-1.11)
    app.processEvents()

    assert not window.mpp_start_button.isEnabled()


# --- the live display -------------------------------------------------------

def test_starting_a_run_locks_the_fields(window):
    widget = window.mpp_widget
    widget.start_mpp()

    assert not window.mpp_param_tab.run_time_input.isEnabled()
    assert not window.mpp_param_tab.channel_selector.channels[0].isEnabled()
    assert widget.status_label.text() == "Tracking"


def test_each_sample_draws_power_and_the_tracked_voltage(window):
    widget = window.mpp_widget
    widget.start_mpp()

    widget.update_plot(0.0, -0.0020, -1.110)    # 2.0 mW generated
    widget.update_plot(10.0, -0.0021, -1.100)

    assert len(widget.power_curve.getData()[0]) == 2
    assert len(widget.voltage_curve.getData()[0]) == 2
    # Power as a magnitude, like Pmpp; the voltage signed, like Vmpp.
    assert widget._powers == pytest.approx([2.0, 2.1])
    assert widget._voltages == pytest.approx([-1.110, -1.100])
    assert widget.tracked_label.text() == "-1.100 V"
    assert widget.elapsed_label.text() == "10 s"


def test_finishing_unlocks_the_fields_and_shows_the_result(window):
    widget = window.mpp_widget
    widget.start_mpp()
    widget.update_plot(0.0, -0.0020, -1.110)

    widget.on_mpp_finished(
        {"mean_power_mw": 2.05, "drift_percent": -0.5, "final_voltage_v": -1.095},
        report_path="/tmp/mpp_report.csv")

    assert window.mpp_param_tab.run_time_input.isEnabled()
    assert widget.status_label.text() == "Complete"
    assert widget.report_path == "/tmp/mpp_report.csv"
    assert widget.tracked_label.text() == "-1.095 V"


# --- the Start Voltage card -------------------------------------------------

def test_the_card_is_there_and_labelled_for_mpp(window):
    card = window.mpp_param_tab.vmpp_card

    assert card.title() == "Start Voltage"
    assert card.vmpp_input.placeholderText() == "Enter start voltage (V)"
    # Both ways of setting it are offered, as on the SPO tab.
    assert card.manual_mode_button is not None
    assert card.quick_jv_button is not None


def test_typing_in_the_card_enables_start(window, app, monkeypatch):
    """The wiring, end to end: card -> tab signal -> window -> button."""
    window._show_mode("mpp")
    monkeypatch.setattr(window, "_filename_valid", True)
    monkeypatch.setattr(window.controller, "mpp_running", False, raising=False)

    assert not window.mpp_start_button.isEnabled()

    window.mpp_param_tab.vmpp_card.vmpp_input.setText("-1.1104")
    app.processEvents()

    assert window.mpp_start_button.isEnabled()
    assert window.mpp_param_tab.get_parameters()["start_voltage"] == pytest.approx(-1.1104)


def test_clearing_the_card_disables_start_again(window, app, monkeypatch):
    window._show_mode("mpp")
    monkeypatch.setattr(window, "_filename_valid", True)
    monkeypatch.setattr(window.controller, "mpp_running", False, raising=False)
    card = window.mpp_param_tab.vmpp_card

    card.vmpp_input.setText("-1.11")
    app.processEvents()
    assert window.mpp_start_button.isEnabled()

    card.vmpp_input.setText("")
    app.processEvents()
    assert not window.mpp_start_button.isEnabled()


def test_the_badge_says_where_the_voltage_came_from(window, app):
    card = window.mpp_param_tab.vmpp_card

    assert card.vmpp_badge.text() == "No Vmpp"

    card.vmpp_input.setText("-1.11")
    app.processEvents()
    assert card.vmpp_badge.text() == "Manual"

    # A J-V handoff marks its own provenance, so the operator can see the
    # number was measured rather than typed.
    card.set_voltage(-1.0987, source="quick_jv")
    app.processEvents()
    assert card.vmpp_badge.text() == "Quick JV"


def test_the_toggle_swaps_between_manual_and_quick_jv(window, app):
    card = window.mpp_param_tab.vmpp_card

    assert card.mode_stack.currentIndex() == 0          # Manual
    card.quick_jv_mode_button.setChecked(True)
    card._on_mode_button_clicked()
    app.processEvents()
    assert card.mode_stack.currentIndex() == 1          # Quick JV

    card.manual_mode_button.setChecked(True)
    card._on_mode_button_clicked()
    app.processEvents()
    assert card.mode_stack.currentIndex() == 0


def test_quick_jv_measures_the_selected_channel(window):
    """The sweep asks the tab's selector; the card has no channel of its own."""
    tab = window.mpp_param_tab
    tab.channel_selector.set_selected_channel(4)

    assert tab.vmpp_card._channel_provider() == 4


def test_quick_jv_is_refused_while_a_run_is_in_progress(window, monkeypatch):
    """It drives the same Keithley the tracker is using."""
    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(window.controller, "mpp_running", True, raising=False)

    window.mpp_param_tab.vmpp_card._run_quick_jv()

    assert warned, "Quick JV should refuse to run during an MPP run"
    assert not window.mpp_param_tab.vmpp_card._quick_jv_running


def test_a_run_locks_the_card_too(window):
    """Otherwise the voltage could be edited out from under a live tracker."""
    window.mpp_widget.start_mpp()
    card = window.mpp_param_tab.vmpp_card

    assert not card.vmpp_input.isEnabled()
    assert not card.quick_jv_button.isEnabled()

    window.mpp_widget.on_mpp_finished({})
    assert card.vmpp_input.isEnabled()


# --- the display layout, matching SPO ---------------------------------------

def _group_titles(widget):
    return [child.title() for child in widget.findChildren(QtWidgets.QGroupBox)]


def test_the_display_is_laid_out_like_the_spo_one(window):
    """A plot card that takes the slack, a metrics card at natural height.

    The two views sit in the same place in the same window, so an operator
    who learns one should be able to read the other without re-orienting.
    """
    mpp, spo = window.mpp_widget, window.spo_widget

    assert "Live Metrics" in _group_titles(mpp)
    assert "Live Metrics" in _group_titles(spo), "SPO's card was renamed"

    layout = mpp.layout()
    assert layout.count() == 2, "expected exactly a plot card and a metrics card"
    # The plot absorbs the extra height; the metrics card does not. Without
    # this the plot grew to fill the panel and the readouts were a sliver.
    assert layout.stretch(0) == 1
    assert layout.stretch(1) == 0
    assert layout.contentsMargins() == spo.layout().contentsMargins()
    assert layout.spacing() == spo.layout().spacing()


def test_every_readout_lives_in_the_metrics_card(window):
    widget = window.mpp_widget
    card = next(g for g in widget.findChildren(QtWidgets.QGroupBox)
                if g.title() == "Live Metrics")

    for label in (widget.mean_power_label, widget.drift_label,
                  widget.tracked_label, widget.elapsed_label,
                  widget.channel_label, widget.status_label):
        assert label.parent() is card, "a readout escaped the metrics card"


def test_the_plot_card_offers_an_open_button(window):
    """SPO has one because the J-V experiment browser is unreachable here."""
    widget = window.mpp_widget
    assert widget.open_report_button.text() == "Open MPP Report…"


# --- opening a saved report -------------------------------------------------

def test_a_saved_report_loads_back_into_the_view(window, tmp_path):
    """Round trip: a report this app wrote must be one it can reopen."""
    from solarjv_analyzer.mpp.mpp_report import MppReport

    report = MppReport(str(tmp_path / "mpp_ch2_raw.csv"))
    report.init({"Start Voltage": (-1.11, "V"), "Channel": (2, "")})
    report.write_row(0.0, -1.110, 0.0018, -0.0020, "baseline")
    report.write_row(10.0, -1.100, 0.0019, -0.0021, "probe")
    target = report.finalize({"mean_power_mw": 2.05, "drift_percent": -0.5,
                              "final_voltage_v": -1.100},
                             {"mean_power_mw": "mW", "drift_percent": "%"})

    widget = window.mpp_widget
    assert widget.load_report(target) is True

    assert widget._times == pytest.approx([0.0, 10.0])
    # The report stores power in mW; the view must not scale it again.
    assert widget._powers == pytest.approx([2.0, 2.1])
    assert widget._voltages == pytest.approx([-1.110, -1.100])
    assert len(widget.power_curve.getData()[0]) == 2
    assert widget.report_path == target
    assert "Loaded" in widget.status_label.text()


def test_a_file_that_is_not_an_mpp_report_is_declined(window, tmp_path):
    """False, not an exception — the caller falls back to the J-V loader."""
    widget = window.mpp_widget

    jv_like = tmp_path / "jv_report.csv"
    jv_like.write_text("[[ EXPERIMENTAL PARAMETERS ]]\nx,y\n1,2\n")
    assert widget.load_report(str(jv_like)) is False

    assert widget.load_report(str(tmp_path / "does_not_exist.csv")) is False

    empty_series = tmp_path / "empty.csv"
    empty_series.write_text("[[ TIME SERIES DATA ]]\nTime (s),V,I,P,Phase\n\n")
    assert widget.load_report(str(empty_series)) is False


# ---------------------------------------------------------------------------
# The parameter contract between the tab and the procedure
#
# `start_mpp` hands the tab's dictionary straight to MppProcedure as **kwargs,
# and pymeasure SILENTLY DROPS a keyword it does not recognise -- no warning,
# no exception. So renaming a field on one side and not the other does not
# break anything loudly: that setting just reverts to its default and the run
# looks completely normal, with a report that states the value the operator
# never actually got. These two tests make that rename fail here instead.
# ---------------------------------------------------------------------------

def test_every_tab_parameter_is_one_the_procedure_declares(window):
    from solarjv_analyzer.mpp.mpp_procedure import MppProcedure

    sent = set(window.mpp_param_tab.get_parameters())
    declared = set(MppProcedure().parameter_objects())

    orphans = sent - declared
    assert not orphans, (
        f"the tab sends {sorted(orphans)}, which MppProcedure does not "
        f"declare -- pymeasure would drop these without a word")


def test_the_tab_values_actually_reach_the_procedure(window):
    """Names matching is not enough; check the values land."""
    from solarjv_analyzer.mpp.mpp_procedure import MppProcedure

    tab = window.mpp_param_tab
    tab.run_time_input.setText("7")
    tab.probe_step_input.setText("25")
    tab.probe_interval_input.setText("45")
    tab.probe_points_input.setText("4")
    tab.max_excursion_input.setText("150")
    tab.preconditioning_input.setText("3.5")

    params = tab.get_parameters()
    procedure = MppProcedure(**params)

    for key, expected in params.items():
        actual = getattr(procedure, key)
        if isinstance(expected, bool):
            assert bool(actual) == expected, f"{key} did not survive"
        elif isinstance(expected, (int, float)):
            assert float(actual) == pytest.approx(float(expected)), (
                f"{key} arrived as {actual}, not {expected}")


# ---------------------------------------------------------------------------
# Quick JV inherits the device settings
#
# It used to pass only device_area and incident_power, so JVProcedure fell
# back to its own defaults for the other two: 0.18 A and "n-i-p". The
# compliance one is a device-safety bug -- it would push 180 mA through a cell
# the operator had limited to 20 mA. The architecture one made a correct p-i-n
# sweep report a polarity mismatch and send someone chasing their wiring.
# ---------------------------------------------------------------------------

def test_quick_jv_passes_the_cards_own_settings_to_the_procedure(window, app,
                                                                 monkeypatch):
    from solarjv_analyzer.gui.widgets import vmpp_card as vc

    card = window.mpp_param_tab.vmpp_card
    card.quick_jv_mode_button.setChecked(True)
    card.manual_mode_button.setChecked(False)
    card.compliance_field.set_value(0.025)       # 25 mA, on the CARD
    card.area_field.set_value(0.25)
    card.architecture_toggle.setChecked(True)    # p-i-n
    window.analysis_settings_tab.incident_power.setText("85")
    card.quick_jv_start_input.set_value(0.7)
    card.quick_jv_stop_input.set_value(-0.2)
    app.processEvents()

    class _StubManager:
        """Just enough instrument manager. The real one opens serial ports."""
        keithley = object()
        mux = object()

        def connect_keithley(self, **kw):
            pass

        def connect_mux(self, **kw):
            pass

        def is_keithley_alive(self):
            return True

        def is_mux_alive(self):
            return True

        def get_status(self):
            return {"keithley_connected": True, "mux_connected": True}

    monkeypatch.setattr(window, "instrument_manager", _StubManager())

    captured = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _NoWorker:
        def __init__(self, *a, **kw):
            self.results_ready = self.progress_changed = self.status_changed = \
                self.run_finished = self.run_failed = self.finished = _Signal()

        def start(self):
            pass

        def deleteLater(self):
            pass

    class _Signal:
        def connect(self, *a, **kw):
            pass

    monkeypatch.setattr(vc, "JVProcedure", _Spy)
    monkeypatch.setattr(vc, "SpoWorker", _NoWorker)

    card._run_quick_jv()

    assert captured, "the sweep never reached JVProcedure"
    assert captured["device_area"] == pytest.approx(0.25)
    assert captured["incident_power"] == pytest.approx(85.0)
    assert captured["compliance_current"] == pytest.approx(0.025), (
        "compliance must come from the card in AMPS, not the 0.18 A default "
        "-- this is the one that can damage a cell")
    assert captured["architecture"] == "p-i-n", (
        "architecture must follow the toggle, not default to n-i-p")
    assert captured["step_size"] == pytest.approx(
        card.quick_jv_step_input.value()), "step size is a field now, not a constant"


def test_quick_jv_mode_reads_the_cards_own_device_fields(window, app):
    """In Quick JV the panel is self-contained: its own compliance, area and
    architecture, independent of the J-V Sweep tab."""
    card = window.mpp_param_tab.vmpp_card
    jv = window.params_tab

    jv.compliance_current.setText("40")
    jv.device_area.setText("1.5")
    jv.architecture_toggle.setChecked(True)          # p-i-n on the JV tab
    card.compliance_field.set_value(0.007)           # 7 mA on the card
    card.area_field.set_value(0.25)
    card.architecture_toggle.setChecked(False)       # n-i-p on the card
    card.quick_jv_mode_button.setChecked(True)
    card.manual_mode_button.setChecked(False)
    app.processEvents()

    settings = card.device_settings()
    assert settings["compliance_current"] == pytest.approx(0.007)
    assert settings["device_area"] == pytest.approx(0.25)
    assert settings["architecture"] == "n-i-p"


def test_manual_mode_falls_back_to_the_jv_parameters_tab(window, app):
    """Manual mode shows only the voltage box, so there is nothing on this
    card to read -- the J-V tab is the only honest source."""
    card = window.mpp_param_tab.vmpp_card
    jv = window.params_tab

    jv.compliance_current.setText("40")
    jv.device_area.setText("1.5")
    jv.architecture_toggle.setChecked(True)
    card.compliance_field.set_value(0.007)           # must be ignored
    card.area_field.set_value(0.25)
    card.manual_mode_button.setChecked(True)
    card.quick_jv_mode_button.setChecked(False)
    app.processEvents()

    settings = card.device_settings()
    assert settings["compliance_current"] == pytest.approx(0.040)
    assert settings["device_area"] == pytest.approx(1.5)
    assert settings["architecture"] == "p-i-n"


def test_the_quick_jv_panel_has_the_seven_fields_in_order(window):
    """Start, Stop, Step Size, Sweep Rate, Compliance, Device Area,
    Device Architecture -- the order an operator was told to expect."""
    card = window.mpp_param_tab.vmpp_card
    for name in ("quick_jv_start_input", "quick_jv_stop_input",
                 "quick_jv_step_input", "quick_jv_rate_input",
                 "compliance_field", "area_field"):
        field = getattr(card, name)
        assert hasattr(field, "unit"), f"{name} has no unit selector"
    assert hasattr(card, "architecture_toggle")

    form = next(
        f for f in card.findChildren(QtWidgets.QFormLayout)
        if any(f.itemAt(r, QtWidgets.QFormLayout.FieldRole)
               and f.itemAt(r, QtWidgets.QFormLayout.FieldRole).widget()
               is card.compliance_field
               for r in range(f.rowCount())))
    labels = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QtWidgets.QFormLayout.LabelRole)
        if item is not None and item.widget() is not None:
            labels.append(item.widget().text().rstrip(":"))
    assert labels == ["Start", "Stop", "Step Size", "Sweep Rate",
                      "Compliance", "Device Area", "Device Architecture"], labels


def test_manual_mode_shows_only_the_voltage_box(window, app):
    card = window.mpp_param_tab.vmpp_card
    card.manual_mode_button.click()
    app.processEvents()
    assert not card.compliance_field.isVisibleTo(card), (
        "the sweep parameters should be hidden in Manual mode")
    assert card.vmpp_input.isVisibleTo(card)
    assert card.vmpp_unit.isVisibleTo(card)


def test_changing_a_unit_keeps_the_physical_value(window, app):
    """Switching mV -> V must not turn 10 mV into 10 V.

    This is the whole risk of adding unit selectors: the number on screen is
    meaningless without its unit, so the conversion has to happen when the
    unit changes, not just when the value is read.
    """
    tab = window.mpp_param_tab

    for field, key, new_unit in (
        (tab.probe_step_input, "probe_step", "V"),
        (tab.run_time_input, "run_time", "s"),
        (tab.max_excursion_input, "max_excursion", "V"),
        (tab.preconditioning_input, "preconditioning_time", "min"),
    ):
        before = tab.get_parameters()[key]
        field.unit.setCurrentText(new_unit)
        app.processEvents()
        after = tab.get_parameters()[key]
        assert after == pytest.approx(before), (
            f"{key} changed from {before} to {after} when the unit was "
            f"switched to {new_unit} -- the displayed number was not converted")

    card = window.mpp_param_tab.vmpp_card
    card.set_voltage(0.5, "manual")
    card.vmpp_unit.setCurrentText("mV")
    app.processEvents()
    assert card.voltage() == pytest.approx(0.5), (
        "the hold voltage changed value when its unit was switched")
    assert card.vmpp_input.text() == "500"


def test_every_mpp_field_with_a_dimension_has_a_unit_selector(window):
    """The ones that do not are genuinely dimensionless."""
    tab = window.mpp_param_tab
    for name in ("run_time_input", "probe_step_input", "probe_interval_input",
                 "data_interval_input", "max_excursion_input",
                 "preconditioning_input"):
        field = getattr(tab, name)
        assert hasattr(field, "unit"), f"{name} has no unit selector"
        assert field.unit.count() >= 2, f"{name}'s selector offers no choice"


# ---------------------------------------------------------------------------
# The architecture toggle has to DO something
#
# Reported from the lab: "n-i-p / p-i-n is not changing anything in Quick JV".
# Correct, and not for the reason it looks like. The metric math is
# deliberately architecture-agnostic -- it finds the maximum wherever the data
# puts it -- so the toggle changes no computed number anywhere. On the J-V tab
# it earns its place by mirroring the sweep window: a p-i-n cell generates in
# quadrant 2, so an n-i-p window never reaches its operating point and Vmpp
# comes back meaningless. This card's toggle was connected to nothing at all.
# ---------------------------------------------------------------------------

def test_flipping_the_architecture_mirrors_the_quick_jv_window(window, app):
    card = window.mpp_param_tab.vmpp_card

    start_nip = card.quick_jv_start_input.value()
    stop_nip = card.quick_jv_stop_input.value()
    assert start_nip > 0 and stop_nip < 0, "n-i-p should sweep down from +V"

    card.architecture_toggle.setChecked(True)          # p-i-n
    app.processEvents()

    assert card.quick_jv_start_input.value() == pytest.approx(-start_nip), (
        "flipping to p-i-n left the sweep in quadrant 4, where the cell does "
        "not generate -- the toggle did nothing")
    assert card.quick_jv_stop_input.value() == pytest.approx(-stop_nip)

    card.architecture_toggle.setChecked(False)         # back to n-i-p
    app.processEvents()
    assert card.quick_jv_start_input.value() == pytest.approx(start_nip)
    assert card.quick_jv_stop_input.value() == pytest.approx(stop_nip)


def test_quick_jv_starts_from_the_same_values_as_the_jv_sweep_tab(window):
    """Same measurement, same starting numbers.

    Anything else means two places to learn and two sets of habits, and an
    operator comparing a Quick JV result with a full sweep would be comparing
    different measurements without being told.
    """
    card = window.mpp_param_tab.vmpp_card
    jv = window.params_tab.get_parameters()

    assert card.quick_jv_start_input.value() == pytest.approx(jv["start_voltage"])
    assert card.quick_jv_stop_input.value() == pytest.approx(jv["stop_voltage"])
    assert card.quick_jv_step_input.value() == pytest.approx(jv["step_size"])
    assert card.quick_jv_rate_input.value() == pytest.approx(jv["sweep_rate"])


def test_a_p_i_n_sweep_is_not_rejected(window, app, monkeypatch):
    """The reported bug: 'start voltage must be greater than stop voltage'.

    That rule was invented for this card. JVProcedure sweeps in whichever
    direction the two voltages imply -- it has an explicit `if start > stop:
    ... else: ...` -- and the J-V Sweep tab has never enforced it. A p-i-n
    cell is measured from negative to positive, so the rule rejected exactly
    the cells the architecture toggle exists to support.
    """
    from solarjv_analyzer.gui.widgets import vmpp_card as vc

    card = window.mpp_param_tab.vmpp_card
    card.architecture_toggle.setChecked(True)          # p-i-n: -1.2 -> +0.2
    app.processEvents()
    assert card.quick_jv_start_input.value() < card.quick_jv_stop_input.value()

    class _StubManager:
        keithley = mux = object()

        def connect_keithley(self, **kw):
            pass

        def connect_mux(self, **kw):
            pass

        def is_keithley_alive(self):
            return True

        def is_mux_alive(self):
            return True

    monkeypatch.setattr(window, "instrument_manager", _StubManager())

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[1:3]))

    captured = {}

    class _Spy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class _Signal:
        def connect(self, *a, **kw):
            pass

    class _NoWorker:
        def __init__(self, *a, **kw):
            self.results_ready = self.progress_changed = self.status_changed = \
                self.run_finished = self.run_failed = self.finished = _Signal()

        def start(self):
            pass

        def deleteLater(self):
            pass

    monkeypatch.setattr(vc, "JVProcedure", _Spy)
    monkeypatch.setattr(vc, "SpoWorker", _NoWorker)

    card._run_quick_jv()

    assert not warned, f"a p-i-n sweep was refused: {warned}"
    assert captured, "the sweep never reached JVProcedure"
    assert captured["start_voltage"] == pytest.approx(-1.2)
    assert captured["stop_voltage"] == pytest.approx(0.2)


def test_a_zero_width_sweep_is_still_refused(window, app, monkeypatch):
    """Dropping the direction rule must not drop the real check."""
    card = window.mpp_param_tab.vmpp_card
    card.quick_jv_start_input.set_value(0.5)
    card.quick_jv_stop_input.set_value(0.5)
    app.processEvents()

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[1]))
    card._run_quick_jv()
    assert warned, "start == stop should still be refused"


def test_the_selected_architecture_is_visibly_emphasised(window, app):
    card = window.mpp_param_tab.vmpp_card
    card.architecture_toggle.setChecked(False)
    app.processEvents()
    nip_style, pin_style = (card._arch_nip_label.styleSheet(),
                            card._arch_pin_label.styleSheet())
    assert nip_style != pin_style, "both labels look the same; nothing is selected"
    assert "700" in nip_style, "the active side should be the bold one"

    card.architecture_toggle.setChecked(True)
    app.processEvents()
    assert "700" in card._arch_pin_label.styleSheet()
    assert card._arch_nip_label.styleSheet() != nip_style


def test_a_quick_jv_that_fails_to_start_does_not_lock_out_mpp_and_spo(window, app,
                                                                      monkeypatch):
    """Reported from the lab: after Quick JV the app insists it is in JV mode.

    `_run_quick_jv` sets controller.is_busy = True and then builds the
    procedure and worker UNGUARDED. Anything raising in there left the flag
    set with no run to clear it -- and start_mpp() and start_spo() both
    refuse while is_busy is True, so both were locked out for the rest of
    the session.
    """
    from solarjv_analyzer.gui.widgets import vmpp_card as vc

    card = window.mpp_param_tab.vmpp_card
    card.quick_jv_mode_button.setChecked(True)
    card.manual_mode_button.setChecked(False)
    app.processEvents()

    class _StubManager:
        keithley = mux = object()

        def connect_keithley(self, **kw):
            pass

        def connect_mux(self, **kw):
            pass

        def disconnect_keithley(self):
            pass

        def disconnect_mux(self):
            pass

        def is_keithley_alive(self):
            return True

        def is_mux_alive(self):
            return True

    monkeypatch.setattr(window, "instrument_manager", _StubManager())
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *a, **k: None)

    def _explode(**kwargs):
        raise ValueError("simulated bad parameter")

    monkeypatch.setattr(vc, "JVProcedure", _explode)

    assert window.controller.is_busy is False
    card._run_quick_jv()
    app.processEvents()

    assert window.controller.is_busy is False, (
        "a Quick JV that never started left the app marked busy -- MPP and "
        "SPO will both refuse to run")
    assert card._quick_jv_running is False
    assert card.quick_jv_button.isEnabled(), "the controls stayed locked"


def test_a_finished_quick_jv_leaves_the_app_idle(window, app, monkeypatch):
    """The ordinary path: once the sweep is done, MPP and SPO are startable."""
    card = window.mpp_param_tab.vmpp_card

    class _StubManager:
        def disconnect_keithley(self):
            pass

        def disconnect_mux(self):
            pass

        def is_keithley_alive(self):
            return True

        def is_mux_alive(self):
            return True

    monkeypatch.setattr(window, "instrument_manager", _StubManager())

    class _Proc:
        analysis_results = {card._channel_provider(): {"Forward": {
            "v_mpp": -0.52, "voc": -0.9, "jsc": 20.0,
            "ff": 75.0, "efficiency": 15.0}}}

    window.controller.is_busy = True          # as the sweep leaves it
    card._quick_jv_running = True
    card._on_quick_jv_finished(_Proc())
    app.processEvents()

    assert window.controller.is_busy is False
    assert card._quick_jv_running is False
