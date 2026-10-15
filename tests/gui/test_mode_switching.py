"""What the Advanced tab's mode toggle shows, hides and refuses.

Written BEFORE the mode table replaced the hand-written show/hide lists, so
the refactor has something to be measured against. Every assertion here
records behaviour that already existed; if one of them changes, the refactor
changed something it should not have.

The guards matter as much as the visibility. Switching mode mid-measurement
is refused and the toggle is put back, which is the difference between an
operator losing a sweep and being told why they cannot switch.
"""
import os
import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


ADVANCED_TAB = 1

# Every widget the two mode handlers touch. Named once so a widget added to
# one handler and forgotten in the other shows up here.
MODE_WIDGETS = (
    "vertical_splitter",
    "spo_display",
    "mpp_display",
    "params_tab",
    "spo_param_tab",
    "mpp_param_tab",
    "queue_button",
    "abort_button",
    "spo_start_button",
    "spo_abort_button",
    "mpp_start_button",
    "mpp_abort_button",
)


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
    # The window opens on JV+SPO; these are the Advanced tab's controls, and
    # a widget whose parent page is hidden reports itself hidden too.
    win.sidebar_mode_tabs.setCurrentIndex(ADVANCED_TAB)
    app.processEvents()
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def visibility(window):
    """Which mode widgets are shown, as the handlers left them.

    `isHidden()` rather than `isVisible()`: the top-level window is never
    show()n in a headless test, so isVisible() is False for everything and
    would make these assertions vacuous. isHidden() reflects the explicit
    show()/hide() calls, which is exactly the contract being pinned.
    """
    return {name: not getattr(window, name).isHidden() for name in MODE_WIDGETS}


# --- what each mode shows ---------------------------------------------------

def test_jv_mode_shows_the_sweep_controls(window, app):
    window._show_jv_mode()
    app.processEvents()

    assert visibility(window) == {
        "vertical_splitter": True,      # plot / browser / analysis
        "spo_display": False,
        "mpp_display": False,
        "params_tab": True,
        "spo_param_tab": False,
        "mpp_param_tab": False,
        "queue_button": True,
        "abort_button": True,
        "spo_start_button": False,
        "spo_abort_button": False,
        "mpp_start_button": False,
        "mpp_abort_button": False,
    }


def test_spo_mode_shows_the_hold_controls(window, app):
    window._show_spo_mode()
    app.processEvents()

    assert visibility(window) == {
        "vertical_splitter": False,
        "spo_display": True,
        "mpp_display": False,
        "params_tab": False,
        "spo_param_tab": True,
        "mpp_param_tab": False,
        "queue_button": False,
        "abort_button": False,
        "spo_start_button": True,
        "spo_abort_button": True,
        "mpp_start_button": False,
        "mpp_abort_button": False,
    }


def test_switching_back_and_forth_is_stable(window, app):
    """No widget may be left behind by a round trip."""
    window._show_jv_mode()
    app.processEvents()
    first = visibility(window)

    window._show_spo_mode()
    app.processEvents()
    window._show_jv_mode()
    app.processEvents()

    assert visibility(window) == first


def test_every_widget_belongs_to_exactly_one_mode(window, app):
    """None shared, none orphaned.

    With two modes this was "the modes are exact opposites". With three it is
    the more useful statement, and it is what the mode table is for: a widget
    named in no mode is unreachable, and one named in two is a bug the old
    hand-written lists made easy to write.
    """
    seen = {}
    for key in ("jv", "mpp", "spo"):
        window._show_mode(key)
        app.processEvents()
        seen[key] = visibility(window)

    for name in MODE_WIDGETS:
        owners = [key for key in seen if seen[key][name]]
        assert len(owners) == 1, (
            f"{name} is shown in {owners or 'no mode'}; "
            "every mode widget must belong to exactly one")


def test_mpp_mode_shows_the_tracking_controls(window, app):
    window._show_mode("mpp")
    app.processEvents()

    shown = visibility(window)
    assert shown["mpp_display"] and shown["mpp_param_tab"]
    assert shown["mpp_start_button"] and shown["mpp_abort_button"]
    assert not shown["vertical_splitter"] and not shown["spo_display"]


def test_the_toggle_reverts_to_the_mode_we_are_actually_in(window, app, monkeypatch):
    """The revert used to take a boolean, which only works with two modes."""
    window._show_mode("mpp")
    app.processEvents()
    window.mpp_mode_button.setChecked(True)

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(window, "_jv_is_busy", lambda: True)

    window._on_mode_button_clicked(window.spo_mode_button)
    app.processEvents()

    assert window.mpp_mode_button.isChecked(), (
        "a refused switch must put the toggle back on MPP, not on JV")
    assert not window.mpp_display.isHidden()


# --- the guards -------------------------------------------------------------

def test_switching_is_refused_while_spo_runs(window, app, monkeypatch):
    window._show_spo_mode()
    app.processEvents()
    window.spo_mode_button.setChecked(True)

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(window.controller, "spo_running", True, raising=False)

    window._on_mode_button_clicked(window.jv_mode_button)
    app.processEvents()

    assert warned, "switching mid-SPO should warn"
    # Still in SPO, and the toggle says so.
    assert not window.spo_display.isHidden()
    assert window.spo_mode_button.isChecked()


def test_switching_is_refused_while_jv_is_busy(window, app, monkeypatch):
    window._show_jv_mode()
    app.processEvents()
    window.jv_mode_button.setChecked(True)

    warned = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        lambda *a, **k: warned.append(a))
    monkeypatch.setattr(window, "_jv_is_busy", lambda: True)

    window._on_mode_button_clicked(window.spo_mode_button)
    app.processEvents()

    assert warned, "switching mid-sweep should warn"
    assert not window.vertical_splitter.isHidden()
    assert window.jv_mode_button.isChecked()


def test_a_permitted_switch_actually_switches(window, app, monkeypatch):
    """The guards must not refuse when nothing is running."""
    window._show_jv_mode()
    app.processEvents()
    monkeypatch.setattr(window.controller, "spo_running", False, raising=False)
    monkeypatch.setattr(window, "_jv_is_busy", lambda: False)

    window._on_mode_button_clicked(window.spo_mode_button)
    app.processEvents()

    assert not window.spo_display.isHidden()
    assert window.vertical_splitter.isHidden()


# ---------------------------------------------------------------------------
# Leaving the Advanced tab and coming back
#
# Every other test in this file calls _show_mode() or _on_mode_button_clicked()
# directly. That is why they all passed while the application was visibly
# broken: the bug lived on the OTHER path into Advanced -- the outer
# JV+SPO / Advanced tab -- which restored the mode with a two-way
# "JV, or else SPO" test written back when there were only two modes.
#
# MPP fell into the "or else", so re-entering Advanced showed SPO's panel,
# Start button, report button and plot while the MPP toggle stayed lit. These
# tests drive the real buttons, the way an operator does.
# ---------------------------------------------------------------------------

def _visible_mode(window):
    """The mode the operator can actually SEE, by its display widget."""
    shown = [key for key, spec in window.MODE_SPECS.items()
             if not spec.widgets[0].isHidden()]
    assert len(shown) == 1, f"expected exactly one mode visible, got {shown}"
    return shown[0]


def test_returning_to_advanced_keeps_the_mode_you_left_it_in(window, app):
    """The reported bug: MPP, leave Advanced, come back -> SPO's panel."""
    window._on_mode_button_click(1)              # into Advanced
    app.processEvents()
    window.mpp_mode_button.click()               # a REAL click
    app.processEvents()
    assert _visible_mode(window) == "mpp"

    window._on_mode_button_click(0)              # out to JV + SPO
    app.processEvents()
    window._on_mode_button_click(1)              # and back
    app.processEvents()

    assert _visible_mode(window) == "mpp", (
        "returning to Advanced dropped MPP -- the old two-way "
        "'JV, or else SPO' restore is back")
    assert window._active_mode == "mpp"


def test_the_lit_button_always_matches_what_is_on_screen(window, app):
    """The symptom that made this confusing: the toggle said MPP, the panel
    said SPO. Whatever happens, those two must agree."""
    window._on_mode_button_click(1)
    app.processEvents()

    for button, key in ((window.mpp_mode_button, "mpp"),
                        (window.spo_mode_button, "spo"),
                        (window.jv_mode_button, "jv"),
                        (window.mpp_mode_button, "mpp")):
        button.click()
        app.processEvents()
        window._on_mode_button_click(0)
        app.processEvents()
        window._on_mode_button_click(1)
        app.processEvents()

        lit = next(k for k, spec in window.MODE_SPECS.items()
                   if spec.button.isChecked())
        assert lit == key == _visible_mode(window), (
            f"after a round trip through JV+SPO: button says {lit}, "
            f"screen shows {_visible_mode(window)}, expected {key}")


def test_both_tabs_order_their_cards_the_same_way(window):
    """Parameters, then the voltage card, then Channel Selection -- in every
    tab that has those two cards, so the panel reads the same whichever mode
    the operator is in.

    MPP has to build the channel selector BEFORE the voltage card (the Quick
    JV sweep asks it which channel to measure) while showing it after, so the
    build order and the screen order disagree on purpose. Asserting screen
    position rather than construction is the point.
    """
    def card_titles(tab):
        boxes = [b for b in tab.findChildren(QtWidgets.QGroupBox) if b.title()]
        ordered = sorted(boxes,
                         key=lambda b: b.mapTo(tab, b.rect().topLeft()).y())
        seen, titles = set(), []
        for box in ordered:
            if box.title() not in seen:
                seen.add(box.title())
                titles.append(box.title())
        return titles

    for tab, voltage_card in ((window.mpp_param_tab, "Start Voltage"),
                              (window.spo_param_tab, "Hold Voltage")):
        titles = card_titles(tab)
        assert voltage_card in titles and "Channel Selection" in titles
        assert titles.index(voltage_card) < titles.index("Channel Selection"), (
            f"{titles} -- the voltage card belongs above Channel Selection")


def test_every_card_in_the_sidebar_is_the_same_width(window, app):
    """Cards must line up with File Output.

    FilePanel sits directly in the sidebar; the parameter cards are nested
    several layouts deeper. Each of those layouts used to add its own left
    and right inset, so Hold Voltage and Channel Selection came out 21px
    narrower than File Output with their titles out of line. Nothing says
    "these must agree" except this test.

    Checks every mode, not just one. Measuring showed the suite's known
    teardown segfault gets likelier with each test that BUILDS A WINDOW, not
    with what the test then does to it -- looping over the modes here costs
    nothing extra over checking one. See claude/test-suite-health.md.
    """
    window._on_mode_button_click(1)              # Advanced
    window.mpp_mode_button.click()               # the mode with both cards
    # Qt only computes real geometry once a window is shown, and the widths
    # below are meaningless without it. WA_DontShowOnScreen gets the layout
    # pass without creating a native window.
    window.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    window.show()
    app.processEvents()

    dock = window.findChildren(QtWidgets.QDockWidget)[0]
    sidebar = dock.findChild(QtWidgets.QScrollArea).widget()

    def visible_cards():
        found = []
        for box in sidebar.findChildren(QtWidgets.QGroupBox):
            if not box.title() or box.isHidden():
                continue
            parent, buried = box.parentWidget(), False
            while parent is not None and parent is not sidebar:
                if parent.isHidden():
                    buried = True
                    break
                parent = parent.parentWidget()
            if not buried:
                found.append((box.title(),
                              box.mapTo(sidebar, box.rect().topLeft()).x(),
                              box.width()))
        return found

    cards = visible_cards()

    def check(mode, found):
        assert len(found) >= 2, f"{mode}: expected several cards, saw {found}"
        assert len({x for _, x, _ in found}) == 1, (
            f"{mode}: cards start at different x -- titles will not line up: {found}")
        assert len({w for _, _, w in found}) == 1, (
            f"{mode}: cards have different widths: {found}")

    check("mpp", cards)
    assert len(cards) >= 3, (
        f"expected Start Voltage, Channel Selection and File Output: {cards}")

    for key, spec in window.MODE_SPECS.items():
        spec.button.click()
        app.processEvents()
        check(key, visible_cards())


def test_the_sidebar_title_is_centred_and_uses_the_heading_colour(window):
    """'Inputs' is a dock title. Qt draws those itself in the platform style,
    so it ignored the app's heading colour and sat left-aligned."""
    from solarjv_analyzer.gui.theme import tokens

    dock = window.findChildren(QtWidgets.QDockWidget)[0]
    bar = dock.titleBarWidget()
    assert isinstance(bar, QtWidgets.QLabel), (
        "the dock still uses Qt's own title bar, which cannot be styled to match")
    assert bar.text() == "Inputs"
    assert bar.alignment() & QtCore.Qt.AlignHCenter
    assert tokens.SECONDARY in bar.styleSheet(), (
        "the title should use the same colour as the card headings")


def test_open_folder_and_browse_follow_the_mode_you_are_looking_at(window, app,
                                                                   tmp_path,
                                                                   monkeypatch):
    """Reported from the lab: Open Folder in MPP opened the JV folder.

    The directory manager is a process-wide singleton, so every lookup puts
    its mode back afterwards -- otherwise a later J-V run would file itself
    under MPP. That restore is the trap: handing the file panel a plain
    `dir_manager.store_destination` reference means that by the time the
    operator clicks, the mode is back to JV. The getters have to close over
    the mode.

    SPO had this for Open Folder but not for Browse; MPP had neither.
    """
    monkeypatch.setenv("SOLARJV_STORE_ENABLED", "1")
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(tmp_path / "staging"))
    from solarjv_analyzer import store
    store.set_app_user("tester")

    window._on_mode_button_click(1)              # Advanced
    app.processEvents()

    try:
        for key, spec in window.MODE_SPECS.items():
            spec.button.click()
            app.processEvents()

            destination = str(window.file_panel.store_destination_getter() or "")
            assert destination, f"{key}: no store destination at all"
            assert destination.rstrip(os.sep).endswith(os.sep + spec.folder), (
                f"Open Folder in {key} mode points at {destination!r}, "
                f"not the {spec.folder} folder")

            start_dir = str(window.file_panel.dialog_start_dir_getter() or "")
            assert start_dir, f"{key}: Browse has nowhere to start"
    finally:
        store.set_app_user("")


def test_a_start_button_does_not_care_which_order_you_fill_the_form(window, app):
    """Reported from the lab: run Quick JV first, type the filename second,
    and Start MPP stayed dead until the sweep was re-run.

    Every mode's start button is gated on BOTH a filename and its own
    voltage, but only the voltage half announces itself with a signal. So
    the filename handler has to re-ask each mode -- and it asked SPO and the
    J-V queue, not MPP. Checking all of them here means a fourth mode cannot
    quietly repeat it.
    """
    window._on_mode_button_click(1)
    app.processEvents()

    checks = [
        ("mpp", window.mpp_start_button,
         lambda v: window.mpp_param_tab.vmpp_card.set_voltage(v, "quick_jv")),
        ("spo", window.spo_start_button,
         lambda v: window.spo_param_tab.vmpp_card.set_voltage(v, "quick_jv")),
    ]

    for name, button, set_voltage in checks:
        # voltage FIRST, filename SECOND -- the order that was broken
        window.file_panel.filename_input.setText("")
        set_voltage(-0.695)
        app.processEvents()
        assert not button.isEnabled(), (
            f"{name}: enabled with no filename")

        window.file_panel.filename_input.setText("run_one")
        app.processEvents()
        assert button.isEnabled(), (
            f"{name}: a filename typed AFTER the voltage did not enable the "
            f"button -- the filename handler does not re-ask this mode")

        # and the other order, which always worked
        window.file_panel.filename_input.setText("")
        app.processEvents()
        window.file_panel.filename_input.setText("run_two")
        set_voltage(-0.695)
        app.processEvents()
        assert button.isEnabled(), f"{name}: filename-first order broke"
