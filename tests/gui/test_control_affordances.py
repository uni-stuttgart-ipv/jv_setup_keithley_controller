"""Two controls that did not look like what they were.

Both bugs were the same kind: the widget worked, but its colour told the
operator something false. These tests read the rendered pixels rather than the
stylesheet text, so they still hold if the QSS is refactored.
"""
import logging

import pytest

pytest.importorskip("PyQt5")

from PyQt5 import QtGui, QtWidgets                              # noqa: E402

from solarjv_analyzer.gui.theme import tokens                   # noqa: E402
from solarjv_analyzer.utils.directory_manager import DirectoryManager  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def clean_logging():
    root = logging.getLogger()
    saved, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in saved:
            root.removeHandler(handler)
    for handler in saved:
        if handler not in root.handlers:
            root.addHandler(handler)
    root.setLevel(level)


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
    yield win
    win.close()
    win.deleteLater()
    app.processEvents()


def _pixels(widget):
    image = widget.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
    return [QtGui.QColor(image.pixel(x, y))
            for y in range(image.height()) for x in range(image.width())]


def _has_color_near(colors, target_hex, tolerance=40):
    target = QtGui.QColor(target_hex)
    return any(
        abs(c.red() - target.red()) <= tolerance
        and abs(c.green() - target.green()) <= tolerance
        and abs(c.blue() - target.blue()) <= tolerance
        for c in colors)


# ------------------------------------------------- device architecture toggle
def test_the_architecture_toggle_is_teal_for_nip_and_red_for_pin(window):
    """Both sides used to be teal, so the selected architecture was not
    readable — and it flips the sign convention of every metric."""
    toggle = window.params_tab.architecture_toggle

    toggle.setChecked(False)
    toggle.sync_visual_state()
    assert _has_color_near(_pixels(toggle), tokens.ARCH_NIP_TEAL), \
        "n-i-p is not the primary teal"

    toggle.setChecked(True)
    toggle.sync_visual_state()
    pin_pixels = _pixels(toggle)
    assert _has_color_near(pin_pixels, tokens.ARCH_PIN_RED), \
        "p-i-n is not the muted red"
    assert not _has_color_near(pin_pixels, tokens.ARCH_NIP_TEAL, tolerance=25), \
        "p-i-n still renders teal — the two states look alike"


def test_the_pin_red_is_muted_not_the_alarm_red(window):
    """Functional red means something is WRONG. An architecture choice never
    is, so it must not borrow that colour."""
    muted = QtGui.QColor(tokens.ARCH_PIN_RED)
    alarm = QtGui.QColor(tokens.ERROR_BRIGHT)

    assert muted.saturation() < alarm.saturation(), "p-i-n red is not muted"
    assert muted.red() > muted.green() and muted.red() > muted.blue(), \
        "p-i-n is not recognisably red"


def test_the_active_architecture_label_carries_the_emphasis(window):
    tab = window.params_tab

    tab.architecture_toggle.setChecked(False)
    assert "700" in tab._arch_nip_label.styleSheet()
    assert "700" not in tab._arch_pin_label.styleSheet()

    tab.architecture_toggle.setChecked(True)
    assert "700" in tab._arch_pin_label.styleSheet()
    assert "700" not in tab._arch_nip_label.styleSheet()
    assert tokens.ARCH_PIN_RED_TEXT in tab._arch_pin_label.styleSheet()


def test_the_architecture_value_is_unchanged_by_the_restyle(window):
    tab = window.params_tab
    tab.architecture_toggle.setChecked(False)
    assert tab.get_parameters()["architecture"] == "n-i-p"
    tab.architecture_toggle.setChecked(True)
    assert tab.get_parameters()["architecture"] == "p-i-n"


# ------------------------------------------------ Advanced -> SPO hold source
def test_the_unselected_quick_jv_button_does_not_look_disabled(window, app):
    """"Quick JV" is the automatic short sweep that measures Vmpp. Styled as
    pale grey on transparent it read as a disabled control, so operators did
    not know it was a choice."""
    window._on_mode_button_click(1)              # Advanced
    window.spo_mode_button.setChecked(True)
    window._show_spo_mode()
    app.processEvents()

    button = window.spo_param_tab.quick_jv_mode_button
    assert button.isEnabled()
    assert not button.isChecked(), "expected the unselected state"

    assert _has_color_near(_pixels(button), tokens.PRIMARY, tolerance=60), \
        "the unselected Quick JV button carries no accent colour at all"


def test_the_hold_source_buttons_have_their_own_object_name(window):
    """Sharing ModeButton is what pulled in the inner-tab style whose
    unchecked state is indistinguishable from disabled."""
    tab = window.spo_param_tab
    for button in (tab.manual_mode_button, tab.quick_jv_mode_button):
        assert button.objectName() == "HoldModeButton"
        assert button.toolTip(), "no tooltip saying what this mode does"


# ------------------------------------------------------ SPO "Save Report"
def test_the_spo_card_has_no_save_report_button(window):
    """The report is written by SpoReport.finalize() and published by the
    store before this card updates. A button that only copied the finished
    file elsewhere implied it was not saved unless pressed."""
    assert not hasattr(window.spo_widget, "save_report_button")
    assert not hasattr(window.spo_widget, "save_report")


def test_pyqtgraphs_view_registry_does_not_grow_per_window(app, fresh_dirs):
    """`ViewBox.__init__` walks `ViewBox.AllViews` and calls a method on every
    entry. The dict is weak on the Python wrapper, not the C++ object, so
    closed windows' ViewBoxes linger and the walk dereferences freed memory —
    a segfault while building the next window's plots, with no traceback."""
    from PyQt5 import QtCore
    from pyqtgraph.graphicsItems.ViewBox import ViewBox
    from solarjv_analyzer.gui.jv_analyzer_window import JVAnalyzerWindow

    kept = []                       # a fixture or traceback frame does this
    sizes = []
    for _ in range(3):
        win = JVAnalyzerWindow("yaman3397")
        kept.append(win)
        win.close()
        win.deleteLater()
        app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        app.processEvents()
        sizes.append(len(ViewBox.AllViews))

    assert max(sizes) == min(sizes), f"the registry grew: {sizes}"
