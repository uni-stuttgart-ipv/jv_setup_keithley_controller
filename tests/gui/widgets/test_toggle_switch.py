"""Tests for toggle_switch.py — ToggleSwitch widget."""

from PyQt5 import QtCore, QtTest
from PyQt5.QtWidgets import QApplication
import pytest

from solarjv_analyzer.gui.widgets.toggle_switch import ToggleSwitch


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class TestToggleSwitch:
    def test_default_unchecked(self, qapp):
        ts = ToggleSwitch()
        assert not ts.isChecked()

    def test_set_checked_updates_state(self, qapp):
        ts = ToggleSwitch()
        ts.setChecked(True)
        assert ts.isChecked()
        ts.setChecked(False)
        assert not ts.isChecked()

    def test_toggled_signal_emits(self, qapp):
        ts = ToggleSwitch()
        signals = []
        ts.toggled.connect(lambda v: signals.append(v))
        QtTest.QTest.mouseClick(ts, QtCore.Qt.LeftButton)
        assert signals == [True]
        QtTest.QTest.mouseClick(ts, QtCore.Qt.LeftButton)
        assert signals == [True, False]
