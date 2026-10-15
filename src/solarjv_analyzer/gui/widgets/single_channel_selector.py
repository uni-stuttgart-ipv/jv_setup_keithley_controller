"""Pick exactly one of the six channels, beside the reference pinout.

SPO and MPP both hold a single channel for the length of a run, so both need
this and neither needs the J-V tab's multi-select. The layout matches
`channel_pinout.png` and `ParameterTab`, so the toggles sit where the physical
channels do:

    Ch3   Ch4
    Ch2   Ch5
    Ch1   Ch6

`SpoParameterTab` still builds its own copy inline. Migrating it here is a
contained follow-up, worth doing on its own so the SPO tab can be verified by
itself rather than inside a larger change.
"""

from typing import List

from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer.gui.widgets.toggle_switch import ToggleSwitch
from solarjv_analyzer.gui.widgets.channel_pinout import build_pinout_label

# Physical positions, matching the pinout image.
_GRID = {3: (0, 0), 4: (0, 1),
         2: (1, 0), 5: (1, 1),
         1: (2, 0), 6: (2, 1)}

_CHIP_ON = ("background-color: #d1e3e9; color: #053a46; font-weight: 600;"
            "border-radius: 14px;")
_CHIP_OFF = "background-color: transparent; color: #94a3b8; font-weight: 600;"


class SingleChannelSelector(QtWidgets.QGroupBox):
    """One-of-six channel selection."""

    selection_changed = QtCore.pyqtSignal(int)

    def __init__(self, title: str = "Channel Selection", default: int = 1,
                 parent=None):
        super().__init__(title, parent)
        self.channels: List[ToggleSwitch] = []
        self.channel_number_labels: List[QtWidgets.QLabel] = []
        self._build_ui(default)

    def _build_ui(self, default: int):
        card_layout = QtWidgets.QVBoxLayout(self)
        card_layout.setSpacing(10)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)

        pinout_column = QtWidgets.QVBoxLayout()
        pinout_column.addWidget(build_pinout_label())
        caption = QtWidgets.QLabel("Reference Pinout")
        caption.setAlignment(QtCore.Qt.AlignCenter)
        caption.setStyleSheet("color: #64748b; font-size: 8pt;")
        pinout_column.addWidget(caption)
        body.addLayout(pinout_column)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        body.addWidget(divider)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.channel_button_group = QtWidgets.QButtonGroup(self)
        self.channel_button_group.setExclusive(True)

        for i in range(1, 7):
            number_label = QtWidgets.QLabel(str(i))
            number_label.setFixedSize(28, 28)
            number_label.setAlignment(QtCore.Qt.AlignCenter)
            self.channel_number_labels.append(number_label)

            toggle = ToggleSwitch()
            self.channels.append(toggle)
            self.channel_button_group.addButton(toggle, i)

        for channel, (row, col) in _GRID.items():
            idx = channel - 1
            pair = QtWidgets.QHBoxLayout()
            pair.setSpacing(8)
            pair.addWidget(self.channel_number_labels[idx])
            pair.addWidget(self.channels[idx])
            grid.addLayout(pair, row, col)

        body.addLayout(grid)
        body.addStretch(1)
        card_layout.addLayout(body)

        default = default if 1 <= default <= 6 else 1
        self.channels[default - 1].setChecked(True)
        self._paint_chip(default - 1, True)

        for idx, toggle in enumerate(self.channels):
            toggle.toggled.connect(
                lambda checked, i=idx: self._on_toggled(i, checked))

    def _on_toggled(self, index: int, checked: bool):
        self._paint_chip(index, checked)
        if checked:
            self.selection_changed.emit(index + 1)

    def _paint_chip(self, index: int, checked: bool):
        """Soft green when selected, muted when not."""
        self.channel_number_labels[index].setStyleSheet(
            _CHIP_ON if checked else _CHIP_OFF)

    # -- public API -------------------------------------------------------
    def selected_channel(self) -> int:
        """The selected channel, 1-based. Falls back to 1 if none is set."""
        checked = self.channel_button_group.checkedId()
        return checked if checked != -1 else 1

    def set_selected_channel(self, channel: int):
        if 1 <= channel <= 6:
            self.channels[channel - 1].setChecked(True)

    def set_config_enabled(self, enabled: bool):
        """Lock the toggles while a run is in progress."""
        for toggle in self.channels:
            toggle.setEnabled(enabled)
