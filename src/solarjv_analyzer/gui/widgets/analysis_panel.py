"""
Analysis Panel for J-V Measurement Results

Displays per-channel solar cell metrics in a full-width horizontal matrix table.
Each row represents a (channel, direction) pair. Column 0 shows a vibrant
left-colour indicator matching the plot-curve colour for that channel.
Forward/Reverse rows for the same channel share the same accent colour.
"""

import logging
from typing import Dict, List, Tuple

from PyQt5 import QtWidgets, QtCore, QtGui

from solarjv_analyzer.gui.style import (
    COLOR_TEXT_MUTED,
    COLOR_ACCENT_BLUE,
    COLOR_ACCENT_GREEN,
    COLOR_ACCENT_RED,
)

logger = logging.getLogger(__name__)

# Height of the coloured indicator bar rendered in column 0 of each row.
_INDICATOR_HEIGHT = 14
_INDICATOR_WIDTH = 5


class AnalysisPanel(QtWidgets.QWidget):
    """
    Horizontal matrix panel displaying analysis metrics for all measured channels.

    Emits:
        row_selected(int, str): channel number and direction when a row is clicked.
    """

    # Vibrant channel accent colours — kept in sync with AppController.CHANNEL_COLORS.
    CHANNEL_COLORS = {
        1: COLOR_ACCENT_BLUE,   # "#053a46"
        2: COLOR_ACCENT_GREEN,  # "#10b981"
        3: COLOR_ACCENT_RED,    # "#ef4444"
        4: "#8b5cf6",           # Violet
        5: "#f59e0b",           # Amber
        6: "#ec4899",           # Pink
    }

    DEFAULT_LABELS_UNITS = [
        ("EFF", "%"),
        ("FF", "%"),
        ("Voc", "mV"),
        ("Jsc", "mA/cm2"),
        ("Vmpp", "mV"),
        ("Jmpp", "mA/cm2"),
        ("Pmpp", "mW"),
        ("Isc", "A"),
        ("Rsh", "Ohm"),
        ("Rs", "Ohm"),
        ("Rho_shunt", "Ohm.cm"),
        ("Rsq", "Ohm/sq"),
        ("Area", "cm2"),
        ("Incd. Pwr", "mW/cm2"),
    ]

    PLACEHOLDER_TEXT = "No analysis data yet – run a sweep or load a file"

    # -------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------
    row_selected = QtCore.pyqtSignal(int, str)
    """Emitted when the user clicks any cell in a row: (channel, direction)."""

    # -------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        small_font = self.font()
        small_font.setPointSize(10)
        self.setFont(small_font)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        # --- Internal state --------------------------------------------------
        self._labels_units: List[Tuple[str, str]] = []
        self._col_map: Dict[str, int] = {}  # label → column index
        self._row_map: Dict[Tuple[int, str], int] = {}  # (ch, dir) → row index
        self._single_sweep_mode = False

        # --- Placeholder (page 0 of stack) ----------------------------------
        self._placeholder = QtWidgets.QLabel(self.PLACEHOLDER_TEXT)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 14px;"
            f" font-style: italic; background: transparent;"
        )

        # --- Data table (page 1 of stack) -----------------------------------
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(0)
        self._table.setRowCount(0)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(False)
        self._table.setShowGrid(False)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)

        # Horizontal scrollbar for when metric columns exceed viewport width
        self._table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        # Horizontal header – all columns ResizeToContents
        fm = self.fontMetrics()
        pad_h = max(4, int(fm.height() * 0.25))
        pad_v = max(2, int(fm.height() * 0.15))

        h_header = self._table.horizontalHeader()
        h_header.setMinimumSectionSize(70)
        h_header.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        h_header.setSectionsMovable(False)

        # Dynamic cell padding so text never hugs cell borders
        self._base_stylesheet = (
            f"QTableWidget::item {{"
            f"  padding: {pad_v}px {pad_h}px;"
            f"}}"
        )
        self._table.setStyleSheet(self._base_stylesheet)

        # Vertical header – compact, fixed row height
        v_header = self._table.verticalHeader()
        v_header.setDefaultSectionSize(26)
        v_header.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)

        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        # --- QStackedWidget switches between placeholder and table ----------
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._placeholder)  # page 0
        self._stack.addWidget(self._table)         # page 1
        self._stack.setCurrentIndex(0)

        # --- Outer layout ---------------------------------------------------
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack)

    # -------------------------------------------------------------------
    # Public API  (signatures preserved exactly)
    # -------------------------------------------------------------------

    def set_single_sweep_mode(self, enabled: bool) -> None:
        """Show or hide Reverse-direction rows."""
        self._single_sweep_mode = enabled
        self._apply_row_visibility()

    def reset_channels(
        self, channels: List[int], labels_units: List[Tuple[str, str]]
    ) -> None:
        """
        Rebuild the table rows for *channels* using *labels_units* as the
        metric column definitions.
        """
        self._labels_units = labels_units or self.DEFAULT_LABELS_UNITS

        # ---- column map ----------------------------------------------------
        self._col_map.clear()
        for idx, (label, _unit) in enumerate(self._labels_units):
            self._col_map[label] = idx

        num_cols = 1 + len(self._labels_units)

        # ---- row map (always both directions) ------------------------------
        self._row_map.clear()
        for ch in sorted(channels):
            self._row_map[(ch, "Forward")] = len(self._row_map)
            self._row_map[(ch, "Reverse")] = len(self._row_map)

        num_rows = len(self._row_map)

        # ---- configure table dimensions ------------------------------------
        self._table.setRowCount(num_rows)
        self._table.setColumnCount(num_cols)

        # ---- column headers ------------------------------------------------
        headers = ["Channel / Direction"]
        for label, unit in self._labels_units:
            headers.append(self._format_header(label, unit))
        self._table.setHorizontalHeaderLabels(headers)

        # ---- populate rows -------------------------------------------------
        for (ch, direction), row in self._row_map.items():
            # Column 0 – channel label with vibrant left-colour indicator
            label_text = f"Ch {ch} · {direction}"
            label_item = QtWidgets.QTableWidgetItem(label_text)
            label_item.setFlags(
                QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
            )
            label_item.setData(QtCore.Qt.UserRole, (ch, direction))

            hex_colour = self.CHANNEL_COLORS.get(ch)
            if hex_colour:
                colour = QtGui.QColor(hex_colour)

                # Bold text in the channel accent colour
                font = label_item.font()
                font.setBold(True)
                label_item.setFont(font)
                label_item.setForeground(colour)

                # Small coloured bar icon as a left-edge indicator
                indicator = QtGui.QPixmap(_INDICATOR_WIDTH, _INDICATOR_HEIGHT)
                indicator.fill(colour)
                label_item.setData(QtCore.Qt.DecorationRole, QtGui.QIcon(indicator))

            self._table.setItem(row, 0, label_item)

            # Metric columns – em-dash placeholder until data arrives
            for col in range(1, num_cols):
                item = QtWidgets.QTableWidgetItem("—")
                item.setTextAlignment(
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                )
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                )
                self._table.setItem(row, col, item)

        # ---- finalise ------------------------------------------------------
        self._apply_row_visibility()

        # All columns size to content; horizontal scrollbar appears if needed
        hh = self._table.horizontalHeader()
        for col in range(num_cols):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)

        # Let columns measure, then enforce font-metrics minimums
        self._table.resizeColumnsToContents()
        fm = self.fontMetrics()
        min_label = fm.horizontalAdvance("Ch 6 · Reverse") + fm.horizontalAdvance("XX")
        if self._table.columnWidth(0) < min_label:
            self._table.setColumnWidth(0, min_label)

        self._update_stack_page()

    def analysis(self, data: Dict) -> None:
        """
        Update the table with analysis results for one (channel, direction).

        *data* must contain at minimum ``Channel`` (int) and ``Direction``
        (str); remaining keys are metric labels matching ``_labels_units``.
        """
        try:
            channel = int(data.get("Channel"))
            direction = data.get("Direction", "Forward")

            row = self._row_map.get((channel, direction))
            if row is None:
                logger.debug(f"No row for Ch{channel} {direction}")
                return

            for label, value in data.items():
                if label in ("Channel", "Direction"):
                    continue

                col = self._col_map.get(label)
                if col is None:
                    continue

                unit = self._labels_units[col][1]
                text = self._format_value(value, unit)

                item = self._table.item(row, 1 + col)
                if item:
                    item.setText(text)
                else:
                    new_item = QtWidgets.QTableWidgetItem(text)
                    new_item.setTextAlignment(
                        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                    )
                    new_item.setFlags(
                        QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
                    )
                    self._table.setItem(row, 1 + col, new_item)

        except Exception as e:
            logger.warning(f"Analysis update failed: {e}")

    def set_active_channel(
        self, channel: int, direction: str = "Forward"
    ) -> None:
        """Select and scroll to the row for *channel* / *direction*."""
        row = self._row_map.get((channel, direction))
        if row is None:
            return
        self._table.selectRow(row)
        item = self._table.item(row, 0)
        if item:
            self._table.scrollToItem(
                item, QtWidgets.QAbstractItemView.PositionAtCenter
            )

    def clear_all(self) -> None:
        """Reset every metric cell to its zero value."""
        if not self._labels_units:
            self._labels_units = self.DEFAULT_LABELS_UNITS

        for _key, row in self._row_map.items():
            for col_idx, (_label, _unit) in enumerate(self._labels_units):
                item = self._table.item(row, 1 + col_idx)
                if item:
                    item.setText("—")

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _apply_row_visibility(self) -> None:
        """Show / hide Reverse rows based on ``_single_sweep_mode``."""
        for (_ch, direction), row in self._row_map.items():
            hidden = self._single_sweep_mode and direction == "Reverse"
            self._table.setRowHidden(row, hidden)

    def _update_stack_page(self) -> None:
        """Show the placeholder when no channels are loaded, else the table."""
        self._stack.setCurrentIndex(0 if not self._row_map else 1)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        """Forward a row click as ``row_selected(channel, direction)``."""
        item = self._table.item(row, 0)
        if item is None:
            return
        ch, direction = item.data(QtCore.Qt.UserRole)
        self.row_selected.emit(ch, direction)

    def _on_selection_changed(self) -> None:
        """Dynamically colour the selection highlight to match the channel."""
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._table.setStyleSheet(self._base_stylesheet)
            return
        row = rows[0].row()
        item = self._table.item(row, 0)
        if item is None:
            self._table.setStyleSheet(self._base_stylesheet)
            return
        ch, _direction = item.data(QtCore.Qt.UserRole)
        hex_colour = self.CHANNEL_COLORS.get(ch, "#053a46")
        self._table.setStyleSheet(
            self._base_stylesheet
            + f" QTableWidget::item:selected {{"
              f"  background-color: {hex_colour};"
              f"  color: white;"
              f"}}"
        )

    # -------------------------------------------------------------------
    # Static formatting helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _format_header(label: str, unit: str) -> str:
        """Build a column header string, e.g. ``"EFF (%)"``."""
        return f"{label} ({unit})" if unit else label

    @staticmethod
    def _format_value(value, _unit: str = "") -> str:
        """Format a numeric metric value for compact display.
        Headers already carry units, so only the number is shown.
        """
        if not isinstance(value, (int, float)):
            return str(value)

        if value == 0:
            return "0.00" if isinstance(value, float) else "0"
        if abs(value) >= 1e4 or (abs(value) < 1e-3):
            return f"{value:.3e}"
        if abs(value) < 0.1:
            return f"{value:.4f}"
        return f"{value:.2f}"
