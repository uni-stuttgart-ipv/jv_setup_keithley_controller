"""
Analysis Panel for J-V Measurement Results

Displays per-channel solar cell metrics in a tabbed table format.
For dual sweep mode: Each channel has Forward and Reverse subtabs.
For single sweep mode: Each channel shows metrics directly (no subtabs).
"""

import logging
from typing import Dict, List, Tuple

from PyQt5 import QtWidgets, QtCore

logger = logging.getLogger(__name__)


class AnalysisPanel(QtWidgets.QWidget):
    """
    Tabbed panel displaying analysis metrics for each measured channel.
    """

    DEFAULT_LABELS_UNITS = [
        ("EFF", "%"),
        ("FF", "%"),
        ("Voc", "mV"),
        ("Jsc", "mA/cm2"),
        ("Vmax", "mV"),
        ("Jmax", "mA/cm2"),
        ("Isc", "A"),
        ("Rsh", "Ohm"),
        ("Rs", "Ohm"),
        ("Area", "cm2"),
        ("Incd. Pwr", "mW/cm2"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        small_font = self.font()
        small_font.setPointSize(10)
        self.setFont(small_font)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        self._labels_units: List[Tuple[str, str]] = []
        self._tables: Dict[int, QtWidgets.QTableWidget] = {}
        self._direction_tabs: Dict[int, QtWidgets.QTabWidget] = {}
        self._single_sweep_mode = False

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)

        self._group = QtWidgets.QGroupBox("Channel Analysis", self)
        group_layout = QtWidgets.QVBoxLayout(self._group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.addWidget(self._tabs)

        self._group.setMinimumWidth(300)
        self._group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._group)

    def set_single_sweep_mode(self, enabled: bool):
        """Set whether we are in single sweep mode (no Forward/Reverse subtabs)."""
        self._single_sweep_mode = enabled

    def _build_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])

        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        table.verticalHeader().setDefaultSectionSize(22)

        return table

    def _populate_table(self, table: QtWidgets.QTableWidget):
        """Populate table with metric labels and default zero values."""
        table.setRowCount(len(self._labels_units))

        for row, (label, unit) in enumerate(self._labels_units):
            label_item = QtWidgets.QTableWidgetItem(label)
            label_item.setFlags(QtCore.Qt.ItemIsEnabled)

            default_text = "0" if unit not in ("cm2",) else "0.00"
            value_item = QtWidgets.QTableWidgetItem(f"{default_text} {unit}".strip())
            value_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            value_item.setFlags(QtCore.Qt.ItemIsEnabled)

            table.setItem(row, 0, label_item)
            table.setItem(row, 1, value_item)

    def reset_channels(self, channels: List[int],
                       labels_units: List[Tuple[str, str]]) -> None:
        """
        Rebuild the tab interface for the specified channels.
        """
        self._labels_units = labels_units or self.DEFAULT_LABELS_UNITS
        self._tables.clear()
        self._direction_tabs.clear()

        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            widget.deleteLater()

        for channel in sorted(channels):
            if self._single_sweep_mode:
                # Single sweep mode: one table per channel (no subtabs)
                table = self._build_table()
                self._populate_table(table)
                self._tabs.addTab(table, f"Ch {channel}")
                self._tables[channel] = table
            else:
                # Dual sweep mode: two subtabs per channel
                direction_tabs = QtWidgets.QTabWidget()
                
                forward_table = self._build_table()
                self._populate_table(forward_table)
                direction_tabs.addTab(forward_table, "Forward")
                
                reverse_table = self._build_table()
                self._populate_table(reverse_table)
                direction_tabs.addTab(reverse_table, "Reverse")
                
                self._tabs.addTab(direction_tabs, f"Ch {channel}")
                self._direction_tabs[channel] = direction_tabs
                self._tables[channel] = {
                    "Forward": forward_table,
                    "Reverse": reverse_table
                }

    def analysis(self, data: Dict) -> None:
        """Update the table with new analysis results."""
        try:
            channel = int(data.get("Channel"))
            direction = data.get("Direction", "Forward")
            
            logger.debug(f"Analysis update: Ch{channel}, Dir={direction}")
            
            if self._single_sweep_mode:
                table = self._tables.get(channel)
                if not table:
                    logger.warning(f"No table for Channel {channel}")
                    return
                self._update_table(table, data)
            else:
                tables = self._tables.get(channel)
                if not tables:
                    logger.warning(f"No tables for Channel {channel}")
                    return
                table = tables.get(direction)
                if not table:
                    logger.warning(f"No table for Channel {channel}, Direction {direction}")
                    return
                self._update_table(table, data)
                
        except Exception as e:
            logger.warning(f"Analysis update failed: {e}")

    def _update_table(self, table: QtWidgets.QTableWidget, data: Dict):
        """Update a single table with metrics data."""
        label_to_row = {
            label: idx for idx, (label, _) in enumerate(self._labels_units)
        }

        for label, value in data.items():
            if label in ["Channel", "Direction"]:
                continue
                
            if label not in label_to_row:
                continue

            row = label_to_row[label]
            unit = self._labels_units[row][1]

            if isinstance(value, (int, float)):
                if value == 0:
                    display = "0"
                elif abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0):
                    display = f"{value:.3e}"
                else:
                    display = f"{value:.4f}"
                text = f"{display} {unit}".strip()
            else:
                text = f"{value} {unit}".strip()

            item = table.item(row, 1)
            if item:
                item.setText(text)
            else:
                new_item = QtWidgets.QTableWidgetItem(text)
                new_item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                new_item.setFlags(QtCore.Qt.ItemIsEnabled)
                table.setItem(row, 1, new_item)

    def set_active_channel(self, channel: int, direction: str = "Forward") -> None:
        """Switch to the specified channel (and direction for dual mode)."""
        target_text = f"Ch {channel}"
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == target_text:
                self._tabs.setCurrentIndex(i)
                if not self._single_sweep_mode:
                    direction_tabs = self._tabs.widget(i)
                    if direction_tabs:
                        dir_idx = 0 if direction == "Forward" else 1
                        direction_tabs.setCurrentIndex(dir_idx)
                return

    def clear_all(self) -> None:
        """Reset all displayed values to zeros."""
        if not self._labels_units:
            self._labels_units = self.DEFAULT_LABELS_UNITS
        
        if self._single_sweep_mode:
            for table in self._tables.values():
                for row, (_, unit) in enumerate(self._labels_units):
                    zero_text = "0.00 cm2" if unit == "cm2" else f"0 {unit}".strip()
                    item = table.item(row, 1)
                    if item:
                        item.setText(zero_text)
        else:
            for tables in self._tables.values():
                if isinstance(tables, dict):
                    # Dual sweep mode: tables is {direction: QTableWidget}
                    for table in tables.values():
                        for row, (_, unit) in enumerate(self._labels_units):
                            zero_text = "0.00 cm2" if unit == "cm2" else f"0 {unit}".strip()
                            item = table.item(row, 1)
                            if item:
                                item.setText(zero_text)
                else:
                    # Shouldn't happen, but handle gracefully
                    for row, (_, unit) in enumerate(self._labels_units):
                        zero_text = "0.00 cm2" if unit == "cm2" else f"0 {unit}".strip()
                        item = tables.item(row, 1)
                        if item:
                            item.setText(zero_text)