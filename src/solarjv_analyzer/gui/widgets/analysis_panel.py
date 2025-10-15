from PyQt5 import QtWidgets, QtCore
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class AnalysisPanel(QtWidgets.QWidget):
    """
    A right-side panel that displays per-channel analysis in a compact table.

    This widget uses vertical tabs, one for each measurement channel. Each tab
    contains a 2-column QTableWidget to display the calculated performance
    metrics and their corresponding values and units.
    """

    DEFAULT_LABELS_UNITS = [
        ("EFF - Efficency", "%"),
        ("FF- fill factor", "%"),
        ("Voc - open circuit volatge", "mV"),
        ("Jsc - short circ. current density", "mA/cm2"),
        ("Vmax", "mV"),
        ("Jmax", "mA/cm2"),
        ("Isc - short circ. current", "A"),
        ("Rsc - short circ. resistence", "Ohm"),
        ('Roc  open ""', "Ohm"),
        ("A - Area", "cm2"),
        ("Incd. Pwr", "mW/cm2"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        small_font = self.font()
        small_font.setPointSize(10)
        self.setFont(small_font)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._labels_units: List[Tuple[str, str]] = []
        self._tables: Dict[int, QtWidgets.QTableWidget] = {}

        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)

        self._group = QtWidgets.QGroupBox("Channel Analysis", self)
        g_layout = QtWidgets.QVBoxLayout(self._group)
        g_layout.setContentsMargins(6, 6, 6, 6)
        g_layout.addWidget(self._tabs)
        self._group.setMinimumWidth(300)
        self._group.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._group)

    def _build_table(self) -> QtWidgets.QTableWidget:
        """Creates and configures a new 2-column table for displaying metrics."""
        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        table.verticalHeader().setDefaultSectionSize(22)
        return table

    def reset_channels(self, channels: List[int], labels_units: List[Tuple[str, str]]) -> None:
        """
        Prepares the UI with tabs for the selected channels.

        This method clears any existing tabs and creates a new tab with a table
        for each channel specified in the list.

        Args:
            channels: A list of channel numbers to create tabs for.
            labels_units: A list of (label, unit) tuples for the table rows.
        """
        self._labels_units = labels_units or self.DEFAULT_LABELS_UNITS
        self._tables.clear()
        while self._tabs.count():
            w = self._tabs.widget(0)
            self._tabs.removeTab(0)
            w.deleteLater()

        for ch in sorted(channels):
            table = self._build_table()
            table.setRowCount(len(self._labels_units))

            for row, (label, unit) in enumerate(self._labels_units):
                item_label = QtWidgets.QTableWidgetItem(label)
                item_label.setFlags(QtCore.Qt.ItemIsEnabled)

                default_text = "0" if unit not in ("cm2",) else "0.00"
                item_value = QtWidgets.QTableWidgetItem(f"{default_text} {unit}".strip())
                item_value.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                item_value.setFlags(QtCore.Qt.ItemIsEnabled)

                table.setItem(row, 0, item_label)
                table.setItem(row, 1, item_value)

            self._tabs.addTab(table, f"Ch {ch}")
            self._tables[ch] = table

    def analysis(self, data: Dict) -> None:
        """
        Updates a channel's table with the provided analysis metrics.

        Args:
            data: A dictionary containing the metrics. Must include a 'Channel'
                  key to identify the correct tab to update.
        """
        try:
            ch = int(data.get("Channel"))
            table = self._tables.get(ch)
            if not table:
                logger.warning(f"Analysis update called for Channel {ch}, but no table was found.")
                return

            label_to_row = {label: idx for idx, (label, _u) in enumerate(self._labels_units)}

            for label, value in data.items():
                if label in label_to_row:
                    row = label_to_row[label]
                    unit = self._labels_units[row][1]

                    if isinstance(value, (int, float)):
                        if value == 0:
                            disp = "0"
                        elif abs(value) >= 1e4 or (abs(value) < 1e-3 and value != 0):
                            disp = f"{value:.3e}"
                        else:
                            disp = f"{value:.4f}"
                        text = f"{disp} {unit}".strip()
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

        except Exception as e:
            logger.warning(f"AnalysisPanel update failed: {e}")

    def clear_all(self) -> None:
        """Resets all displayed values to zeros while keeping tabs."""
        if not self._labels_units:
            self._labels_units = self.DEFAULT_LABELS_UNITS
        for _ch, table in self._tables.items():
            for row, (_label, unit) in enumerate(self._labels_units):
                zero_text = f"0.00 {unit}".strip() if unit in ("cm2",) else f"0 {unit}".strip()
                item = table.item(row, 1)
                if item is not None:
                    item.setText(zero_text)