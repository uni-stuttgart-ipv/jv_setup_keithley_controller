"""
Analysis Panel for J-V Measurement Results

Displays per-channel solar cell metrics in a tabbed table format.
Supports real-time updates during measurement and persistent display
of previously measured channels.
"""

import logging
from typing import Dict, List, Tuple

from PyQt5 import QtWidgets, QtCore

logger = logging.getLogger(__name__)


class AnalysisPanel(QtWidgets.QWidget):
    """
    Tabbed panel displaying analysis metrics for each measured channel.

    Each channel has its own table showing calculated solar cell parameters
    such as efficiency, fill factor, Voc, Jsc, and others.
    """

    # Default metrics displayed when no custom labels are provided
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
        ("A", "cm2"),
        ("Incd. Pwr", "mW/cm2"),
    ]

    def __init__(self, parent=None):
        """Initialize the analysis panel."""
        super().__init__(parent)

        # Set smaller font for compact display
        small_font = self.font()
        small_font.setPointSize(10)
        self.setFont(small_font)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Internal state
        self._labels_units: List[Tuple[str, str]] = []
        self._tables: Dict[int, QtWidgets.QTableWidget] = {}

        # Create tab widget for channels
        self._tabs = QtWidgets.QTabWidget(self)
        self._tabs.setTabPosition(QtWidgets.QTabWidget.West)
        self._tabs.setDocumentMode(True)
        self._tabs.setMovable(False)

        # Group box for visual grouping
        self._group = QtWidgets.QGroupBox("Channel Analysis", self)
        group_layout = QtWidgets.QVBoxLayout(self._group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.addWidget(self._tabs)

        self._group.setMinimumWidth(300)
        self._group.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Expanding
        )

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._group)

    # -------------------------------------------------------------------------
    # Table Management
    # -------------------------------------------------------------------------

    def _build_table(self) -> QtWidgets.QTableWidget:
        """
        Create and configure a new metrics table.

        Returns:
            QTableWidget configured for metric display
        """
        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])

        # Appearance settings
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )

        # Header configuration
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setDefaultAlignment(
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
        )
        header.setMinimumSectionSize(80)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)

        # Row height
        table.verticalHeader().setDefaultSectionSize(22)

        return table

    # -------------------------------------------------------------------------
    # Channel Management
    # -------------------------------------------------------------------------

    def reset_channels(self, channels: List[int],
                       labels_units: List[Tuple[str, str]]) -> None:
        """
        Rebuild the tab interface for the specified channels.

        Args:
            channels: List of channel numbers to display
            labels_units: List of (label, unit) tuples for metrics
        """
        self._labels_units = labels_units or self.DEFAULT_LABELS_UNITS
        self._tables.clear()

        # Clear existing tabs
        while self._tabs.count():
            widget = self._tabs.widget(0)
            self._tabs.removeTab(0)
            widget.deleteLater()

        # Create new tabs for each channel
        for channel in sorted(channels):
            table = self._build_table()
            table.setRowCount(len(self._labels_units))

            # Populate metric labels
            for row, (label, unit) in enumerate(self._labels_units):
                label_item = QtWidgets.QTableWidgetItem(label)
                label_item.setFlags(QtCore.Qt.ItemIsEnabled)

                # Default placeholder value
                default_text = "0" if unit not in ("cm2",) else "0.00"
                value_item = QtWidgets.QTableWidgetItem(f"{default_text} {unit}".strip())
                value_item.setTextAlignment(
                    QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                )
                value_item.setFlags(QtCore.Qt.ItemIsEnabled)

                table.setItem(row, 0, label_item)
                table.setItem(row, 1, value_item)

            self._tabs.addTab(table, f"Ch {channel}")
            self._tables[channel] = table

    # -------------------------------------------------------------------------
    # Data Update
    # -------------------------------------------------------------------------

    def analysis(self, data: Dict) -> None:
        """
        Update the table with new analysis results for a channel.

        Args:
            data: Dictionary containing 'Channel' key and metric values
        """
        try:
            channel = int(data.get("Channel"))
            table = self._tables.get(channel)

            if not table:
                # Channel not yet initialized - this can happen during
                # partial loading or before reset_channels is called
                return

            # Build lookup map for label to row index
            label_to_row = {
                label: idx for idx, (label, _) in enumerate(self._labels_units)
            }

            for label, value in data.items():
                if label not in label_to_row:
                    continue

                row = label_to_row[label]
                unit = self._labels_units[row][1]

                # Format the value for display
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

                # Update or create the value cell
                item = table.item(row, 1)
                if item:
                    item.setText(text)
                else:
                    new_item = QtWidgets.QTableWidgetItem(text)
                    new_item.setTextAlignment(
                        QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
                    )
                    new_item.setFlags(QtCore.Qt.ItemIsEnabled)
                    table.setItem(row, 1, new_item)

        except Exception as e:
            logger.warning(f"Analysis update failed: {e}")

    def set_active_channel(self, channel: int) -> None:
        """
        Switch the active tab to the specified channel.

        Args:
            channel: Channel number to display
        """
        target_text = f"Ch {channel}"
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == target_text:
                self._tabs.setCurrentIndex(i)
                return

    # -------------------------------------------------------------------------
    # Reset Functionality
    # -------------------------------------------------------------------------

    def clear_all(self) -> None:
        """Reset all displayed values to zeros while preserving tabs."""
        if not self._labels_units:
            self._labels_units = self.DEFAULT_LABELS_UNITS

        for table in self._tables.values():
            for row, (_, unit) in enumerate(self._labels_units):
                # Format zero based on unit type
                if unit == "cm2":
                    zero_text = f"0.00 {unit}".strip()
                else:
                    zero_text = f"0 {unit}".strip()

                item = table.item(row, 1)
                if item is not None:
                    item.setText(zero_text)