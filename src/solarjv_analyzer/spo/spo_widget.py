"""
SPO (Set-Point Operation) Widgets

Two widgets are defined here:
    SpoParameterTab - the SPO configuration fields (Hold Duration, Sampling
        Interval, Pre-conditioning, Channel, Vmpp, Quick JV). This is
        swapped into the sidebar's existing "Parameters" tab slot in place
        of the JV ParameterTab while SPO mode is active.
    SpoWidget - the live Power-vs-Time plot and live metrics, which
        replaces the JV plot/browser/analysis display area while SPO mode
        is active.

Both follow the same modern, flat visual style (QGroupBox cards / plain
form layouts, same color palette) already defined by JVAnalyzerWindow's
global stylesheet, which cascades down to these widgets automatically.
"""

import logging
import os

from typing import List

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.spo.spo_procedure import SpoWorker
from solarjv_analyzer.gui.widgets.toggle_switch import ToggleSwitch
from solarjv_analyzer.gui.widgets.channel_pinout import build_pinout_label

logger = logging.getLogger(__name__)


class _TightStackedWidget(QtWidgets.QStackedWidget):
    """QStackedWidget that sizes to the *current* page, not the tallest page.

    The base QStackedWidget/QStackedLayout reports hasHeightForWidth() as
    True and computes heightForWidth() as the maximum across *all* pages
    (including hidden ones), which box layouts prefer over sizeHint(). That
    left dead space below the short "Manual" page whenever the much taller
    "Quick JV" page (sweep params, mini-plot, results grid) was the other
    page in the stack.

    Simply delegating heightForWidth()/hasHeightForWidth() to the current
    page does NOT work: Qt's QWidgetItem bypasses a widget's own
    heightForWidth() override and queries widget.layout().totalHeightForWidth()
    directly whenever the widget has its own layout — which QStackedWidget
    always does internally — so it still lands on the unfixed, all-pages-
    aggregating internal QStackedLayout. Instead, hasHeightForWidth() is
    disabled here (forcing callers onto the sizeHint()/minimumSizeHint()
    path, which Qt *does* route through our override), and those two
    methods bake in the current page's width-dependent wrapped height
    (via its own heightForWidth()) using this widget's current width, so
    word-wrapped labels (e.g. the Manual hint text) still get their
    correct height reserved.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def _page_size(self, base_size_getter):
        w = self.currentWidget()
        if w is None:
            return base_size_getter(super())
        size = base_size_getter(w)
        if w.hasHeightForWidth():
            width = self.width() if self.width() > 0 else size.width()
            hfw = w.heightForWidth(width)
            if hfw > 0:
                size = QtCore.QSize(size.width(), hfw)
        return size

    def sizeHint(self):
        return self._page_size(lambda obj: obj.sizeHint())

    def minimumSizeHint(self):
        return self._page_size(lambda obj: obj.minimumSizeHint())

    def hasHeightForWidth(self):
        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            # Wrapped-label height depends on width; re-derive it once the
            # new width is known so sizeHint() stays accurate.
            self.updateGeometry()


def _strip_unit(header: str) -> str:
    """`mean_power_mw (mW)` -> `mean_power_mw`."""
    name = header.strip()
    if name.endswith(")") and "(" in name:
        name = name[:name.rindex("(")].strip()
    return name


def _parse_metrics_block(lines: list) -> dict:
    """Read the metrics from either report layout.

    Current reports use the column-wise `[[ ANALYSIS SUMMARY ]]` block that
    matches the J-V report. Reports written before that change used a
    row-wise `[[ SPO METRICS ]]` block. Both are read here so a file saved
    last week still opens.
    """
    def as_number(text):
        try:
            return float(text)
        except ValueError:
            return text.strip()

    if "[[ ANALYSIS SUMMARY ]]" in lines:
        start = lines.index("[[ ANALYSIS SUMMARY ]]")
        if len(lines) > start + 2:
            headers = [h.strip() for h in lines[start + 1].split(",")]
            values = [v.strip() for v in lines[start + 2].split(",")]
            metrics = {}
            for header, value in zip(headers, values):
                if header.lower() == "channel":
                    continue
                metrics[_strip_unit(header)] = as_number(value)
            return metrics

    if "[[ SPO METRICS ]]" in lines:
        start = lines.index("[[ SPO METRICS ]]") + 2      # skip column header
        metrics = {}
        for row in lines[start:]:
            if not row.strip() or row.startswith("[["):
                break
            parts = row.split(",")
            if len(parts) >= 2:
                metrics[parts[0].strip()] = as_number(parts[1])
        return metrics

    return {}


class SpoParameterTab(QtWidgets.QWidget):
    """
    SPO configuration fields, designed to occupy the sidebar's
    "Parameters" tab slot in place of the JV ParameterTab while SPO mode
    is active. Mirrors the existing ParameterTab's plain, borderless
    form-layout style (no extra QGroupBox chrome needed since it already
    lives inside a QTabWidget page).
    """

    # Emitted whenever the Vmpp field becomes valid/invalid, so the window
    # can enable/disable the "Start SPO" button accordingly.
    vmpp_ready = QtCore.pyqtSignal(bool)

    # Fixed, fast single-sweep parameters used only to locate Vmpp quickly.
    QUICK_JV_START = 1.0
    QUICK_JV_STOP = -0.1
    QUICK_JV_STEP = 0.02
    QUICK_JV_RATE = 1.0  # V/s

    def __init__(self, main_window, parent=None):
        """
        Args:
            main_window: The JVAnalyzerWindow instance, used to reach the
                shared InstrumentManager and parameter tabs for Quick JV.
        """
        super().__init__(parent)
        self.main_window = main_window
        self._quick_jv_worker = None
        self._quick_jv_running = False
        self._vmpp_source = None  # None | "manual" | "quick_jv"
        self._quick_jv_voltages = []
        self._quick_jv_currents = []
        self._quick_jv_expected_points = 0
        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        form.setVerticalSpacing(8)

        self.hold_duration_input = QtWidgets.QLineEdit("300")
        form.addRow("Hold Duration (s):", self.hold_duration_input)

        self.sampling_interval_input = QtWidgets.QLineEdit("1.0")
        form.addRow("Sampling Interval (s):", self.sampling_interval_input)

        self.preconditioning_input = QtWidgets.QLineEdit("5.0")
        form.addRow("Pre-conditioning (s):", self.preconditioning_input)

        self._create_channel_selector(form)

        # ---- Hold Voltage section (card with Quick JV) --------------------
        self._create_hold_voltage_card(form)

    def _create_channel_selector(self, parent_layout):
        """
        Create the SPO "Channel Selection" card using the same reference
        pinout image + toggle-switch style as the main window's
        ParameterTab, but as a single-selection group (SPO holds only one
        channel at a time) — no "Select All" toggle.

        Layout (physical), matching ParameterTab / channel_pinout.png:
            Ch3   Ch4
            Ch2   Ch5
            Ch1   Ch6
        """
        self.channels: List[ToggleSwitch] = []
        self.channel_number_labels: List[QtWidgets.QLabel] = []

        card = QtWidgets.QGroupBox("Channel Selection")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setSpacing(10)

        body_layout = QtWidgets.QHBoxLayout()
        body_layout.setSpacing(16)

        pinout_column = QtWidgets.QVBoxLayout()
        pinout_column.addWidget(build_pinout_label())
        caption = QtWidgets.QLabel("Reference Pinout")
        caption.setAlignment(QtCore.Qt.AlignCenter)
        caption.setStyleSheet("color: #64748b; font-size: 8pt;")
        pinout_column.addWidget(caption)
        body_layout.addLayout(pinout_column)

        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.VLine)
        divider.setFrameShadow(QtWidgets.QFrame.Sunken)
        body_layout.addWidget(divider)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        mapping = {
            3: (0, 0), 4: (0, 1),
            2: (1, 0), 5: (1, 1),
            1: (2, 0), 6: (2, 1),
        }

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

        for ch_num, (row, col) in mapping.items():
            idx = ch_num - 1
            pair_layout = QtWidgets.QHBoxLayout()
            pair_layout.setSpacing(8)
            pair_layout.addWidget(self.channel_number_labels[idx])
            pair_layout.addWidget(self.channels[idx])
            grid.addLayout(pair_layout, row, col)

        body_layout.addLayout(grid)
        body_layout.addStretch(1)
        card_layout.addLayout(body_layout)

        parent_layout.addRow(card)

        # Default to Channel 1 selected, matching the SPO procedure default.
        self.channels[0].setChecked(True)
        self._update_channel_number_chip(0, True)

        for idx, toggle in enumerate(self.channels):
            toggle.toggled.connect(lambda checked, i=idx: self._update_channel_number_chip(i, checked))

    def _update_channel_number_chip(self, index: int, checked: bool):
        """Style the channel number chip: soft green background when the
        channel is selected, plain/muted when it isn't."""
        label = self.channel_number_labels[index]
        if checked:
            label.setStyleSheet(
                "background-color: #d1e3e9; color: #053a46; font-weight: 600;"
                "border-radius: 14px;"
            )
        else:
            label.setStyleSheet(
                "background-color: transparent; color: #94a3b8; font-weight: 600;"
            )

    def get_selected_channel(self) -> int:
        """Get the currently selected channel number (1-based)."""
        checked_id = self.channel_button_group.checkedId()
        return checked_id if checked_id != -1 else 1

    # -------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """Collect SPO-specific parameters from the configuration fields."""
        return {
            'hold_voltage': float(self.vmpp_input.text() or 0.0),
            'hold_duration': float(self.hold_duration_input.text() or 300.0),
            'sampling_interval': float(self.sampling_interval_input.text() or 1.0),
            'preconditioning_time': float(self.preconditioning_input.text() or 5.0),
            'active_channel': self.get_selected_channel(),
        }

    def has_valid_hold_voltage(self) -> bool:
        """True if the Vmpp field currently holds a parsable float."""
        try:
            float(self.vmpp_input.text())
            return True
        except (TypeError, ValueError):
            return False

    def _on_vmpp_changed(self, _text):
        # Track provenance: user edits reset to "manual"; Quick JV sets
        # _vmpp_source = "quick_jv" *before* calling setText so this
        # handler keeps the green badge.
        if self._vmpp_source != "quick_jv":
            self._vmpp_source = "manual" if self.has_valid_hold_voltage() else None
        self._update_vmpp_badge()
        self.vmpp_ready.emit(self.has_valid_hold_voltage())

    def set_config_enabled(self, enabled: bool):
        """Enable/disable all configuration inputs (locked while running)."""
        for widget in (
            self.hold_duration_input, self.sampling_interval_input,
            self.preconditioning_input, self.vmpp_input,
            self.manual_mode_button, self.quick_jv_mode_button,
            self.quick_jv_start_input, self.quick_jv_stop_input,
            self.quick_jv_rate_input, self.quick_jv_button,
            *self.channels,
        ):
            widget.setEnabled(enabled)

    # -------------------------------------------------------------------
    # Hold Voltage Card (Manual / Quick JV)
    # -------------------------------------------------------------------

    def _create_hold_voltage_card(self, form):
        """Build the reinvented Hold Voltage card with Manual/Quick JV toggle,
        shared Vmpp entry, live mini-plot, progress bar, and results grid."""
        card = QtWidgets.QGroupBox("Hold Voltage")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setSpacing(8)

        # ---- Row 1: Shared Vmpp entry + source badge --------------------
        vmpp_row = QtWidgets.QHBoxLayout()
        vmpp_row.setSpacing(8)

        self.vmpp_badge = QtWidgets.QLabel("No Vmpp")
        self.vmpp_badge.setFixedWidth(78)
        self.vmpp_badge.setAlignment(QtCore.Qt.AlignCenter)

        self.vmpp_input = QtWidgets.QLineEdit()
        self.vmpp_input.setPlaceholderText("Enter hold voltage (V)")
        self.vmpp_input.textChanged.connect(self._on_vmpp_changed)

        self._update_vmpp_badge()

        vmpp_unit = QtWidgets.QLabel("V")
        vmpp_unit.setStyleSheet(
            "color: #64748b; font-size: 12px; font-weight: 600; background: transparent;"
        )

        vmpp_row.addWidget(self.vmpp_badge)
        vmpp_row.addWidget(self.vmpp_input, stretch=1)
        vmpp_row.addWidget(vmpp_unit)
        layout.addLayout(vmpp_row)

        # ---- Row 2: Mode toggle -----------------------------------------
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(6)

        # Object name is HoldModeButton, not the generic ModeButton: these two
        # inherited the inner-tab ModeButton style, whose unchecked state is
        # pale grey on transparent — indistinguishable from a disabled button,
        # so operators did not realise "Quick JV" was a choice they could make.
        # See QPushButton#HoldModeButton in the window's stylesheet.
        self.manual_mode_button = QtWidgets.QPushButton("Manual")
        self.manual_mode_button.setCheckable(True)
        self.manual_mode_button.setChecked(True)
        self.manual_mode_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.manual_mode_button.setObjectName("HoldModeButton")
        self.manual_mode_button.setToolTip("Type the hold voltage yourself")

        self.quick_jv_mode_button = QtWidgets.QPushButton("Quick JV")
        self.quick_jv_mode_button.setCheckable(True)
        self.quick_jv_mode_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.quick_jv_mode_button.setObjectName("HoldModeButton")
        self.quick_jv_mode_button.setToolTip(
            "Run a short automatic J-V sweep and take Vmpp from it")

        self._mode_button_group = QtWidgets.QButtonGroup(self)
        self._mode_button_group.setExclusive(True)
        self._mode_button_group.addButton(self.manual_mode_button, 0)
        self._mode_button_group.addButton(self.quick_jv_mode_button, 1)
        self._mode_button_group.buttonClicked.connect(self._on_mode_button_clicked)

        mode_row.addWidget(self.manual_mode_button)
        mode_row.addWidget(self.quick_jv_mode_button)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ---- QStackedWidget: Manual hint / Quick JV panel ----------------
        self.mode_stack = _TightStackedWidget()

        # Page 0: Manual hint
        manual_page = QtWidgets.QWidget()
        manual_layout = QtWidgets.QVBoxLayout(manual_page)
        manual_layout.setContentsMargins(0, 2, 0, 0)
        manual_hint = QtWidgets.QLabel(
            "Enter the hold voltage manually, or switch to Quick JV "
            "to measure Vmpp automatically."
        )
        manual_hint.setStyleSheet(
            "color: #94a3b8; font-size: 11px; background: transparent;"
        )
        manual_hint.setWordWrap(True)
        manual_layout.addWidget(manual_hint)
        self.mode_stack.addWidget(manual_page)  # index 0

        # Page 1: Quick JV panel
        quick_jv_page = QtWidgets.QWidget()
        quick_jv_layout = QtWidgets.QVBoxLayout(quick_jv_page)
        quick_jv_layout.setContentsMargins(0, 2, 0, 0)
        quick_jv_layout.setSpacing(8)

        # Sweep parameter row
        sweep_form = QtWidgets.QFormLayout()
        sweep_form.setVerticalSpacing(6)

        self.quick_jv_start_input = QtWidgets.QDoubleSpinBox()
        self.quick_jv_start_input.setRange(-10.0, 10.0)
        self.quick_jv_start_input.setValue(self.QUICK_JV_START)
        self.quick_jv_start_input.setDecimals(2)
        self.quick_jv_start_input.setSingleStep(0.1)
        self.quick_jv_start_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        sweep_form.addRow("Start (V):", self.quick_jv_start_input)

        self.quick_jv_stop_input = QtWidgets.QDoubleSpinBox()
        self.quick_jv_stop_input.setRange(-10.0, 10.0)
        self.quick_jv_stop_input.setValue(self.QUICK_JV_STOP)
        self.quick_jv_stop_input.setDecimals(2)
        self.quick_jv_stop_input.setSingleStep(0.1)
        self.quick_jv_stop_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        sweep_form.addRow("Stop (V):", self.quick_jv_stop_input)

        self.quick_jv_rate_input = QtWidgets.QDoubleSpinBox()
        self.quick_jv_rate_input.setRange(0.01, 100.0)
        self.quick_jv_rate_input.setValue(self.QUICK_JV_RATE)
        self.quick_jv_rate_input.setDecimals(2)
        self.quick_jv_rate_input.setSingleStep(0.1)
        self.quick_jv_rate_input.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.NoButtons
        )
        sweep_form.addRow("Rate (V/s):", self.quick_jv_rate_input)

        quick_jv_layout.addLayout(sweep_form)

        # Run button
        self.quick_jv_button = QtWidgets.QPushButton("Run Quick JV")
        self.quick_jv_button.setObjectName("QueueButton")
        self.quick_jv_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.quick_jv_button.clicked.connect(self._run_quick_jv)
        quick_jv_layout.addWidget(self.quick_jv_button)

        # Progress bar (hidden until running)
        self.quick_jv_progress = QtWidgets.QProgressBar()
        self.quick_jv_progress.setRange(0, 100)
        self.quick_jv_progress.setValue(0)
        self.quick_jv_progress.setTextVisible(False)
        self.quick_jv_progress.setFixedHeight(6)
        self.quick_jv_progress.setStyleSheet(
            "QProgressBar { background: #f1f5f9; border: none; border-radius: 3px; }"
            "QProgressBar::chunk { background: #053a46; border-radius: 3px; }"
        )
        self.quick_jv_progress.hide()
        quick_jv_layout.addWidget(self.quick_jv_progress)

        # Mini I-V plot
        self.quick_jv_plot = pg.PlotWidget()
        self.quick_jv_plot.setBackground('w')
        self.quick_jv_plot.showGrid(x=True, y=True, alpha=0.3)
        self.quick_jv_plot.setLabel('bottom', 'Voltage', units='V')
        self.quick_jv_plot.setLabel('left', 'Current', units='A')
        self.quick_jv_plot.setFixedHeight(150)
        self.quick_jv_plot.hideButtons()
        self.quick_jv_curve = self.quick_jv_plot.plot(
            [], [], pen=pg.mkPen(color='#053a46', width=2)
        )
        # MPP marker (green scatter, hidden until results arrive)
        self.quick_jv_mpp = pg.ScatterPlotItem(
            [], [], symbol='d', size=12, brush=pg.mkBrush('#16a34a'),
            pen=pg.mkPen(color='#15803d', width=1),
        )
        self.quick_jv_plot.addItem(self.quick_jv_mpp)
        quick_jv_layout.addWidget(self.quick_jv_plot)

        # Results grid
        results_group = QtWidgets.QWidget()
        results_grid = QtWidgets.QGridLayout(results_group)
        results_grid.setContentsMargins(0, 0, 0, 0)
        results_grid.setHorizontalSpacing(6)
        results_grid.setVerticalSpacing(2)

        header_style = "color: #64748b; font-size: 10px; font-weight: 600; background: transparent;"
        value_style = (
            "font-size: 12px; font-weight: 600; color: #1e293b; background: transparent;"
        )

        headers = ["Voc (V)", "Jsc (mA/cm²)", "FF (%)", "Vmpp (V)", "Eff (%)"]
        self.quick_jv_result_labels = {}

        for col, hdr in enumerate(headers):
            lbl = QtWidgets.QLabel(hdr)
            lbl.setStyleSheet(header_style)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            results_grid.addWidget(lbl, 0, col)

            val = QtWidgets.QLabel("—")
            val.setStyleSheet(value_style)
            val.setAlignment(QtCore.Qt.AlignCenter)
            results_grid.addWidget(val, 1, col)
            self.quick_jv_result_labels[hdr] = val

        quick_jv_layout.addWidget(results_group)

        # Status label
        self.quick_jv_status = QtWidgets.QLabel("Idle")
        self.quick_jv_status.setStyleSheet(
            "color: #94a3b8; font-size: 11px; background: transparent;"
        )
        quick_jv_layout.addWidget(self.quick_jv_status)

        self.mode_stack.addWidget(quick_jv_page)  # index 1
        self.mode_stack.setCurrentIndex(0)
        layout.addWidget(self.mode_stack)

        form.addRow(card)

    def _update_vmpp_badge(self):
        """Update the source badge chip color and text."""
        base = (
            "border-radius: 10px; font-size: 10px; font-weight: 600; "
            "padding: 2px 0px;"
        )
        if self.has_valid_hold_voltage():
            if self._vmpp_source == "quick_jv":
                self.vmpp_badge.setStyleSheet(
                    f"{base} color: #053a46; background-color: #d1e3e9;"
                )
                self.vmpp_badge.setText("Quick JV")
            else:
                self.vmpp_badge.setStyleSheet(
                    f"{base} color: #053a46; background-color: #d1e3e9;"
                )
                self.vmpp_badge.setText("Manual")
        else:
            self.vmpp_badge.setStyleSheet(
                f"{base} color: #64748b; background-color: #f1f5f9;"
            )
            self.vmpp_badge.setText("No Vmpp")

    def _on_mode_button_clicked(self, _button=None):
        """Swap between Manual (page 0) and Quick JV (page 1) panels."""
        self.mode_stack.setCurrentIndex(
            0 if self.manual_mode_button.isChecked() else 1
        )
        # Nudge the sidebar to recompute heights immediately (rather than
        # waiting for the next window resize) so switching to the much
        # shorter Manual page doesn't leave dead space below this card.
        relayout = getattr(self.main_window, '_relayout_sidebar', None)
        if callable(relayout):
            relayout()

    # -------------------------------------------------------------------
    # Quick JV (temporary, in-memory sweep solely to locate Vmpp)
    # -------------------------------------------------------------------

    def _run_quick_jv(self):
        """Run a fast JV sweep to auto-detect Vmpp with live plotting."""
        if self._quick_jv_running:
            return

        # Guard against concurrent runs: Quick JV must not overlap with
        # JV queue / SPO / Combined runs that use the same Keithley.
        controller = self.main_window.controller
        if getattr(controller, 'is_busy', False) or getattr(controller, 'spo_running', False):
            QtWidgets.QMessageBox.warning(
                self, "Instrument Busy",
                "Another measurement is already in progress.\n\n"
                "Please wait for it to finish or abort it before running Quick JV."
            )
            return

        # ---- Pre-validate sweep parameters (before touching hardware) ----
        start_v = self.quick_jv_start_input.value()
        stop_v = self.quick_jv_stop_input.value()

        if start_v <= stop_v:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Sweep Range",
                "Start voltage must be greater than stop voltage.\n\n"
                "The Quick JV sweep runs from start to stop."
            )
            return

        expected_points = int(abs(stop_v - start_v) / self.QUICK_JV_STEP) + 1
        if expected_points > 2500:
            QtWidgets.QMessageBox.warning(
                self, "Sweep Too Large",
                f"The sweep range produces {expected_points} points, which "
                f"exceeds the instrument's 2500-point buffer.\n\n"
                f"Reduce the voltage range or increase the step size."
            )
            return

        # Validate device area (compute_jv_metrics raises ValueError if ≤ 0)
        device_area = 0.089
        try:
            device_area = float(
                self.main_window.params_tab.get_parameters().get(
                    'device_area', device_area
                )
            )
        except Exception:
            pass
        if device_area <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Device Area",
                "Device area must be greater than zero for JV analysis."
            )
            return

        # ---- Enter running state ------------------------------------------
        self._quick_jv_running = True
        self.main_window.controller.is_busy = True
        self._quick_jv_voltages = []
        self._quick_jv_currents = []
        self._quick_jv_expected_points = expected_points

        # Clear previous results
        self.quick_jv_curve.setData([], [])
        self.quick_jv_mpp.setData([], [])
        for lbl in self.quick_jv_result_labels.values():
            lbl.setText("—")
            lbl.setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #1e293b; background: transparent;"
            )
        self.quick_jv_progress.setValue(0)
        self.quick_jv_progress.show()
        self.quick_jv_status.setText("Connecting...")

        # Lock controls
        self.quick_jv_button.setEnabled(False)
        self.quick_jv_button.setText("Running Quick JV...")
        self.manual_mode_button.setEnabled(False)
        self.quick_jv_mode_button.setEnabled(False)
        self.quick_jv_start_input.setEnabled(False)
        self.quick_jv_stop_input.setEnabled(False)
        self.quick_jv_rate_input.setEnabled(False)
        self.vmpp_input.setEnabled(False)

        # ---- Connect hardware ---------------------------------------------
        instrument_manager = self.main_window.instrument_manager
        try:
            instrument_manager.connect_keithley(simulation=False)
            instrument_manager.connect_mux(simulation=False)
        except Exception as e:
            self._quick_jv_running = False
            self.main_window.controller.is_busy = False
            self._reset_quick_jv_controls()
            QtWidgets.QMessageBox.critical(
                self, "Connection Failed",
                f"Could not connect to instruments:\n{e}\n\n"
                "Please check that the Keithley is powered on and connected."
            )
            return
        self.main_window.update_instrument_lights()

        channel = self.get_selected_channel()

        # In-memory only: no `results` object is attached, so JVProcedure
        # never writes a file. This cannot interfere with the JV analysis
        # panel or browser.
        proc = JVProcedure(
            instrument=instrument_manager.keithley,
            mux=instrument_manager.mux,
            manager=instrument_manager,
            simulation=False,
            active_channel=channel,
            start_voltage=start_v,
            stop_voltage=stop_v,
            step_size=self.QUICK_JV_STEP,
            sweep_rate=self.quick_jv_rate_input.value(),
            single_sweep_mode=True,
            sweep_direction="Forward",
            device_area=device_area,
            incident_power=self._get_incident_power(),
            check_errors_between_points=False,
        )

        self._quick_jv_worker = SpoWorker(proc)
        self._quick_jv_worker.results_ready.connect(self._on_quick_jv_point)
        self._quick_jv_worker.progress_changed.connect(self._on_quick_jv_progress)
        self._quick_jv_worker.status_changed.connect(self._on_quick_jv_status)
        self._quick_jv_worker.run_finished.connect(self._on_quick_jv_finished)
        self._quick_jv_worker.run_failed.connect(self._on_quick_jv_failed)
        self._quick_jv_worker.finished.connect(
            self._quick_jv_worker.deleteLater
        )
        self._quick_jv_worker.start()

    def _get_incident_power(self) -> float:
        """Incident power from the user's Analysis Settings (was hard-coded
        to 100, which made the Quick JV efficiency ignore the user's value)."""
        try:
            analysis = self.main_window.analysis_settings_tab.get_parameters()
            return float(analysis.get('incident_power', 100.0))
        except Exception:
            return 100.0

    def _on_quick_jv_point(self, record: dict):
        """Append one (V, I) data point and update the live mini-plot."""
        voltage = record.get("Voltage (V)", 0.0)
        current = record.get("Current (A)", 0.0)
        self._quick_jv_voltages.append(voltage)
        self._quick_jv_currents.append(current)
        self.quick_jv_curve.setData(
            self._quick_jv_voltages, self._quick_jv_currents
        )
        # GUI-side progress: JVProcedure emits progress only at 100,
        # so drive the bar from points received vs expected.
        if self._quick_jv_expected_points > 0:
            pct = min(
                99,
                int(
                    len(self._quick_jv_voltages)
                    / self._quick_jv_expected_points
                    * 100
                ),
            )
            self.quick_jv_progress.setValue(pct)

    def _on_quick_jv_progress(self, pct: float):
        """Snap the progress bar to the reported value (forward-compat)."""
        self.quick_jv_progress.setValue(int(pct))

    def _on_quick_jv_status(self, status: str):
        """Update the status label (forward-compat; JVProcedure rarely emits)."""
        self.quick_jv_status.setText(status)

    def _on_quick_jv_finished(self, proc):
        """Extract metrics, populate results grid, auto-fill Vmpp."""
        self._quick_jv_running = False
        self.main_window.controller.is_busy = False
        self.quick_jv_progress.setValue(100)
        self._reset_quick_jv_controls()
        self._disconnect_quick_jv_instruments()

        try:
            channel = self.get_selected_channel()
            channel_results = proc.analysis_results.get(channel, {})
            metrics = channel_results.get("Forward")
            if not metrics:
                self.quick_jv_status.setText("No analysis result produced.")
                QtWidgets.QMessageBox.warning(
                    self, "Quick JV", "No analysis result was produced."
                )
                return

            # Populate results grid (keys stored in native units: mV, mA/cm²)
            voc_v = metrics.get("Voc", 0.0) / 1000.0
            jsc_ma = metrics.get("Jsc", 0.0)
            ff_pct = metrics.get("FF", 0.0)
            vmpp_v = metrics.get("Vmpp", 0.0) / 1000.0
            eff_pct = metrics.get("EFF", 0.0)

            self.quick_jv_result_labels["Voc (V)"].setText(f"{voc_v:.3f}")
            self.quick_jv_result_labels["Jsc (mA/cm²)"].setText(f"{jsc_ma:.2f}")
            self.quick_jv_result_labels["FF (%)"].setText(f"{ff_pct:.2f}")
            self.quick_jv_result_labels["Vmpp (V)"].setText(f"{vmpp_v:.4f}")
            self.quick_jv_result_labels["Eff (%)"].setText(f"{eff_pct:.2f}")

            # Highlight the Vmpp value in green
            self.quick_jv_result_labels["Vmpp (V)"].setStyleSheet(
                "font-size: 12px; font-weight: 600; color: #053a46; background: transparent;"
            )

            # Place MPP marker on the plot
            if self._quick_jv_voltages and self._quick_jv_currents:
                # Find the measured point closest to Vmpp
                diffs = [
                    abs(v - vmpp_v) for v in self._quick_jv_voltages
                ]
                idx = diffs.index(min(diffs))
                self.quick_jv_mpp.setData(
                    [self._quick_jv_voltages[idx]],
                    [self._quick_jv_currents[idx]],
                )

            # Auto-fill the Vmpp field — set source BEFORE setText so
            # _on_vmpp_changed keeps the green "Quick JV" badge.
            self._vmpp_source = "quick_jv"
            self.vmpp_input.setText(f"{vmpp_v:.4f}")

            self.quick_jv_status.setText(f"Vmpp found: {vmpp_v:.4f} V")

        except Exception as e:
            logger.error(f"Quick JV post-processing failed: {e}")
            self.quick_jv_status.setText("Post-processing failed.")
            QtWidgets.QMessageBox.warning(
                self, "Quick JV", f"Failed to read Vmpp: {e}"
            )

    def _on_quick_jv_failed(self, message: str):
        """Handle Quick JV worker failure."""
        self._quick_jv_running = False
        self.main_window.controller.is_busy = False
        self._reset_quick_jv_controls()
        self._disconnect_quick_jv_instruments()
        self.quick_jv_status.setText("Failed")
        QtWidgets.QMessageBox.warning(self, "Quick JV Failed", message)

    def _reset_quick_jv_controls(self):
        """Re-enable all Quick JV controls.
        Guarded: if an SPO run is active, leave controls locked (SPO's
        set_config_enabled(False) wins)."""
        spo_running = getattr(
            self.main_window.controller, 'spo_running', False
        )
        if spo_running:
            return
        self.quick_jv_button.setEnabled(True)
        self.quick_jv_button.setText("Run Quick JV")
        self.manual_mode_button.setEnabled(True)
        self.quick_jv_mode_button.setEnabled(True)
        self.quick_jv_start_input.setEnabled(True)
        self.quick_jv_stop_input.setEnabled(True)
        self.quick_jv_rate_input.setEnabled(True)
        self.vmpp_input.setEnabled(True)

    def _disconnect_quick_jv_instruments(self):
        """Disconnect hardware after a Quick JV sweep."""
        try:
            self.main_window.instrument_manager.disconnect_keithley()
            self.main_window.instrument_manager.disconnect_mux()
        except Exception:
            pass
        finally:
            self.main_window.update_instrument_lights()


class SpoWidget(QtWidgets.QWidget):
    """
    Live plot and live metrics for an SPO stability test. Replaces the JV
    plot/browser/analysis display area while SPO mode is active. The
    configuration fields themselves live in `SpoParameterTab`, which is
    swapped into the sidebar's "Parameters" tab instead of being duplicated
    here.

    Public methods (called by the main window / app controller):
        set_mode_spo(), start_spo(), abort_spo(), update_plot(),
        update_metrics(), on_spo_finished()
    """

    # Re-emitted from param_tab so existing connections (main window) don't
    # need to reach into the parameter tab directly.
    vmpp_ready = QtCore.pyqtSignal(bool)

    def __init__(self, main_window, param_tab, parent=None):
        """
        Args:
            main_window: The JVAnalyzerWindow instance.
            param_tab: The SpoParameterTab instance living in the sidebar's
                "Parameters" tab, used to read config and lock it while running.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.param_tab = param_tab
        self.param_tab.vmpp_ready.connect(self.vmpp_ready.emit)

        self._running = False
        self._times = []
        self._powers = []
        self._report_path = None

        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        layout.addWidget(self._build_plot_card(), stretch=1)
        layout.addWidget(self._build_metrics_card())

    def _build_plot_card(self):
        container = QtWidgets.QGroupBox("Live Power vs Time")
        layout = QtWidgets.QVBoxLayout(container)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Power', units='W')
        self.curve = self.plot_widget.plot([], [], pen=pg.mkPen(color='#053a46', width=2))

        layout.addWidget(self.plot_widget)

        # Open a saved SPO report without leaving this view. The J-V side has
        # its own Open in the experiment browser, which is not reachable from
        # here — so SPO needs its own way in.
        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.open_report_button = QtWidgets.QPushButton("Open SPO Report…")
        self.open_report_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_report_button.setToolTip(
            "Load a previously saved SPO report into this view.")
        self.open_report_button.clicked.connect(self._on_open_report_clicked)
        button_row.addWidget(self.open_report_button)
        layout.addLayout(button_row)
        return container

    def _on_open_report_clicked(self):
        """Pick a saved SPO report and show it here.

        Starts in the SPO folder on the store — where finished reports
        actually live — rather than in the staging directory they pass
        through on the way.
        """
        start_dir = os.path.expanduser("~")
        manager = getattr(self.main_window, "dir_manager", None)
        if manager is not None:
            start_dir = manager.dialog_start_dir("SPO")

        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open SPO Report", start_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        if not self.load_report(path):
            QtWidgets.QMessageBox.warning(
                self, "Not an SPO Report",
                f"{os.path.basename(path)} does not contain an SPO time "
                "series.\n\nOpen J-V reports from the experiment browser "
                "instead."
            )

    def _build_metrics_card(self):
        group = QtWidgets.QGroupBox("Live Metrics")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        self.mean_power_label = QtWidgets.QLabel("--")
        self.drift_label = QtWidgets.QLabel("--")
        self.elapsed_label = QtWidgets.QLabel("--")
        self.status_label = QtWidgets.QLabel("Idle")

        grid.addWidget(QtWidgets.QLabel("Mean Power"), 0, 0)
        grid.addWidget(self.mean_power_label, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Drift %"), 0, 2)
        grid.addWidget(self.drift_label, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Elapsed Time"), 1, 0)
        grid.addWidget(self.elapsed_label, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Status"), 1, 2)
        grid.addWidget(self.status_label, 1, 3)

        # No "Save Report" button. SpoReport.finalize() writes the report to
        # the SPO folder before this card is ever updated, and the store
        # publishes it from there — a button that only copied the finished
        # file somewhere else implied the report was NOT saved unless pressed,
        # which is the opposite of the truth. It was also the one place a
        # measurement file was written with a plain file copy, which bypasses
        # the publisher's exclusive-create guarantee if aimed at the store.
        return group

    # -------------------------------------------------------------------
    # Parameters (delegated to the SpoParameterTab in the sidebar)
    # -------------------------------------------------------------------

    def get_parameters(self) -> dict:
        """Collect SPO-specific parameters from the sidebar's config fields."""
        return self.param_tab.get_parameters()

    def has_valid_hold_voltage(self) -> bool:
        """True if the Vmpp field currently holds a parsable float."""
        return self.param_tab.has_valid_hold_voltage()

    # -------------------------------------------------------------------
    # Public API used by the main window / app controller
    # -------------------------------------------------------------------

    def set_mode_spo(self):
        """Called when the window switches into SPO mode."""
        if not self._running:
            self.status_label.setText("Idle")

    def set_filename(self, text: str) -> None:
        """Sync the shared filename field and trigger validation so the SPO
        start button reflects the current filename state immediately."""
        self.main_window.file_panel.filename_input.setText(text)
        # Force the window-side validation slot to re-evaluate button states
        if hasattr(self.main_window, '_on_filename_changed'):
            self.main_window._on_filename_changed(text)

    def start_spo(self):
        """Reset live views to a running state. Called when an SPO run starts."""
        self._running = True
        self._times = []
        self._powers = []
        self._report_path = None
        self.curve.setData([], [])
        self.mean_power_label.setText("--")
        self.drift_label.setText("--")
        self.elapsed_label.setText("0 s")
        self.status_label.setText("Running")
        self.param_tab.set_config_enabled(False)

    def abort_spo(self):
        """Called immediately after an abort has been requested."""
        self.status_label.setText("Aborting...")

    def update_plot(self, elapsed_s: float, power_w: float):
        """Append one point to the live Power vs Time curve."""
        self._times.append(elapsed_s)
        self._powers.append(power_w)
        self.curve.setData(self._times, self._powers)
        self.elapsed_label.setText(f"{elapsed_s:.1f} s")

    def update_metrics(self, metrics: dict):
        """Refresh the live metrics labels from a partial or final metrics dict."""
        self.mean_power_label.setText(f"{metrics.get('mean_power_mw', 0.0):.3f} mW")
        self.drift_label.setText(f"{metrics.get('drift_percent', 0.0):.2f} %")

    # -------------------------------------------------------------------
    # Loading a finished report back in
    # -------------------------------------------------------------------
    @staticmethod
    def parse_report(path: str) -> tuple:
        """Read a finalised SPO report into (times_s, powers_w, metrics).

        Reads the `[[ TIME SERIES DATA ]]` and `[[ SPO METRICS ]]` sections
        written by `SpoReport.finalize()`. The report stores power in mW
        because that is what people read; the live plot works in W, so the
        series is converted here and the widget stays unit-consistent whether
        a curve came from a live run or from disk.

        Raises:
            ValueError: if the file has no time-series section, i.e. it is
                not an SPO report.
        """
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        label = "[[ TIME SERIES DATA ]]"
        if label not in lines:
            raise ValueError("no [[ TIME SERIES DATA ]] section")
        # Last occurrence: operator notes are written verbatim above it and
        # could contain the marker text. Data rows are numeric and cannot.
        start = len(lines) - 1 - lines[::-1].index(label)

        times, powers = [], []
        for row in lines[start + 2:]:          # +1 marker, +1 column header
            if not row.strip():
                continue
            parts = row.split(",")
            if len(parts) < 4:
                continue
            try:
                times.append(float(parts[0]))
                # mW -> W, and negated to match the LIVE plot. A live run
                # is fed `-record["Power (W)"]` by the controller, so the
                # curve is drawn positive; the report stores the raw
                # (negative) instrument value. Without this a loaded curve
                # is mirrored about zero and sits off the bottom of an axis
                # scaled for positive power.
                powers.append(-float(parts[3]) / 1000.0)
            except ValueError:
                continue                        # a stray non-numeric row

        metrics = _parse_metrics_block(lines)

        if not times:
            raise ValueError("no time-series rows")
        return times, powers, metrics

    def load_report(self, path: str) -> bool:
        """Show a saved SPO report: its curve, its metrics and its filename.

        Returns True when the file was an SPO report and has been displayed.
        Never raises — a malformed file is reported to the caller as False so
        it can fall back to the J-V loader.
        """
        try:
            times, powers, metrics = self.parse_report(path)
        except (OSError, ValueError) as exc:
            logger.debug(f"Not a loadable SPO report ({path}): {exc}")
            return False

        self._times = list(times)
        self._powers = list(powers)
        self.curve.setData(self._times, self._powers)
        # autoRange() rescales now; enableAutoRange() only arms it for the
        # next update, which never comes for a static loaded curve.
        self.plot_widget.autoRange()

        self.update_metrics(metrics)
        self.elapsed_label.setText(f"{times[-1]:.1f} s")
        self.status_label.setText(f"Loaded — {os.path.basename(path)}")
        self._report_path = path
        logger.info(f"Loaded SPO report: {path} ({len(times)} points)")
        return True

    def on_spo_finished(self, metrics: dict, report_path: str = None):
        """Called once the run (completed or aborted) has fully stopped and
        a formatted report has been generated (report_path may be None if
        no data was collected, e.g. aborted during pre-conditioning)."""
        self._running = False
        self._report_path = report_path
        self.update_metrics(metrics)
        self.status_label.setText("Complete" if report_path else "Aborted (no data)")
        self.param_tab.set_config_enabled(True)

