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
from solarjv_analyzer.gui.widgets.vmpp_card import VmppCard
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
        # The sweep state used to live here. It belongs to the card now, and
        # the read-only properties below forward to it -- so these must NOT
        # be re-assigned here or they shadow the property with an attribute.
        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        # Cards added to this form (Hold Voltage, Channel Selection) must line
        # up with File Output, which is not nested in here. Qt's default 9px
        # form inset is what pushed them in and made them narrower.
        form.setContentsMargins(0, 9, 0, 9)
        form.setVerticalSpacing(8)

        self.hold_duration_input = QtWidgets.QLineEdit("300")
        form.addRow("Hold Duration (s):", self.hold_duration_input)

        self.sampling_interval_input = QtWidgets.QLineEdit("1.0")
        form.addRow("Sampling Interval (s):", self.sampling_interval_input)

        self.preconditioning_input = QtWidgets.QLineEdit("5.0")
        form.addRow("Pre-conditioning (s):", self.preconditioning_input)

        # Order is Parameters, Hold Voltage, Channel Selection -- the same in
        # every tab that has these two cards, so the panel reads the same way
        # whichever mode you are in.
        # ---- Hold Voltage section (the shared card) ----------------------
        # Built before the channel selector is added to the form but AFTER it
        # exists, because Quick JV asks the selector which channel to sweep.
        self.vmpp_card = VmppCard(
            self.main_window,
            title="Hold Voltage",
            placeholder="Enter hold voltage (V)",
            channel_provider=self.get_selected_channel,
        )
        self.vmpp_card.voltage_ready.connect(self.vmpp_ready.emit)
        form.addRow(self.vmpp_card)

        self._create_channel_selector(form)

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
            'hold_voltage': self.vmpp_card.voltage(0.0),
            'hold_duration': float(self.hold_duration_input.text() or 300.0),
            'sampling_interval': float(self.sampling_interval_input.text() or 1.0),
            'preconditioning_time': float(self.preconditioning_input.text() or 5.0),
            'active_channel': self.get_selected_channel(),
        }

    def has_valid_hold_voltage(self) -> bool:
        """True if the hold-voltage field currently holds a parsable float."""
        return self.vmpp_card.has_valid_voltage()

    def set_hold_voltage(self, value: float, source: str = "manual") -> None:
        """Used by the combined J-V + SPO run to seed Vmpp from the sweep."""
        self.vmpp_card.set_voltage(value, source)

    def set_config_enabled(self, enabled: bool):
        """Enable/disable all configuration inputs (locked while running)."""
        for widget in (
            self.hold_duration_input, self.sampling_interval_input,
            self.preconditioning_input,
            *self.channels,
        ):
            widget.setEnabled(enabled)
        # The card locks its own inputs, including the ones SPO never had.
        self.vmpp_card.set_config_enabled(enabled)

    # -------------------------------------------------------------------
    # Hold Voltage Card (Manual / Quick JV)
    # -------------------------------------------------------------------

    # ------------------------------------------------------------------
    # The hold-voltage card now lives in gui/widgets/vmpp_card.py
    #
    # 519 lines of Manual / Quick JV logic used to be duplicated here and in
    # the MPP tab, so every fix had to be made twice and the two drifted
    # apart anyway -- SPO never got the unit selectors, the editable device
    # fields, the working architecture toggle or the busy-flag fix.
    #
    # These forwarders keep SpoParameterTab's public surface identical, so
    # nothing that calls this tab had to change.
    # ------------------------------------------------------------------

    @property
    def vmpp_input(self):
        return self.vmpp_card.vmpp_input

    @property
    def manual_mode_button(self):
        return self.vmpp_card.manual_mode_button

    @property
    def quick_jv_mode_button(self):
        return self.vmpp_card.quick_jv_mode_button

    @property
    def quick_jv_button(self):
        return self.vmpp_card.quick_jv_button

    @property
    def quick_jv_start_input(self):
        return self.vmpp_card.quick_jv_start_input

    @property
    def quick_jv_stop_input(self):
        return self.vmpp_card.quick_jv_stop_input

    @property
    def quick_jv_rate_input(self):
        return self.vmpp_card.quick_jv_rate_input

    @property
    def _quick_jv_worker(self):
        """jv_analyzer_window reaches for this by name during shutdown."""
        return getattr(self.vmpp_card, "_quick_jv_worker", None)

    @property
    def _quick_jv_running(self):
        return getattr(self.vmpp_card, "_quick_jv_running", False)

    def device_settings(self) -> dict:
        """Area, compliance and architecture for an SPO run -- the card's own
        fields in Quick JV mode, the J-V Parameters tab in Manual."""
        return self.vmpp_card.device_settings()

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

