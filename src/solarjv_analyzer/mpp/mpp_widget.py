"""The MPP sidebar fields and the live tracking display.

Two pieces, matching the SPO split: `MppParameterTab` goes into the sidebar's
"Parameters" slot, `MppWidget` replaces the J-V plot area.

The display differs from SPO's on purpose. SPO holds one voltage, so power
against time says everything. MPP moves the voltage, and watching it hunt is
how an operator tells a working tracker from a stuck one — so the tracked
voltage is drawn on a second axis beside the power.
"""

import logging
import os

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer.gui.widgets.single_channel_selector import SingleChannelSelector
from solarjv_analyzer.gui.widgets.unit_field import UnitField
from solarjv_analyzer.gui.widgets.vmpp_card import VmppCard

logger = logging.getLogger(__name__)

MAX_RUN_TIME_MIN = 180.0        # 3 hours, agreed with the professor

POWER_COLOR = "#053a46"
VOLTAGE_COLOR = "#c2410c"


def _float_or_none(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


class MppParameterTab(QtWidgets.QWidget):
    """MPP configuration, in the sidebar's Parameters tab."""

    # Mirrors SpoParameterTab.vmpp_ready: the window uses it to enable Start.
    start_voltage_ready = QtCore.pyqtSignal(bool)

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        form = QtWidgets.QFormLayout(self)
        # Cards added to this form (Hold Voltage, Channel Selection) must line
        # up with File Output, which is not nested in here. Qt's default 9px
        # form inset is what pushed them in and made them narrower.
        form.setContentsMargins(0, 9, 0, 9)
        form.setVerticalSpacing(8)

        # Every dimensioned field carries its own unit selector, as on the
        # J-V tab. The unit is no longer baked into the label, so an operator
        # who thinks in volts or in minutes is not doing arithmetic in their
        # head before typing.
        self.run_time_input = UnitField("time_min", 5.0, "min")
        self.run_time_input.setToolTip(
            f"Maximum {MAX_RUN_TIME_MIN:.0f} minutes (3 hours).")
        form.addRow("Run Time:", self.run_time_input)

        self.probe_step_input = UnitField("voltage", 0.010, "mV")
        self.probe_step_input.setToolTip(
            "How far the tracker steps when it probes for a better voltage.")
        form.addRow("Probe Step:", self.probe_step_input)

        self.probe_interval_input = UnitField("time_s", 60.0, "s")
        self.probe_interval_input.setToolTip(
            "How long it holds still between probes. One cycle is this plus "
            "Probe Points x Data Interval, and the tracker moves one Probe "
            "Step per cycle -- so this sets how fast it can travel.")
        form.addRow("Probe Interval:", self.probe_interval_input)

        self.data_interval_input = UnitField("time_s", 10.0, "s")
        form.addRow("Data Interval:", self.data_interval_input)

        self.probe_points_input = QtWidgets.QLineEdit("3")
        form.addRow("Probe Points:", self.probe_points_input)

        self.averages_input = QtWidgets.QLineEdit("10")
        form.addRow("Averages per Point:", self.averages_input)

        self.max_excursion_input = UnitField("voltage", 0.300, "mV")
        self.max_excursion_input.setToolTip(
            "The tracked voltage may not leave this window around the start. "
            "Hitting it is counted and reported, not silently obeyed.")
        form.addRow("Max Excursion:", self.max_excursion_input)

        self._create_threshold_row(form)

        self.preconditioning_input = UnitField("time_s", 5.0, "s")
        self.preconditioning_input.setToolTip(
            "Held at the start voltage before any data is recorded.")
        form.addRow("Pre-conditioning:", self.preconditioning_input)

        # Built first because the Quick JV sweep needs to know which channel
        # to measure and asks the selector -- but added to the form LAST, so
        # that Hold Voltage sits above Channel Selection. Order is Parameters,
        # Hold Voltage, Channel Selection, matching SPO.
        self.channel_selector = SingleChannelSelector()

        # Typing the start voltage by hand is not good enough here. One probe
        # step is 10 mV and a cycle takes about 90 s, so a 5-minute run
        # travels roughly 30 mV in total — a start voltage 100 mV out is a run
        # that never reaches the maximum and a report that still looks
        # plausible.
        self.vmpp_card = VmppCard(
            self.main_window,
            title="Start Voltage",
            placeholder="Enter start voltage (V)",
            channel_provider=self.get_selected_channel,
        )
        self.vmpp_card.voltage_ready.connect(self.start_voltage_ready.emit)
        form.addRow(self.vmpp_card)
        form.addRow(self.channel_selector)

    def _create_threshold_row(self, form):
        """The noise gate, off by default.

        A 2% gate never opens anywhere the tracker actually operates — a 10 mV
        probe near a maximum moves the power by well under 1%, because P(V) is
        flat there — so switching it on at that value freezes the tracker at
        its start voltage. It is here so real runs can tell us what scale
        works, not because 2% is a sensible default.
        """
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.threshold_check = QtWidgets.QCheckBox()
        self.threshold_check.setChecked(False)
        self.threshold_check.setToolTip(
            "Off by default. A probe must beat the baseline by this much to "
            "be kept; set too high, the tracker never moves.")
        layout.addWidget(self.threshold_check)

        self.threshold_input = QtWidgets.QLineEdit("2.0")
        self.threshold_input.setEnabled(False)
        layout.addWidget(self.threshold_input)
        layout.addStretch(1)

        self.threshold_check.toggled.connect(self.threshold_input.setEnabled)
        form.addRow("Noise Threshold (%):", row)

    # -- validation -------------------------------------------------------
    def has_valid_start_voltage(self) -> bool:
        """True when the card holds a parsable start voltage."""
        return self.vmpp_card.has_valid_voltage()

    def set_start_voltage(self, voltage: float, source: str = "manual"):
        """Seed the card — from a J-V run's signed Vmpp, or by hand."""
        self.vmpp_card.set_voltage(voltage, source)

    # -- public API -------------------------------------------------------
    def get_selected_channel(self) -> int:
        return self.channel_selector.selected_channel()

    def get_parameters(self) -> dict:
        """Collect the MPP fields, falling back to the documented defaults.

        Run time is capped here as well as in the procedure: the operator
        should see the value that will actually be used.
        """
        # UnitField.value() already returns the base unit (minutes, seconds,
        # volts), so the only conversions left are the two the procedure
        # declares in millivolts.
        run_time = min(abs(self.run_time_input.value(5.0)), MAX_RUN_TIME_MIN)

        def number(field, default):
            value = _float_or_none(field.text())
            return default if value is None else value

        return {
            "run_time": run_time,
            "start_voltage": self.vmpp_card.voltage(0.0),
            "probe_step": abs(self.probe_step_input.value(0.010)) * 1000.0,
            "probe_interval": abs(self.probe_interval_input.value(60.0)),
            "data_interval": abs(self.data_interval_input.value(10.0)),
            "probe_points": max(1, int(number(self.probe_points_input, 3))),
            "averages_per_point": max(1, int(number(self.averages_input, 10))),
            "max_excursion": abs(self.max_excursion_input.value(0.300)) * 1000.0,
            "use_noise_threshold": bool(self.threshold_check.isChecked()),
            "noise_threshold_percent": number(self.threshold_input, 2.0),
            "preconditioning_time": abs(self.preconditioning_input.value(5.0)),
            "active_channel": self.get_selected_channel(),
        }

    def set_config_enabled(self, enabled: bool):
        """Lock every input while a run is in progress."""
        for widget in (
            self.run_time_input, self.probe_step_input,
            self.probe_interval_input, self.data_interval_input,
            self.probe_points_input, self.averages_input,
            self.max_excursion_input, self.threshold_check,
            self.preconditioning_input,
        ):
            widget.setEnabled(enabled)
        self.vmpp_card.set_config_enabled(enabled)
        # Only meaningful when the gate is on, whatever the run state.
        self.threshold_input.setEnabled(enabled and self.threshold_check.isChecked())
        self.channel_selector.set_config_enabled(enabled)


class MppWidget(QtWidgets.QWidget):
    """Live power and tracked voltage for an MPP run."""

    def __init__(self, main_window, param_tab, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.param_tab = param_tab

        self._running = False
        self._times = []
        self._powers = []
        self._voltages = []
        self._report_path = None

        self._build_ui()

    def _build_ui(self):
        """Same shape as SpoWidget: a plot card that takes the slack, and a
        metrics card at its natural height below it. Keeping the two views
        laid out identically means an operator who learns one can read the
        other without re-orienting."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        layout.addWidget(self._build_plot_card(), stretch=1)
        layout.addWidget(self._build_metrics_card())

    def _build_plot_card(self):
        container = QtWidgets.QGroupBox("Live Power and Tracked Voltage")
        layout = QtWidgets.QVBoxLayout(container)

        # One plot, two y axes: power on the left, the tracked voltage on the
        # right. The voltage trace is what distinguishes a working tracker
        # from a stuck one, and it belongs against the same time axis.
        #
        # A second y axis means adding a ViewBox to the plot's scene by hand,
        # and pyqtgraph keeps every ViewBox in a module-level registry that is
        # weak on the Python wrapper but NOT on the C++ object. One left
        # behind by a closed window is walked, and dereferenced, the next time
        # any plot is built. `release()` hands it back, and
        # `test_pyqtgraphs_view_registry_does_not_grow_per_window` proves it.
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        plot_item = self.plot_widget.plotItem
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.setLabel("bottom", "Time", units="s")
        plot_item.setLabel("left", "Power", units="mW")
        plot_item.getAxis("left").setPen(pg.mkPen(POWER_COLOR))
        self.power_curve = plot_item.plot(
            [], [], pen=pg.mkPen(POWER_COLOR, width=2))

        self._voltage_view = pg.ViewBox()
        plot_item.showAxis("right")
        plot_item.scene().addItem(self._voltage_view)
        right_axis = plot_item.getAxis("right")
        right_axis.linkToView(self._voltage_view)
        right_axis.setLabel("Tracked", units="V", color=VOLTAGE_COLOR)
        right_axis.setPen(pg.mkPen(VOLTAGE_COLOR))
        self._voltage_view.setXLink(plot_item)

        self.voltage_curve = pg.PlotCurveItem(
            pen=pg.mkPen(VOLTAGE_COLOR, width=2, style=QtCore.Qt.DashLine))
        self._voltage_view.addItem(self.voltage_curve)

        plot_item.vb.sigResized.connect(self._sync_voltage_view)
        self._sync_voltage_view()

        layout.addWidget(self.plot_widget)

        # As on the SPO view: the J-V side opens reports from the experiment
        # browser, which is not reachable from here, so MPP needs its own.
        button_row = QtWidgets.QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        self.open_report_button = QtWidgets.QPushButton("Open MPP Report…")
        self.open_report_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_report_button.setToolTip(
            "Load a previously saved MPP report into this view.")
        self.open_report_button.clicked.connect(self._on_open_report_clicked)
        button_row.addWidget(self.open_report_button)
        layout.addLayout(button_row)
        return container

    def _sync_voltage_view(self):
        """Keep the second axis' box lined up with the first.

        pyqtgraph does not lay out an added ViewBox for you; without this the
        voltage trace is drawn at the wrong geometry.
        """
        view = getattr(self, "_voltage_view", None)
        if view is None:
            return
        try:
            plot_item = self.plot_widget.plotItem
            view.setGeometry(plot_item.vb.sceneBoundingRect())
            view.linkedViewChanged(plot_item.vb, view.XAxis)
        except (RuntimeError, AttributeError):
            pass                    # torn down mid-resize

    def _build_metrics_card(self):
        """Label/value grid, same treatment as SpoWidget's Live Metrics.

        Six readouts rather than SPO's four: MPP adds the tracked voltage,
        which moves, and the channel, because a tracking run is always on one.
        """
        group = QtWidgets.QGroupBox("Live Metrics")
        grid = QtWidgets.QGridLayout(group)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        self.mean_power_label = QtWidgets.QLabel("--")
        self.drift_label = QtWidgets.QLabel("--")
        self.tracked_label = QtWidgets.QLabel("--")
        self.elapsed_label = QtWidgets.QLabel("--")
        self.channel_label = QtWidgets.QLabel("--")
        self.status_label = QtWidgets.QLabel("Idle")

        grid.addWidget(QtWidgets.QLabel("Mean Power"), 0, 0)
        grid.addWidget(self.mean_power_label, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Drift %"), 0, 2)
        grid.addWidget(self.drift_label, 0, 3)

        grid.addWidget(QtWidgets.QLabel("Tracked Voltage"), 1, 0)
        grid.addWidget(self.tracked_label, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Elapsed Time"), 1, 2)
        grid.addWidget(self.elapsed_label, 1, 3)

        grid.addWidget(QtWidgets.QLabel("Channel"), 2, 0)
        grid.addWidget(self.channel_label, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Status"), 2, 2)
        grid.addWidget(self.status_label, 2, 3)

        return group

    # -- opening a saved report -------------------------------------------
    def _on_open_report_clicked(self):
        """Pick a saved MPP report and show it here.

        Starts in the MPP folder on the store, where finished reports live,
        rather than the staging directory they pass through on the way.
        """
        start_dir = os.path.expanduser("~")
        manager = getattr(self.main_window, "dir_manager", None)
        if manager is not None:
            try:
                start_dir = manager.dialog_start_dir("MPP")
            except Exception:                     # noqa: BLE001
                pass

        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open MPP Report", start_dir,
            "CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        if not self.load_report(path):
            QtWidgets.QMessageBox.warning(
                self, "Not an MPP Report",
                f"{os.path.basename(path)} does not contain an MPP time "
                "series.\n\nOpen J-V reports from the experiment browser "
                "instead.")

    @staticmethod
    def parse_report(path: str):
        """Pull the time series and the metrics out of a saved MPP report.

        Raises ValueError when the file is not one, so `load_report` can tell
        the caller to try a different loader rather than showing nothing.
        """
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()

        marker = "[[ TIME SERIES DATA ]]"
        if marker not in lines:
            raise ValueError("no time series block")
        # The LAST occurrence: a note could legitimately contain the marker.
        start = len(lines) - 1 - lines[::-1].index(marker)

        times, powers, voltages = [], [], []
        for line in lines[start + 2:]:            # +2 skips the column header
            if not line.strip():
                break
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                times.append(float(parts[0]))
                voltages.append(float(parts[1]))
                # The report stores the raw signed product in mW; negate it
                # to the generated power the live view plots.
                powers.append(-float(parts[3]))
            except ValueError:
                continue
        if not times:
            raise ValueError("time series block held no rows")

        metrics = {}
        summary = "[[ ANALYSIS SUMMARY ]]"
        if summary in lines:
            head = lines.index(summary)
            try:
                labels = lines[head + 1].split(",")[1:]   # drop "Channel"
                values = lines[head + 2].split(",")[1:]
                for label, value in zip(labels, values):
                    name = label.split(" (")[0].strip()
                    try:
                        metrics[name] = float(value)
                    except ValueError:
                        metrics[name] = value.strip()
            except IndexError:
                pass
        return times, powers, voltages, metrics

    def load_report(self, path: str) -> bool:
        """Show a saved MPP report. Never raises; False means "not mine"."""
        try:
            times, powers, voltages, metrics = self.parse_report(path)
        except (OSError, ValueError) as exc:
            logger.debug(f"Not a loadable MPP report ({path}): {exc}")
            return False

        self._times, self._powers, self._voltages = (
            list(times), list(powers), list(voltages))
        self.power_curve.setData(self._times, self._powers)
        if self.voltage_curve is not None:
            self.voltage_curve.setData(self._times, self._voltages)
        # autoRange() rescales now; enableAutoRange() only arms it for a next
        # update, which never comes for a static loaded curve.
        self.plot_widget.autoRange()
        self._sync_voltage_view()

        self.on_mpp_finished(metrics, report_path=path)
        self.elapsed_label.setText(f"{times[-1]:.1f} s")
        self.status_label.setText(f"Loaded — {os.path.basename(path)}")
        logger.info(f"Loaded MPP report: {path} ({len(times)} points)")
        return True

    # -- public API, mirroring SpoWidget ----------------------------------
    def set_mode_mpp(self):
        """Called when the window switches into MPP mode."""
        if not self._running:
            self.status_label.setText("Idle")

    def set_filename(self, text: str) -> None:
        self.main_window.file_panel.filename_input.setText(text)
        if hasattr(self.main_window, "_on_filename_changed"):
            self.main_window._on_filename_changed(text)

    def start_mpp(self):
        """Reset the live views for a new run."""
        self._running = True
        self._times, self._powers, self._voltages = [], [], []
        self._report_path = None
        self.power_curve.setData([], [])
        self.voltage_curve.setData([], [])
        self.channel_label.setText(str(self.param_tab.get_selected_channel()))
        self.tracked_label.setText("--")
        self.mean_power_label.setText("--")
        self.drift_label.setText("--")
        self.elapsed_label.setText("0 s")
        self.status_label.setText("Tracking")
        self.param_tab.set_config_enabled(False)

    def abort_mpp(self):
        self.status_label.setText("Aborting...")

    def update_plot(self, elapsed_s: float, power_w: float, voltage_v: float):
        """Append one sample to both traces."""
        self._times.append(elapsed_s)
        # GENERATED power, -(V x I) in mW: positive while the cell generates,
        # in quadrant 2 and quadrant 4 alike. The same transform SPO's live
        # view uses, and the same one `load_report` applies, so a reopened
        # report draws the curve the operator watched.
        #
        # Not abs(): a cell that is consuming rather than generating plots
        # below zero, which is exactly what you want to see. abs() would draw
        # it as a healthy positive trace.
        self._powers.append(-power_w * 1000.0)
        self._voltages.append(voltage_v)                # signed, like Vmpp
        self.power_curve.setData(self._times, self._powers)
        self.voltage_curve.setData(self._times, self._voltages)
        self.tracked_label.setText(f"{voltage_v:+.3f} V")
        self.elapsed_label.setText(f"{elapsed_s:.0f} s")
        if self._powers:
            mean = sum(self._powers) / len(self._powers)
            self.mean_power_label.setText(f"{mean:.3f} mW")
            first = self._powers[0]
            if abs(first) > 1e-12:
                drift = (self._powers[-1] - first) / abs(first) * 100.0
                self.drift_label.setText(f"{drift:+.2f} %")

    def on_mpp_finished(self, metrics: dict, report_path: str = None):
        """Settle the display once the run ends."""
        self._running = False
        self._report_path = report_path
        self.status_label.setText("Complete")
        self.param_tab.set_config_enabled(True)
        if metrics:
            self.mean_power_label.setText(
                f"{metrics.get('mean_power_mw', 0.0):.3f} mW")
            self.drift_label.setText(
                f"{metrics.get('drift_percent', 0.0):+.2f} %")
            self.tracked_label.setText(
                f"{metrics.get('final_voltage_v', 0.0):+.3f} V")

    @property
    def report_path(self):
        return self._report_path

    def release(self):
        """Hand the second ViewBox back to pyqtgraph.

        Called from the window's closeEvent: a child widget's own closeEvent
        does NOT fire when its parent window closes, so this cannot be left to
        `closeEvent`. `ViewBox.close()` is what unregisters it; the resize
        connection goes first because it holds a bound method of this widget
        and would otherwise keep the whole plot alive.
        """
        view, self._voltage_view = getattr(self, "_voltage_view", None), None
        if view is None:
            return
        try:
            self.plot_widget.plotItem.vb.sigResized.disconnect(
                self._sync_voltage_view)
        except (TypeError, RuntimeError):
            pass                      # never connected, or already gone
        try:
            view.setXLink(None)
            self.plot_widget.plotItem.getAxis("right").linkToView(None)
        except (RuntimeError, AttributeError, TypeError):
            pass
        try:
            scene = self.plot_widget.plotItem.scene()
            if scene is not None:
                scene.removeItem(view)
        except (RuntimeError, AttributeError):
            pass
        try:
            view.close()              # clear() + unregister()
        except (RuntimeError, AttributeError):
            pass
        self.voltage_curve = None

    def closeEvent(self, event):
        self.release()
        super().closeEvent(event)
