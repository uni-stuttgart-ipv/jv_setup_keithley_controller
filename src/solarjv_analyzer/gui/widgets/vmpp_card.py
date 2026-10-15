"""The voltage entry card: type it, or find it with a Quick JV sweep.

SPO needs a hold voltage and MPP needs a start voltage, and the problem is
identical — the operator must not have to guess. A 10 mV probe step means a
5-minute MPP run travels only about 30 mV, so a start voltage 100 mV out is a
run that never reaches the maximum and a report that still looks plausible.

The Quick JV sweep and its handlers below are MOVED, unchanged, from
`SpoParameterTab`, so the behaviour is the one already proven on the lab
instrument rather than a re-derivation of it. `SpoParameterTab` still carries
its own copy today; migrating it to this card is a contained follow-up, worth
doing on its own so the SPO tab can be verified by itself. Until then the two
exist side by side, and a fix to the sweep belongs in both.
"""

import logging

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from solarjv_analyzer.gui.theme import tokens
from solarjv_analyzer.gui.widgets.toggle_switch import ToggleSwitch
from solarjv_analyzer.gui.widgets.unit_field import UnitField
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.spo.spo_procedure import SpoWorker

logger = logging.getLogger(__name__)


class TightStackedWidget(QtWidgets.QStackedWidget):
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


class VmppCard(QtWidgets.QGroupBox):
    """A signed voltage, entered by hand or found by a quick sweep."""

    # Emitted whenever the entry becomes valid or invalid, so a window can
    # enable or disable its Start button.
    voltage_ready = QtCore.pyqtSignal(bool)

    # Fixed, fast single-sweep parameters used only to locate Vmpp.
    # The J-V Sweep tab's defaults, deliberately identical: this is the same
    # measurement, so it should start from the same numbers and behave the
    # same way. ParameterTab holds the authoritative copy.
    QUICK_JV_START = 1.2        # V
    QUICK_JV_STOP = -0.2        # V
    QUICK_JV_STEP = -0.010      # V  (sign is ignored by JVProcedure)
    QUICK_JV_RATE = 0.1         # V/s  == 100 mV/s

    def __init__(self, main_window, title: str = "Hold Voltage",
                 placeholder: str = "Enter hold voltage (V)",
                 channel_provider=None, parent=None):
        """
        Args:
            channel_provider: callable returning the channel the Quick JV
                sweep should measure. The channel selector lives on the
                parameter tab that owns this card, not on the card itself.
        """
        super().__init__(title, parent)
        self.main_window = main_window
        self._placeholder = placeholder
        self._channel_provider = channel_provider or (lambda: 1)
        self._quick_jv_worker = None
        self._quick_jv_running = False
        self._vmpp_source = None          # None | "manual" | "quick_jv"
        self._quick_jv_voltages = []
        self._quick_jv_currents = []
        self._quick_jv_expected_points = 0
        self._build_ui()

    # -- public API -------------------------------------------------------
    def has_valid_voltage(self) -> bool:
        """True when the entry holds a parsable number."""
        try:
            float(self.vmpp_input.text())      # unit does not affect validity
            return True
        except (TypeError, ValueError):
            return False

    def voltage(self, default: float = 0.0) -> float:
        """The entered voltage, signed. `default` when the field is unusable."""
        try:
            return float(self.vmpp_input.text()) * self._vmpp_factor()
        except (TypeError, ValueError):
            return default

    def set_voltage(self, value: float, source: str = "manual"):
        """Fill the field — from a J-V run's signed Vmpp, say."""
        self._vmpp_source = source
        self.vmpp_input.setText(
            self._format_vmpp(float(value) / self._vmpp_factor()))
        self._vmpp_source = source
        self._update_vmpp_badge()

    def set_config_enabled(self, enabled: bool):
        """Lock the card while a measurement is in progress."""
        for widget in (self.vmpp_input, self.vmpp_unit,
                       self.area_field, self.compliance_field,
                       self.architecture_toggle, self.quick_jv_step_input,
                       self.manual_mode_button,
                       self.quick_jv_mode_button, self.quick_jv_start_input,
                       self.quick_jv_stop_input, self.quick_jv_rate_input,
                       self.quick_jv_button):
            widget.setEnabled(enabled)

    def _on_vmpp_changed(self, _text):
        # Provenance: a user edit resets to "manual". Quick JV sets
        # _vmpp_source before calling setText, so the badge survives.
        if self._vmpp_source != "quick_jv":
            self._vmpp_source = "manual" if self.has_valid_voltage() else None
        self._update_vmpp_badge()
        self.voltage_ready.emit(self.has_valid_voltage())

    def _build_ui(self):
        """Manual/Quick JV toggle, the voltage entry, live mini-plot,
        progress bar and results grid."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # ---- Row 1: Shared Vmpp entry + source badge --------------------
        vmpp_row = QtWidgets.QHBoxLayout()
        vmpp_row.setSpacing(8)

        self.vmpp_badge = QtWidgets.QLabel("No Vmpp")
        self.vmpp_badge.setFixedWidth(78)
        self.vmpp_badge.setAlignment(QtCore.Qt.AlignCenter)

        self.vmpp_input = QtWidgets.QLineEdit()
        self.vmpp_input.setPlaceholderText(self._placeholder)
        self.vmpp_input.textChanged.connect(self._on_vmpp_changed)

        self._update_vmpp_badge()

        # Was a static "V" label. A selector instead, so the hold voltage can
        # be typed in millivolts like every other voltage in the application.
        self.vmpp_unit = QtWidgets.QComboBox()
        self.vmpp_unit.addItems(["V", "mV"])
        self._vmpp_last_unit = self.vmpp_unit.currentText()
        self.vmpp_unit.currentTextChanged.connect(self._on_vmpp_unit_changed)

        vmpp_row.addWidget(self.vmpp_badge)
        vmpp_row.addWidget(self.vmpp_input, stretch=1)
        vmpp_row.addWidget(self.vmpp_unit)
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
        self.mode_stack = TightStackedWidget()

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

        # ---- The Quick JV sweep's own parameters -------------------------
        #
        # All seven belong to THIS sweep, not to the J-V Sweep tab. Start,
        # Stop, Step and Rate always did; Compliance, Device Area and
        # Architecture now do too, so the panel is self-contained and you can
        # set up a quick measurement without leaving the mode you are in.
        #
        # Incident power is the exception, and deliberately so: it describes
        # the lamp, not the cell or the sweep, so it is still read from the
        # Analysis tab.
        sweep_form = QtWidgets.QFormLayout()
        sweep_form.setVerticalSpacing(6)

        self.quick_jv_start_input = UnitField("voltage", self.QUICK_JV_START, "V")
        sweep_form.addRow("Start:", self.quick_jv_start_input)

        self.quick_jv_stop_input = UnitField("voltage", self.QUICK_JV_STOP, "V")
        sweep_form.addRow("Stop:", self.quick_jv_stop_input)

        self.quick_jv_step_input = UnitField("voltage", self.QUICK_JV_STEP, "mV")
        self.quick_jv_step_input.setToolTip(
            "Voltage between measured points. Smaller is slower but resolves "
            "the maximum more finely.")
        sweep_form.addRow("Step Size:", self.quick_jv_step_input)

        self.quick_jv_rate_input = UnitField("rate", self.QUICK_JV_RATE, "mV/s")
        self.quick_jv_rate_input.setToolTip(
            "Determines the measurement speed and the NPLC that follows from it.")
        sweep_form.addRow("Sweep Rate:", self.quick_jv_rate_input)

        self.compliance_field = UnitField("current", 0.180, "mA")
        self.compliance_field.setToolTip(
            "Current limit for this sweep. Set it BELOW what the cell can "
            "take -- it is the only thing standing between a wrong voltage "
            "and a damaged device.")
        sweep_form.addRow("Compliance:", self.compliance_field)

        self.area_field = UnitField("area", 0.089, "cm\u00b2")
        self.area_field.setToolTip(
            "Illuminated area of the cell. Used for the efficiency here, and "
            "for the tracking run's efficiency too.")
        sweep_form.addRow("Device Area:", self.area_field)

        # Same colours and labels as the J-V tab: p-i-n is a muted red, so
        # the selected architecture reads at a glance.
        arch_widget = QtWidgets.QWidget()
        arch_layout = QtWidgets.QHBoxLayout(arch_widget)
        arch_layout.setContentsMargins(0, 0, 0, 0)
        arch_layout.setSpacing(8)
        self._arch_nip_label = QtWidgets.QLabel("n-i-p")
        self._arch_pin_label = QtWidgets.QLabel("p-i-n")
        self.architecture_toggle = ToggleSwitch(
            on_color=tokens.ARCH_PIN_RED, off_color=tokens.ARCH_NIP_TEAL)
        self.architecture_toggle.setChecked(False)      # False = n-i-p
        self.architecture_toggle.setToolTip(
            "Which quadrant the cell generates in. The metrics are computed "
            "from the data either way; this decides whether a polarity "
            "warning is raised.")
        arch_layout.addWidget(self._arch_nip_label)
        arch_layout.addWidget(self.architecture_toggle)
        arch_layout.addWidget(self._arch_pin_label)
        arch_layout.addStretch()
        sweep_form.addRow("Device Architecture:", arch_widget)

        # Make the toggle DO something.
        #
        # Architecture changes no computed number anywhere -- the metric math
        # is deliberately architecture-agnostic and finds the maximum wherever
        # the data puts it. So on its own this toggle only labels the run, and
        # flipping it appeared to do nothing at all.
        #
        # What it is actually for: a p-i-n cell contacted like an n-i-p one
        # generates in the SECOND quadrant, so an n-i-p sweep window never
        # reaches its operating point and Vmpp comes back meaningless. The
        # J-V tab mirrors the window when you flip it; this card now does the
        # same for its own Start / Stop / Step fields.
        self.architecture_toggle.toggled.connect(self._update_architecture_labels)
        self.architecture_toggle.toggled.connect(self._offer_architecture_range)
        self._update_architecture_labels(False)

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


    def _update_vmpp_badge(self):
        """Update the source badge chip color and text."""
        base = (
            "border-radius: 10px; font-size: 10px; font-weight: 600; "
            "padding: 2px 0px;"
        )
        if self.has_valid_voltage():
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
        if (getattr(controller, 'is_busy', False)
                or getattr(controller, 'spo_running', False)
                or getattr(controller, 'mpp_running', False)):
            QtWidgets.QMessageBox.warning(
                self, "Instrument Busy",
                "Another measurement is already in progress.\n\n"
                "Please wait for it to finish or abort it before running Quick JV."
            )
            return

        # ---- Pre-validate sweep parameters (before touching hardware) ----
        start_v = self.quick_jv_start_input.value()
        stop_v = self.quick_jv_stop_input.value()

        # NO "start must be greater than stop" rule. JVProcedure sweeps in
        # whichever direction the two voltages imply, and a p-i-n cell is
        # measured from negative to positive -- so that rule made Quick JV
        # unusable on exactly the cells the architecture toggle exists for.
        # The J-V Sweep tab has never had it either.
        if start_v == stop_v:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Sweep Range",
                "Start and stop voltage are the same, so there is nothing "
                "to sweep."
            )
            return

        step = abs(self.quick_jv_step_input.value(self.QUICK_JV_STEP))
        if step <= 0:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Step Size",
                "Step size must not be zero."
            )
            return
        expected_points = int(abs(stop_v - start_v) / step) + 1
        if expected_points > 2500:
            QtWidgets.QMessageBox.warning(
                self, "Sweep Too Large",
                f"The sweep range produces {expected_points} points, which "
                f"exceeds the instrument's 2500-point buffer.\n\n"
                f"Reduce the voltage range or increase the step size."
            )
            return

        # Area, compliance and architecture: this card's own fields in Quick
        # JV mode, the J-V tab's in Manual. See device_settings().
        settings = self.device_settings()
        device_area = settings["device_area"]
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
            self._abandon_quick_jv()
            QtWidgets.QMessageBox.critical(
                self, "Connection Failed",
                f"Could not connect to instruments:\n{e}\n\n"
                "Please check that the Keithley is powered on and connected."
            )
            return
        # EVERYTHING from here to worker.start() must be guarded.
        #
        # `is_busy` is already True. If anything below raises -- a bad
        # parameter reaching JVProcedure, no channel selected, the worker
        # thread failing to start -- the flag stayed set with no run left to
        # clear it. Both `start_mpp` and `start_spo` refuse while it is True,
        # so the application insisted a J-V measurement was in progress until
        # it was restarted.
        try:
            self.main_window.update_instrument_lights()

            channel = self._channel_provider()

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
                step_size=self.quick_jv_step_input.value(self.QUICK_JV_STEP),
                sweep_rate=self.quick_jv_rate_input.value(),
                single_sweep_mode=True,
                sweep_direction="Forward",
                device_area=device_area,
                incident_power=settings["incident_power"],
                # Both of these used to be left off, so the procedure fell back to
                # its own defaults -- 0.18 A and "n-i-p" -- no matter what the JV
                # tab said. The compliance one could push 180 mA through a cell
                # the operator had limited to 20, and the architecture one made a
                # perfectly good p-i-n sweep report a polarity mismatch.
                compliance_current=settings["compliance_current"],
                architecture=settings["architecture"],
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
        except Exception as exc:                      # noqa: BLE001
            logger.error(f"Quick JV failed to start: {exc}")
            self._abandon_quick_jv()
            QtWidgets.QMessageBox.critical(
                self, "Quick JV Could Not Start",
                f"The sweep could not be started:\n\n{exc}")
            return


    def _vmpp_factor(self) -> float:
        """Volts per displayed unit."""
        return 1e-3 if self.vmpp_unit.currentText() == "mV" else 1.0

    @staticmethod
    def _format_vmpp(number: float) -> str:
        text = f"{number:.4f}".rstrip("0").rstrip(".")
        return text if text not in ("", "-") else "0"

    def _on_vmpp_unit_changed(self, _unit):
        """Restate the same voltage in the newly chosen unit.

        Without this, switching V -> mV would turn 0.5 V into 0.5 mV and the
        tracker would start a thousand times too low.
        """
        if getattr(self, "_vmpp_converting", False):
            return
        try:
            shown = float(self.vmpp_input.text())
        except (TypeError, ValueError):
            return                      # empty or mid-edit: nothing to convert
        previous = 1e-3 if self._vmpp_last_unit == "mV" else 1.0
        volts = shown * previous
        self._vmpp_converting = True
        try:
            self.vmpp_input.setText(
                self._format_vmpp(volts / self._vmpp_factor()))
        finally:
            self._vmpp_converting = False
            self._vmpp_last_unit = self.vmpp_unit.currentText()

    def architecture_range(self, is_pin: bool) -> tuple:
        """The (start, stop, step) window for this architecture, in volts.

        The same windows the J-V Sweep tab uses, mirrored about zero:
        n-i-p generates in quadrant 4 and sweeps +1.2 V -> -0.2 V, p-i-n
        generates in quadrant 2 and sweeps -1.2 V -> +0.2 V.
        """
        if is_pin:
            return (-self.QUICK_JV_START, -self.QUICK_JV_STOP,
                    -self.QUICK_JV_STEP)
        return (self.QUICK_JV_START, self.QUICK_JV_STOP, self.QUICK_JV_STEP)

    def _current_sweep_range(self):
        """What the three fields hold now, in volts, or None."""
        if not (self.quick_jv_start_input.is_valid()
                and self.quick_jv_stop_input.is_valid()
                and self.quick_jv_step_input.is_valid()):
            return None
        return (self.quick_jv_start_input.value(),
                self.quick_jv_stop_input.value(),
                self.quick_jv_step_input.value())

    def apply_architecture_range(self, is_pin: bool) -> None:
        """Write the recommended window into this card's sweep fields."""
        start, stop, step = self.architecture_range(is_pin)
        self.quick_jv_start_input.set_value(start)
        self.quick_jv_stop_input.set_value(stop)
        self.quick_jv_step_input.set_value(step)

    def _offer_architecture_range(self, is_pin: bool) -> None:
        """Offer the mirrored sweep window when the architecture changes.

        It ASKS rather than overwrites, exactly as the J-V tab does: silently
        discarding a deliberately typed window is its own data-quality bug,
        and some cells are measured over an unusual range on purpose.
        """
        recommended = self.architecture_range(is_pin)
        current = self._current_sweep_range()
        if current is not None and all(
                abs(a - b) < 1e-9 for a, b in zip(current, recommended)):
            return                                    # already correct

        other = self.architecture_range(not is_pin)
        looks_untouched = current is not None and all(
            abs(a - b) < 1e-9 for a, b in zip(current, other))
        if looks_untouched:
            self.apply_architecture_range(is_pin)     # just mirror it
            return

        name = "p-i-n" if is_pin else "n-i-p"
        start, stop, step_v = recommended
        step = step_v * 1000.0                        # the dialog says mV
        answer = QtWidgets.QMessageBox.question(
            self, "Apply recommended sweep range?",
            f"A {name} cell generates in "
            f"{'the second' if is_pin else 'the fourth'} quadrant.\n\n"
            f"Apply the recommended Quick JV range for {name}?\n"
            f"    Start {start:g} V\n    Stop {stop:g} V\n"
            f"    Step {step:g} mV\n\n"
            "Choosing No keeps your current values, which may not reach this "
            "device's operating point.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer == QtWidgets.QMessageBox.Yes:
            self.apply_architecture_range(is_pin)

    def _update_architecture_labels(self, is_pin: bool) -> None:
        """Emphasise the selected side, mute the other -- as the J-V tab does.

        The toggle track alone reads as one blob of colour; the label is what
        the operator actually reads.
        """
        active = ("font-weight: 700; font-size: 13px; color: {};"
                  " background: transparent;")
        muted = ("font-weight: 500; font-size: 13px; color: {};"
                 " background: transparent;").format(tokens.ON_SURFACE_VARIANT)
        self._arch_pin_label.setStyleSheet(
            active.format(tokens.ARCH_PIN_RED) if is_pin else muted)
        self._arch_nip_label.setStyleSheet(
            muted if is_pin else active.format(tokens.ARCH_NIP_TEAL))

    def using_quick_jv(self) -> bool:
        """True when the Quick JV panel is the visible one."""
        return bool(self.quick_jv_mode_button.isChecked())

    def _abandon_quick_jv(self) -> None:
        """Return to idle after a Quick JV that never got going.

        Clearing `is_busy` is the important part: it is what `start_mpp` and
        `start_spo` refuse on, so leaving it set locks the operator out of
        both for the rest of the session.
        """
        self._quick_jv_running = False
        try:
            self.main_window.controller.is_busy = False
        except Exception:                             # noqa: BLE001
            pass
        self.quick_jv_progress.hide()
        self._reset_quick_jv_controls()
        self._disconnect_quick_jv_instruments()
        self.quick_jv_status.setText("Idle")

    # Fallbacks match the procedure's own defaults, so a missing tab behaves
    # the way it did before rather than inventing new numbers.
    _SETTING_FALLBACKS = {
        "device_area": 0.089,
        "incident_power": 100.0,
        "compliance_current": 0.18,
        "architecture": "n-i-p",
    }

    def device_settings(self) -> dict:
        """Area, compliance and architecture for whatever runs next.

        WHERE THEY COME FROM DEPENDS ON THE MODE, and that is the point:

        * Quick JV selected -- this card's own fields. The panel is
          self-contained, so a quick measurement can be set up without
          leaving the mode you are in, and the tracking run that follows
          uses the same cell description you just typed.
        * Manual selected -- the J-V Parameters tab. In manual mode this
          card shows nothing but the voltage box, so there is nothing here
          to read and the J-V tab is the only honest source.

        Incident power is always the Analysis tab's: it describes the lamp,
        not the cell, and does not belong to either mode.
        """
        values = dict(self._SETTING_FALLBACKS)
        if self.using_quick_jv():
            values["device_area"] = self.area_field.value(
                self._SETTING_FALLBACKS["device_area"])
            values["compliance_current"] = self.compliance_field.value(
                self._SETTING_FALLBACKS["compliance_current"])
            values["architecture"] = (
                "p-i-n" if self.architecture_toggle.isChecked() else "n-i-p")
        else:
            try:
                jv = self.main_window.params_tab.get_parameters()
                for key in ("device_area", "compliance_current", "architecture"):
                    if jv.get(key) is not None:
                        values[key] = jv[key]
            except Exception as exc:                  # noqa: BLE001
                logger.warning(f"Could not read the JV parameters; using "
                               f"defaults: {exc}")
        values["incident_power"] = self._get_incident_power()

        for key in ("device_area", "incident_power", "compliance_current"):
            try:
                values[key] = float(values[key])
            except (TypeError, ValueError):
                values[key] = self._SETTING_FALLBACKS[key]
        if values["architecture"] not in ("n-i-p", "p-i-n"):
            values["architecture"] = "n-i-p"
        return values

    # Convenience readers for the tracking run, which needs two of them.
    def device_area(self, default: float = 0.089) -> float:
        return self.device_settings().get("device_area", default)

    def compliance_current(self, default: float = 0.18) -> float:
        return self.device_settings().get("compliance_current", default)

    def architecture(self) -> str:
        return self.device_settings().get("architecture", "n-i-p")

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
            channel = self._channel_provider()
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
            self.vmpp_input.setText(
                self._format_vmpp(vmpp_v / self._vmpp_factor()))

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
        Guarded: if a measurement is active, leave the controls locked —
        that run's set_config_enabled(False) wins."""
        controller = self.main_window.controller
        if (getattr(controller, 'spo_running', False)
                or getattr(controller, 'mpp_running', False)):
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
