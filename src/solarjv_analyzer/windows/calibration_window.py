"""
Calibration Window for System Validation

Provides a mandatory calibration gate before accessing the main application.
Includes hardware connection verification, reference cell measurement,
and pass/fail criteria based on Isc tolerance. Calibration data is saved
permanently in the user's reports directory.
"""

import logging
import os
import tempfile
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.display.widgets import PlotWidget, BrowserWidget
from pymeasure.experiment import Results

from solarjv_analyzer.instruments.instrument_manager import InstrumentManager
from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.utils.directory_manager import DirectoryManager
from solarjv_analyzer.gui.style import BASE_STYLESHEET, FONT_FAMILY
from solarjv_analyzer.gui.calibration_style import (
    calibration_stylesheet, load_design_fonts, label_font,
    PRIMARY, ERROR, OK_GREEN, SECONDARY, OUTLINE_VARIANT,
    SURFACE_LOW, ON_SURFACE_VARIANT, WARN_AMBER, WARN_AMBER_BG,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helper Classes (unchanged)
# -------------------------------------------------------------------------

class BrowserProgressRelay(QtCore.QObject):
    """Relay signals from non-QObject items in a thread-safe manner."""
    progress_signal = QtCore.pyqtSignal(float)


class SignalBrowserItem(BrowserItem):
    """BrowserItem that emits a Qt signal when progress is updated."""

    def __init__(self, results, color, progress_callback=None):
        super().__init__(results, color)
        self.relay = BrowserProgressRelay()
        if progress_callback:
            self.relay.progress_signal.connect(progress_callback)

    def setProgress(self, progress):
        super().setProgress(progress)
        self.relay.progress_signal.emit(progress)


class CalibrationProcedure(JVProcedure):
    """Keithley-only J-V procedure for the calibration gate.

    The reference cell is wired directly to the SourceMeter, so there is no
    MUX. ``JVProcedure`` declares ``mux`` as a pymeasure ``Parameter`` and its
    ``check_parameters()`` rejects a ``None`` value, which would crash the
    calibration run. This subclass skips that single parameter so ``mux=None``
    is allowed here — and *only* here. Main-window and SPO runs keep using
    ``JVProcedure``/``SpoProcedure`` unchanged, with a real MUX.
    """

    def check_parameters(self):
        for name, parameter in self._parameters.items():
            if name == "mux":
                continue
            value = getattr(self, name)
            if value is None:
                raise NameError(
                    f"Missing {parameter.__class__} '{name}' in {self.__class__}"
                )


class CalibrationChecklistDialog(QtWidgets.QDialog):
    """Startup checklist dialog ensuring system readiness."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Readiness Check")
        self.resize(480, 460)
        self._setup_ui()

    def _setup_ui(self):
        """Build the checklist dialog UI (Precision Instrument style)."""
        self.setStyleSheet(calibration_stylesheet())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(10)

        title_row = QtWidgets.QHBoxLayout()
        eyebrow = QtWidgets.QLabel("PRE-CALIBRATION")
        eyebrow.setObjectName("CardTitle")
        eyebrow.setFont(label_font(11))
        title_row.addWidget(eyebrow)
        title_row.addStretch()
        layout.addLayout(title_row)

        header = QtWidgets.QLabel("System Readiness Check")
        header.setObjectName("DialogHeader")
        layout.addWidget(header)

        desc = QtWidgets.QLabel("Verify all conditions before energizing the system.")
        desc.setObjectName("DialogSub")
        layout.addWidget(desc)
        layout.addSpacing(10)

        self.steps = [
            "Turn ON Chiller / Sun Sim / Keithley",
            "Wavelabs: Load 'AM1.5G' Recipe",
            "Place Si-Reference Cell (RERA)",
            "Verify Lamp Height (46.8 cm)",
        ]

        self.checks = []
        for n, step in enumerate(self.steps, start=1):
            row = QtWidgets.QFrame()
            row.setObjectName("ChecklistRow")
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(12, 8, 12, 8)
            row_layout.setSpacing(12)

            badge = QtWidgets.QLabel(str(n))
            badge.setObjectName("StepBadge")
            badge.setFixedSize(22, 22)
            badge.setAlignment(QtCore.Qt.AlignCenter)
            row_layout.addWidget(badge)

            checkbox = QtWidgets.QCheckBox(step)
            checkbox.setCursor(QtCore.Qt.PointingHandCursor)
            checkbox.stateChanged.connect(self._validate_checklist)
            row_layout.addWidget(checkbox, stretch=1)

            layout.addWidget(row)
            self.checks.append(checkbox)

        layout.addStretch()

        self.ok_btn = QtWidgets.QPushButton("CONFIRM READINESS")
        self.ok_btn.setObjectName("ConfirmBtn")
        self.ok_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)

    def _validate_checklist(self):
        """Enable confirm button only when all items are checked."""
        all_checked = all(cb.isChecked() for cb in self.checks)
        self.ok_btn.setEnabled(all_checked)


# -------------------------------------------------------------------------
# Main Calibration Window
# -------------------------------------------------------------------------

class CalibrationWindow(QtWidgets.QMainWindow):
    """
    Calibration gate for system validation before main application access.

    Verifies reference cell measurement against target Isc within tolerance.
    Provides skip option for emergency access when calibration is not required.
    """

    calibration_passed = QtCore.pyqtSignal(object)
    logged_out = QtCore.pyqtSignal()

    DEFAULT_TARGET_ISC = 0.0596
    DEFAULT_TARGET_JSC = 14.9
    DEFAULT_TOLERANCE = 5.0
    DEFAULT_AREA = 4.0
    DEFAULT_START_V = 0.7
    DEFAULT_STOP_V = -0.2
    DEFAULT_STEP_V = -0.01

    def __init__(self, username, parent=None, instrument_manager=None):
        super().__init__(parent)
        self.username = username
        self.setWindowTitle("System Calibration")
        self.resize(1360, 860)

        # One InstrumentManager owns the VISA session for the whole process;
        # main.py passes the same object to the main window. A manager is
        # created here only when this window is built standalone (tests).
        self.instrument_manager = (
            instrument_manager if instrument_manager is not None else InstrumentManager()
        )
        self.manager = None
        self.checklist_confirmed = False
        # True once the instruments have been handed to the main window: the
        # close that follows the hand-off must NOT tear the hardware down.
        self._handed_off = False

        self.dir_manager = DirectoryManager(username=self.username, parent=self, mode="Calibration")

        load_design_fonts()
        self._setup_ui()
        self._connect_hardware()

        QtCore.QTimer.singleShot(200, self.launch_checklist_dialog)

    # -------------------------------------------------------------------------
    # Stylesheet — Precision Instrument design system (calibration scope only)
    # -------------------------------------------------------------------------
    @staticmethod
    def _app_stylesheet() -> str:
        return calibration_stylesheet()

    # -------------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------------

    def _setup_ui(self):
        """Build the calibration window interface (Precision Instrument design:
        header bar, card-based left rail, plot + measurements on the right)."""
        pg.setConfigOption('background', '#ffffff')
        pg.setConfigOption('foreground', '#41484b')

        self.setStyleSheet(self._app_stylesheet())

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._create_header_bar())

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QHBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(24)

        content_layout.addWidget(self._create_left_panel())

        right_column = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(24)
        right_layout.addWidget(self._create_plot_panel(), stretch=3)
        right_layout.addWidget(self._create_measurements_card(), stretch=0)
        content_layout.addWidget(right_column, stretch=1)

        outer.addWidget(content, stretch=1)

    # ---- Header bar -------------------------------------------------------

    def _create_header_bar(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName("HeaderBar")
        bar.setFixedHeight(64)
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(16)

        brand = QtWidgets.QLabel("IPV")
        brand.setObjectName("Brand")
        layout.addWidget(brand)

        divider = QtWidgets.QFrame()
        divider.setObjectName("HeaderDivider")
        divider.setFixedSize(1, 26)
        layout.addWidget(divider)

        title = QtWidgets.QLabel("CALIBRATION")
        title.setObjectName("HeaderTitle")
        title.setFont(label_font(12))
        layout.addWidget(title)

        layout.addStretch()

        # Instrument status pill — the dot + text update in _check_readiness,
        # and clicking the pill retries the connection when disconnected.
        self.status_pill = QtWidgets.QPushButton()
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setCursor(QtCore.Qt.PointingHandCursor)
        self.status_pill.clicked.connect(self._on_status_pill_clicked)
        pill_layout = QtWidgets.QHBoxLayout(self.status_pill)
        pill_layout.setContentsMargins(12, 5, 14, 5)
        pill_layout.setSpacing(8)
        # QPushButton's sizeHint ignores child layouts — force the button to
        # adopt the layout's minimum size or the pill collapses to ~2 chars.
        pill_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        self.keithley_light = self._create_light()
        self.keithley_light.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        pill_layout.addWidget(self.keithley_light)

        self.keithley_status_label = QtWidgets.QLabel("KEITHLEY 2400 · CONNECTING…")
        self.keithley_status_label.setFont(label_font(11))
        self.keithley_status_label.setStyleSheet(
            f"color: {ON_SURFACE_VARIANT}; background: transparent; border: none;"
        )
        self.keithley_status_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        pill_layout.addWidget(self.keithley_status_label)
        layout.addWidget(self.status_pill)

        # User chip + logout button match the main window's top-right corner.
        # Styled inline (not via objectName QSS) — the same approach the main
        # window uses, so the two render identically. Font pinned explicitly
        # because calibration_stylesheet's QPushButton rule forces mono.
        user_chip = QtWidgets.QLabel(f"👤 {self.username}")
        user_chip.setStyleSheet(f"""
            QLabel {{
                font-family: {FONT_FAMILY};
                background-color: #f1f5f9;
                color: #334155;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
                margin-right: 8px;
            }}
        """)
        # The header bar's row is taller than the chip (driven by the status
        # pill); a QLabel's vertical size policy is Preferred, so the layout
        # stretches it into a tall pill. Pin its height to the sizeHint so it
        # stays a compact 34px pill like the main window's chip.
        user_chip.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        layout.addWidget(user_chip)

        self.logout_button = QtWidgets.QPushButton("Logout")
        self.logout_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.logout_button.setStyleSheet(f"""
            QPushButton {{
                font-family: {FONT_FAMILY};
                background-color: #fee2e2;
                color: #ef4444;
                border: none;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #fca5a5;
            }}
        """)
        self.logout_button.clicked.connect(self._on_logout)
        layout.addWidget(self.logout_button)

        return bar

    def _on_status_pill_clicked(self):
        """Header pill acts as a reconnect affordance while disconnected."""
        if not self.instrument_manager.is_keithley_alive():
            self.keithley_status_label.setText("KEITHLEY 2400 · CONNECTING…")
            QtWidgets.QApplication.processEvents()
            self._connect_hardware()

    # ---- Left rail --------------------------------------------------------

    def _create_left_panel(self):
        """Create the scrollable left control rail (cards)."""
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setFixedWidth(440)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        left_content = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_content)
        left_layout.setSpacing(16)
        left_layout.setContentsMargins(0, 0, 8, 0)

        left_layout.addWidget(self._create_protocol_group())
        left_layout.addWidget(self._create_config_group())
        left_layout.addWidget(self._create_directory_group())
        left_layout.addWidget(self._create_actions_group())
        left_layout.addWidget(self._create_proceed_buttons())

        left_layout.addStretch()
        scroll_area.setWidget(left_content)
        return scroll_area

    @staticmethod
    def _card(title_text: str):
        """Build an empty design-system card; returns (frame, inner_layout)."""
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(20, 16, 20, 18)
        layout.setSpacing(12)

        if title_text:
            title = QtWidgets.QLabel(title_text)
            title.setObjectName("CardTitle")
            title.setFont(label_font(12))
            layout.addWidget(title)

            rule = QtWidgets.QFrame()
            rule.setObjectName("CardTitleRule")
            rule.setFixedHeight(1)
            layout.addWidget(rule)

        return card, layout

    def _create_protocol_group(self):
        card, layout = self._card("PROTOCOL REFERENCE")

        steps = [
            "Turn ON Chiller / Sun Sim / Keithley",
            "Wavelabs Load 'AM1.5G' Recipe",
            "Place Si-Reference Cell (RERA)",
            "Verify Lamp Height (46.8 cm)",
        ]
        for n, text in enumerate(steps, start=1):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(12)

            badge = QtWidgets.QLabel(str(n))
            badge.setObjectName("StepBadge")
            badge.setFixedSize(22, 22)
            badge.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(badge, alignment=QtCore.Qt.AlignTop)

            step = QtWidgets.QLabel(text)
            step.setObjectName("StepText")
            step.setWordWrap(True)
            row.addWidget(step, stretch=1)

            layout.addLayout(row)

        return card

    def _create_config_group(self):
        card, layout = self._card("")

        # Title row with the unlock chip on the right (design: chip in header)
        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("PARAMETERS")
        title.setObjectName("CardTitle")
        title.setFont(label_font(12))
        title_row.addWidget(title)
        title_row.addStretch()

        self.unlock_btn = QtWidgets.QPushButton("UNLOCK SETTINGS")
        self.unlock_btn.setObjectName("UnlockButton")
        self.unlock_btn.setCheckable(True)
        self.unlock_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.unlock_btn.toggled.connect(self._toggle_inputs)
        title_row.addWidget(self.unlock_btn)
        layout.addLayout(title_row)

        rule = QtWidgets.QFrame()
        rule.setObjectName("CardTitleRule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)

        # Two-column grid of labelled fields (design layout)
        self.spin_target_isc = QtWidgets.QDoubleSpinBox()
        self.spin_target_isc.setDecimals(4)
        self.spin_target_isc.setValue(self.DEFAULT_TARGET_ISC)
        self.spin_target_isc.setSuffix("  A")

        self.spin_tolerance = QtWidgets.QDoubleSpinBox()
        self.spin_tolerance.setValue(self.DEFAULT_TOLERANCE)
        self.spin_tolerance.setSuffix("  %")

        self.spin_area = QtWidgets.QDoubleSpinBox()
        self.spin_area.setValue(self.DEFAULT_AREA)
        self.spin_area.setSuffix("  cm²")

        self.spin_step_v = QtWidgets.QDoubleSpinBox()
        self.spin_step_v.setRange(-1, 1)
        self.spin_step_v.setDecimals(3)
        self.spin_step_v.setValue(self.DEFAULT_STEP_V)
        self.spin_step_v.setSuffix("  V")

        self.spin_start_v = QtWidgets.QDoubleSpinBox()
        self.spin_start_v.setRange(-10, 10)
        self.spin_start_v.setValue(self.DEFAULT_START_V)
        self.spin_start_v.setSuffix("  V")

        self.spin_stop_v = QtWidgets.QDoubleSpinBox()
        self.spin_stop_v.setRange(-10, 10)
        self.spin_stop_v.setValue(self.DEFAULT_STOP_V)
        self.spin_stop_v.setSuffix("  V")

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        fields = [
            ("Target Isc", self.spin_target_isc), ("Tolerance", self.spin_tolerance),
            ("Ref. Area", self.spin_area), ("Step Size", self.spin_step_v),
            ("Start V", self.spin_start_v), ("Stop V", self.spin_stop_v),
        ]
        for idx, (name, widget) in enumerate(fields):
            r, c = divmod(idx, 2)
            lbl = QtWidgets.QLabel(name)
            lbl.setObjectName("FieldLabel")
            # Let fields compress below their (suffix-inflated) size hint so
            # the two-column grid fits the rail width without clipping.
            widget.setMinimumWidth(110)
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Fixed)
            grid.addWidget(lbl, r * 2, c)
            grid.addWidget(widget, r * 2 + 1, c)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._toggle_inputs(False)
        return card

    def _create_directory_group(self):
        card, layout = self._card("OUTPUT")
        dir_widget = self.dir_manager.create_directory_widget(title="")
        # The info label inside carries a long absolute path; without word
        # wrap its minimum width blows out the whole left rail and clips
        # every card against the scroll viewport.
        for lbl in dir_widget.findChildren(QtWidgets.QLabel):
            lbl.setWordWrap(True)
        layout.addWidget(dir_widget)
        return card

    def _create_actions_group(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.run_button = QtWidgets.QPushButton("RUN CALIBRATION")
        self.run_button.setObjectName("RunButton")
        self.run_button.setMinimumHeight(48)
        self.run_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._on_run_clicked)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("SweepProgress")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)

        self.lbl_status_text = QtWidgets.QLabel("WAITING")
        self.lbl_status_text.setObjectName("StatusChip")
        self.lbl_status_text.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_status_text)
        return widget

    def _create_proceed_buttons(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.proceed_button = QtWidgets.QPushButton("PROCEED TO MAIN DASHBOARD  →")
        self.proceed_button.setObjectName("ProceedButton")
        self.proceed_button.setMinimumHeight(44)
        self.proceed_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self._on_proceed)
        layout.addWidget(self.proceed_button)

        # Design: two-line outlined pill — "SKIP PROTOCOL / NOT RECOMMENDED"
        self.skip_button = QtWidgets.QPushButton("⊘  SKIP PROTOCOL\nNOT RECOMMENDED")
        self.skip_button.setObjectName("SkipButton")
        self.skip_button.setMinimumHeight(48)
        self.skip_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.skip_button.clicked.connect(self._on_skip)
        layout.addWidget(self.skip_button)

        warning_label = QtWidgets.QLabel("⚠ Skipping calibration may affect measurement accuracy")
        warning_label.setStyleSheet(
            f"color: {WARN_AMBER}; background-color: {WARN_AMBER_BG};"
            " border: 1px solid #fde68a; border-radius: 8px;"
            " padding: 5px 8px; font-size: 10px;"
        )
        warning_label.setAlignment(QtCore.Qt.AlignCenter)
        warning_label.setWordWrap(True)
        warning_label.setMinimumHeight(26)
        layout.addWidget(warning_label)
        return widget

    # ---- Right column: plot card + measurements card ----------------------

    def _create_plot_panel(self):
        card, layout = self._card("")

        # Card header: title left, pymeasure's own X/Y axis selectors right.
        header_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("IV CHARACTERISTICS")
        title.setObjectName("CardTitle")
        title.setFont(label_font(12))
        header_row.addWidget(title)
        header_row.addStretch()
        layout.addLayout(header_row)

        rule = QtWidgets.QFrame()
        rule.setObjectName("CardTitleRule")
        rule.setFixedHeight(1)
        layout.addWidget(rule)

        self.plot_widget = PlotWidget(
            name="Calibration Curve",
            columns=["Voltage (V)", "Current (A)"],
            x_axis="Voltage (V)",
            y_axis="Current (A)"
        )
        plot = self.plot_widget.plot
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setLabel('left', 'Current', units='A', **{'font-size': '9pt'})
        plot.setLabel('bottom', 'Voltage', units='V', **{'font-size': '9pt'})
        plot.getAxis('left').setPen(pg.mkPen(color=OUTLINE_VARIANT, width=1))
        plot.getAxis('bottom').setPen(pg.mkPen(color=OUTLINE_VARIANT, width=1))
        plot.getAxis('left').setTextPen(pg.mkPen(color=SECONDARY))
        plot.getAxis('bottom').setTextPen(pg.mkPen(color=SECONDARY))

        # ---- Dashed target-Isc line (error red, matches design) ----------
        # Q4 convention: generated current is negative, so the target sits
        # at -|target Isc|. Follows the Target Isc spinbox live.
        self._target_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(ERROR, width=1, style=QtCore.Qt.DashLine),
        )
        self._target_line.setOpacity(0.5)
        plot.addItem(self._target_line, ignoreBounds=True)
        self._update_target_line()
        self.spin_target_isc.valueChanged.connect(self._update_target_line)

        # ---- Crosshair + cursor readout ----------------------------------
        pen = pg.mkPen(OUTLINE_VARIANT, width=1)
        self._vline = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._hline = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        for line in (self._vline, self._hline):
            line.setVisible(False)
            plot.addItem(line, ignoreBounds=True)

        self._cursor_label = pg.TextItem(
            anchor=(0, 1), color=ON_SURFACE_VARIANT,
            border=pg.mkPen(OUTLINE_VARIANT),
            fill=pg.mkBrush(255, 255, 255, 235),
        )
        self._cursor_label.setVisible(False)
        plot.addItem(self._cursor_label, ignoreBounds=True)

        self._crosshair_proxy = pg.SignalProxy(
            plot.scene().sigMouseMoved, rateLimit=45,
            slot=self._on_plot_mouse_moved,
        )

        layout.addWidget(self.plot_widget, stretch=1)
        return card

    def _update_target_line(self):
        self._target_line.setPos(-abs(self.spin_target_isc.value()))

    def _on_plot_mouse_moved(self, evt):
        """Crosshair follows the mouse with a live V/I readout box."""
        pos = evt[0]
        plot = self.plot_widget.plot
        vb = plot.getViewBox()
        if plot.sceneBoundingRect().contains(pos):
            p = vb.mapSceneToView(pos)
            self._vline.setPos(p.x())
            self._hline.setPos(p.y())
            self._cursor_label.setHtml(
                f"<div style='font-family: monospace; font-size: 10px;'>"
                f"V: {p.x():+.3f} V<br>I: {p.y():+.4f} A</div>"
            )
            self._cursor_label.setPos(p.x(), p.y())
            for item in (self._vline, self._hline, self._cursor_label):
                item.setVisible(True)
        else:
            for item in (self._vline, self._hline, self._cursor_label):
                item.setVisible(False)

    def _create_measurements_card(self):
        card, layout = self._card("MEASUREMENTS")
        card.setMinimumHeight(150)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(32)
        grid.setVerticalSpacing(4)

        # -- Isc column ----------------------------------------------------
        name_isc = QtWidgets.QLabel("Isc")
        name_isc.setObjectName("MetricName")
        self.lbl_isc_range = QtWidgets.QLabel("")
        self.lbl_isc_range.setObjectName("MiniLabel")
        self.lbl_measured_isc = QtWidgets.QLabel("--.-- mA")
        self.lbl_measured_isc.setObjectName("BigValue")
        self.lbl_measured_isc.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_target_isc = QtWidgets.QLabel("")
        self.lbl_target_isc.setObjectName("MiniLabel")
        self.lbl_isc_pct = QtWidgets.QLabel("--%")
        self.lbl_isc_pct.setObjectName("MiniLabel")
        self.lbl_isc_pct.setAlignment(QtCore.Qt.AlignRight)
        self.isc_target_bar = QtWidgets.QProgressBar()
        self.isc_target_bar.setRange(0, 100)
        self.isc_target_bar.setValue(0)
        self.isc_target_bar.setTextVisible(False)
        self.isc_target_bar.setFixedHeight(4)

        grid.addWidget(name_isc, 0, 0)
        grid.addWidget(self.lbl_measured_isc, 0, 1)
        grid.addWidget(self.lbl_isc_range, 1, 0)
        grid.addWidget(self.lbl_isc_pct, 1, 1)
        grid.addWidget(self.lbl_target_isc, 2, 0)
        grid.addWidget(self.isc_target_bar, 3, 0, 1, 2)

        # -- vertical divider ---------------------------------------------
        divider = QtWidgets.QFrame()
        divider.setObjectName("CardTitleRule")
        divider.setFixedWidth(1)
        grid.addWidget(divider, 0, 2, 4, 1)

        # -- Jsc column ----------------------------------------------------
        name_jsc = QtWidgets.QLabel("Jsc")
        name_jsc.setObjectName("MetricName")
        self.lbl_jsc_range = QtWidgets.QLabel("")
        self.lbl_jsc_range.setObjectName("MiniLabel")
        self.lbl_measured_jsc = QtWidgets.QLabel("--.-- mA/cm²")
        self.lbl_measured_jsc.setObjectName("BigValue")
        self.lbl_measured_jsc.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.lbl_target_jsc = QtWidgets.QLabel("")
        self.lbl_target_jsc.setObjectName("MiniLabel")
        self.lbl_jsc_pct = QtWidgets.QLabel("--%")
        self.lbl_jsc_pct.setObjectName("MiniLabel")
        self.lbl_jsc_pct.setAlignment(QtCore.Qt.AlignRight)
        self.jsc_target_bar = QtWidgets.QProgressBar()
        self.jsc_target_bar.setRange(0, 100)
        self.jsc_target_bar.setValue(0)
        self.jsc_target_bar.setTextVisible(False)
        self.jsc_target_bar.setFixedHeight(4)

        grid.addWidget(name_jsc, 0, 3)
        grid.addWidget(self.lbl_measured_jsc, 0, 4)
        grid.addWidget(self.lbl_jsc_range, 1, 3)
        grid.addWidget(self.lbl_jsc_pct, 1, 4)
        grid.addWidget(self.lbl_target_jsc, 2, 3)
        grid.addWidget(self.jsc_target_bar, 3, 3, 1, 2)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(4, 1)
        layout.addLayout(grid)

        # Keep target/min-max labels synced with the parameter spinboxes.
        self._refresh_measurement_targets()
        self.spin_target_isc.valueChanged.connect(self._refresh_measurement_targets)
        self.spin_tolerance.valueChanged.connect(self._refresh_measurement_targets)
        self.spin_area.valueChanged.connect(self._refresh_measurement_targets)

        return card

    def _refresh_measurement_targets(self):
        """Recompute target / MAX / MIN annotations from the parameters."""
        target_a = self.spin_target_isc.value()
        tol = self.spin_tolerance.value() / 100.0
        area = self.spin_area.value() or 1.0

        t_ma = target_a * 1000.0
        self.lbl_target_isc.setText(f"Target: {t_ma:.1f} mA")
        self.lbl_isc_range.setText(
            f"MAX: {t_ma * (1 + tol):.1f}   MIN: {t_ma * (1 - tol):.1f}"
        )
        t_jsc = t_ma / area
        self.lbl_target_jsc.setText(f"Target: {t_jsc:.1f} mA/cm²")
        self.lbl_jsc_range.setText(
            f"MAX: {t_jsc * (1 + tol):.1f}   MIN: {t_jsc * (1 - tol):.1f}"
        )

    def _update_measurement_card(self, measured_isc_a: float, measured_jsc: float):
        """Push a finished measurement into the % of target displays."""
        target_a = self.spin_target_isc.value()
        area = self.spin_area.value() or 1.0
        if target_a > 0:
            pct = abs(measured_isc_a) / target_a * 100.0
            self.lbl_isc_pct.setText(f"{pct:.1f}%")
            self.isc_target_bar.setValue(int(max(0, min(100, pct))))
            t_jsc = target_a * 1000.0 / area
            pct_j = abs(measured_jsc) / t_jsc * 100.0 if t_jsc > 0 else 0.0
            self.lbl_jsc_pct.setText(f"{pct_j:.1f}%")
            self.jsc_target_bar.setValue(int(max(0, min(100, pct_j))))

    @staticmethod
    def _create_light():
        label = QtWidgets.QLabel("")
        label.setFixedSize(10, 10)
        label.setStyleSheet("border-radius: 5px; background: #cbd5e1; border: none;")
        return label

    # -------------------------------------------------------------------------
    # Manager Setup
    # -------------------------------------------------------------------------
    def _setup_manager(self):
        if self.manager:
            try:
                if self.manager.is_running():
                    self.manager.abort()
                self.manager.finished.disconnect()
                self.manager.abort_returned.disconnect()
            except Exception:
                pass

        columns = ["Voltage (V)", "Current (A)"]
        dummy_browser = BrowserWidget(JVProcedure, ["active_channel"], columns)
        dummy_browser.hide()
        self._dummy_browser_widget = dummy_browser

        self.manager = Manager(
            [self.plot_widget],
            self._dummy_browser_widget.browser,
            log_level=logging.INFO,
            parent=self
        )
        self.manager.finished.connect(self._on_sweep_finished)
        self.manager.abort_returned.connect(self._on_abort_complete)

    # -------------------------------------------------------------------------
    # Hardware Management
    # -------------------------------------------------------------------------
    def _connect_hardware(self):
        # Calibration needs ONLY the Keithley — the reference cell is wired
        # directly to the SourceMeter. The MUX is connected later by the
        # main window when a multi-channel measurement actually needs it.
        try:
            self.instrument_manager.connect_keithley(simulation=False)
            logger.info("Keithley connection attempted")
        except Exception as e:
            logger.warning(f"Keithley connection failed: {e}")
            self.instrument_manager.keithley = None

        if self.instrument_manager.keithley:
            try:
                self.instrument_manager.keithley.id
            except Exception:
                logger.error("Keithley detected but unresponsive")
                self.instrument_manager.keithley = None

        self._check_readiness()

    def _check_readiness(self):
        # Only the Keithley gates calibration — no MUX involved. Liveness,
        # not "is not None": a closed session leaves the attribute set.
        keithley_ok = self.instrument_manager.is_keithley_alive()

        k_color = OK_GREEN if keithley_ok else '#ef4444'
        self.keithley_light.setStyleSheet(
            f"border-radius:5px; background:{k_color}; border: none;"
        )
        # Header pill text mirrors the connection state.
        if hasattr(self, 'keithley_status_label'):
            state = "CONNECTED" if keithley_ok else "DISCONNECTED — CLICK TO RETRY"
            colour = OK_GREEN if keithley_ok else '#ef4444'
            self.keithley_status_label.setText(f"KEITHLEY 2400 · {state}")
            self.keithley_status_label.setStyleSheet(
                f"color: {colour}; background: transparent; border: none;"
            )

        is_running = self.manager.is_running() if self.manager else False
        is_abort_state = self.run_button.text() in ["ABORT", "ABORTING..."]

        if is_abort_state:
            self.run_button.setEnabled(True)
            self.run_button.setStyleSheet("background-color: #ef4444; color: white;")
            return

        hardware_ok = keithley_ok
        if hardware_ok and not is_running and self.checklist_confirmed:
            self.run_button.setEnabled(True)
            if self.run_button.text() == "RESTART":
                self.run_button.setStyleSheet("background-color: #f59e0b; color: white;")
            elif self.run_button.text() == "RE-CALIBRATE":
                pass  # keep the pass-state styling applied by _set_pass_state
            else:
                self.run_button.setText("RUN CALIBRATION")
                self.run_button.setStyleSheet("")
                self.run_button.setObjectName("RunButton")
                self.run_button.style().unpolish(self.run_button)
                self.run_button.style().polish(self.run_button)

            # The status pill reads "WAITING" until a run starts. Once the
            # checklist is confirmed and hardware is connected, flip it to
            # READY so the user isn't left staring at a stale "Awaiting".
            if self.lbl_status_text.text() == "WAITING":
                self.lbl_status_text.setText("READY")
                self.lbl_status_text.setStyleSheet(
                    f"background: #d1e3e9; color: {PRIMARY};"
                    " padding: 8px; border-radius: 8px;"
                )
        else:
            if not hardware_ok:
                # Keep the button ENABLED as an explicit retry affordance —
                # a dead "Hardware Disconnected" pill with no way to
                # reconnect forced an app restart after powering on late.
                self.run_button.setEnabled(True)
                self.run_button.setText("Hardware Disconnected — Retry")
                self.run_button.setStyleSheet(
                    "background-color: #f59e0b; color: white; font-weight: 600;"
                )
            elif not self.checklist_confirmed:
                self.run_button.setEnabled(False)
                self.run_button.setText("Awaiting Checklist")
                self.run_button.setStyleSheet("background-color: #cbd5e1; color: #64748b;")
            else:
                self.run_button.setEnabled(False)
                self.run_button.setStyleSheet("background-color: #cbd5e1; color: #64748b;")

    # -------------------------------------------------------------------------
    # UI Callbacks
    # -------------------------------------------------------------------------
    def _toggle_inputs(self, checked):
        widgets = [
            self.spin_target_isc, self.spin_tolerance, self.spin_area,
            self.spin_start_v, self.spin_stop_v, self.spin_step_v
        ]
        for widget in widgets:
            widget.setReadOnly(not checked)
            widget.setEnabled(checked)
        self.unlock_btn.setText("Lock Settings" if checked else "Unlock Settings")

    def _on_run_clicked(self):
        if self.manager and self.manager.is_running():
            self.manager.abort()
            self.run_button.setText("ABORTING...")
            self.run_button.setEnabled(False)
        elif not self.instrument_manager.is_keithley_alive():
            # Retry affordance: the button reads "Hardware Disconnected —
            # Retry" in this state. Reconnect and refresh the status pill.
            self.run_button.setText("Connecting...")
            self.run_button.setEnabled(False)
            QtWidgets.QApplication.processEvents()
            self._connect_hardware()
            if not self.instrument_manager.is_keithley_alive():
                QtWidgets.QMessageBox.warning(
                    self, "Hardware Connection",
                    "Could not connect to the Keithley 2400.\n\n"
                    "Check power, cables, and the configured port, then "
                    "click the button to retry."
                )
            # _connect_hardware() ends with _check_readiness(), which
            # restores the correct button text/state for either outcome.
        else:
            self._start_calibration()

    def _on_proceed(self):
        if self.manager and self.manager.is_running():
            self.manager.abort()
        # Set BEFORE the emit: the slot (main.py's launch_main_app) runs
        # synchronously and closes this window, so closeEvent fires inside
        # the emit and must already know this is a hand-off, not a shutdown.
        self._handed_off = True
        self.calibration_passed.emit({
            'instrument_manager': self.instrument_manager,
            'output_directory': self.dir_manager.get_base_directory()
        })
        self.close()

    def _on_skip(self):
        reply = QtWidgets.QMessageBox.warning(
            self, "Skip Calibration",
            "WARNING: Skipping calibration may result in inaccurate measurements.\n\n"
            "Only use this option if the application is frozen and you need to recover data.\n\n"
            "Are you sure you want to skip calibration?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            if self.manager and self.manager.is_running():
                self.manager.abort()
            self._handed_off = True   # see _on_proceed
            self.calibration_passed.emit({
                'instrument_manager': self.instrument_manager,
                'output_directory': self.dir_manager.get_base_directory()
            })
            self.close()

    def _on_logout(self):
        self.logged_out.emit()

    def closeEvent(self, event):
        """Handle the window close (X / Cmd-W) path: de-energize the Keithley
        and disconnect before the process exits, mirroring
        JVAnalyzerWindow.closeEvent. Calibration is Keithley-only (no MUX).

        The hand-off to the main window also closes this window (Proceed and
        Skip emit ``calibration_passed``, whose slot builds JVAnalyzerWindow
        and then closes this one). That close must leave the hardware alone:
        de-energizing, disconnecting or ending the session here would hand the
        main window a closed VISA session — every subsequent sweep then died
        with "Invalid session handle" on its first :OUTP OFF, while the status
        light still showed green. ``_handed_off`` separates the two cases.
        """
        # 1. Abort any running calibration sweep (both paths).
        if self.manager and self.manager.is_running():
            try:
                self.manager.abort()
            except Exception:
                pass

        if self._handed_off:
            logger.info(
                "Calibration window closing after hand-off — instruments and "
                "session stay with the main window."
            )
            event.accept()
            return

        logger.info("Calibration window close requested — performing safety shutdown.")

        # 2. Turn off the output and put the Keithley in an idle state.
        try:
            if self.instrument_manager.is_keithley_alive():
                self.instrument_manager.keithley.write(":OUTP OFF")
                self.instrument_manager.keithley.write(":ABOR")
        except Exception:
            pass

        # 3. Disconnect (runs the instrument's own shutdown and frees the port).
        try:
            self.instrument_manager.disconnect_keithley()
        except Exception:
            pass

        # 4. End the session so main.py's relogin check is consistent.
        try:
            from solarjv_analyzer.auth.session import SessionManager
            SessionManager.end_session()
        except Exception:
            pass

        event.accept()

    # -------------------------------------------------------------------------
    # Calibration Execution
    # -------------------------------------------------------------------------
    def _get_calibration_file_path(self):
        """Generate permanent file path for calibration data."""
        return self.dir_manager.get_file_path(prefix="calibration", create=True)

    def _start_calibration(self):
        self._setup_manager()

        self.proceed_button.setEnabled(False)
        self.lbl_measured_isc.setText("--.-- mA")
        self.lbl_measured_jsc.setText("--.-- mA/cm²")
        self.lbl_status_text.setText("MEASURING...")
        self.lbl_status_text.setStyleSheet(
            f"background: #d1e3e9; color: {PRIMARY}; padding: 8px; border-radius: 8px;"
        )

        self.progress_bar.setValue(0)
        self.run_button.setText("ABORT")
        self.run_button.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold;")
        self.run_button.setEnabled(True)

        try:
            # Reference cell is wired directly to the Keithley — no MUX,
            # no channel selection. Channel 1 is used as a nominal label.
            channel_id = 1
            area = self.spin_area.value()
            start_v = self.spin_start_v.value()
            stop_v = self.spin_stop_v.value()
            step_v = self.spin_step_v.value()
        except ValueError:
            return

        # Permanent file path for calibration report
        file_path = self._get_calibration_file_path()
        self._calibration_file_path = file_path

        procedure = CalibrationProcedure(
            instrument=self.instrument_manager.keithley,
            mux=None,  # direct connection — no MUX in the calibration gate
            active_channel=channel_id,
            start_voltage=start_v,
            stop_voltage=stop_v,
            step_size=step_v,
            device_area=area,
            incident_power=100.0,
            compliance_current=0.1,
            pre_sweep_delay=0.5,
            simulation=False,
            channel1=(channel_id == 1), channel2=(channel_id == 2),
            channel3=(channel_id == 3), channel4=(channel_id == 4),
            channel5=(channel_id == 5), channel6=(channel_id == 6),
            user_name=self.username
        )

        # Store procedure for report writing
        self._calibration_procedure = procedure

        results = Results(procedure, file_path)
        self._current_results = results

        curve = self.plot_widget.new_curve(results, color=pg.mkColor('#0984e3'), width=2)

        browser_item = SignalBrowserItem(
            results, pg.intColor(0),
            progress_callback=self._update_progress_bar
        )
        browser_item.setText(0, "Data")

        experiment = Experiment(results, [curve], browser_item)
        try:
            self.manager.queue(experiment)
        except Exception as e:
            # A failed queue (e.g. a missing parameter) leaves pymeasure's
            # Manager half-started — is_running() True but no Worker — so
            # every later abort() raises on the None Worker. Rebuild a clean
            # Manager directly (we must not abort() the wedged one) so the
            # button recovers instead of staying stuck.
            logger.error(f"Failed to start calibration: {e}")
            self.manager = Manager(
                [self.plot_widget],
                self._dummy_browser_widget.browser,
                log_level=logging.INFO,
                parent=self
            )
            self.manager.finished.connect(self._on_sweep_finished)
            self.manager.abort_returned.connect(self._on_abort_complete)
            self.run_button.setText("RESTART")
            self._check_readiness()

    def _write_calibration_report(self, filepath, procedure, metrics, result_text):
        """Write a permanent calibration report CSV."""
        try:
            params = {
                'Start Voltage': f"{procedure.start_voltage} V",
                'Stop Voltage': f"{procedure.stop_voltage} V",
                'Step Size': f"{procedure.step_size} V",
                'Sweep Rate': f"{procedure.sweep_rate} V/s",
                'Device Area': f"{procedure.device_area} cm²",
                'Target Isc': f"{self.spin_target_isc.value() * 1000:.1f} mA",
                'Tolerance': f"{self.spin_tolerance.value()} %",
                'Compliance Current': f"{procedure.compliance_current} A",
                'User Name': procedure.user_name,
            }
            # Provenance: record how many raw points were suppressed because
            # they sat at the compliance clamp (not measurements of the cell).
            n_clipped = getattr(procedure, 'compliance_clipped_points', 0)
            if n_clipped:
                params['Compliance-Clipped Points (excluded)'] = str(n_clipped)

            analysis = {
                'Isc': (metrics.get('Isc', 0), 'A'),
                'Jsc': (metrics.get('Jsc', 0), 'mA/cm²'),
                'Voc': (metrics.get('Voc', 0), 'mV'),
                'FF': (metrics.get('FF', 0), '%'),
                'EFF': (metrics.get('EFF', 0), '%'),
                'Result': (result_text, ''),
            }

            # Only REAL measurements go into the report — points clamped at
            # the compliance limit are the instrument's ceiling, not the
            # cell, and are excluded from data, plot, and metrics alike.
            v_clean, i_clean, _ = procedure.filter_compliance_points(
                procedure._voltages, procedure._currents,
                float(procedure.compliance_current)
            )
            data = {
                'Voltage (V)': list(v_clean),
                'Current (A)': list(i_clean),
            }

            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                f.write("Parameter,Value,Unit\n")
                for key, value in params.items():
                    if isinstance(value, str):
                        parts = value.rsplit(' ', 1)
                        if len(parts) == 2 and parts[1] in ('V', 'mV', 'A', 'mA', 'cm²', '%', 's', 'V/s'):
                            f.write(f"{key},{parts[0]},{parts[1]}\n")
                        else:
                            f.write(f"{key},{value},\n")
                    else:
                        f.write(f"{key},{value},\n")
                f.write("\n")

                f.write("[[ ANALYSIS SUMMARY ]]\n")
                f.write("Parameter,Value,Unit\n")
                for key, (value, unit) in analysis.items():
                    if isinstance(value, float):
                        formatted = f"{value:.6f}"
                    else:
                        formatted = str(value)
                    f.write(f"{key},{formatted},{unit}\n")
                f.write("\n")

                f.write("[[ MEASUREMENT DATA ]]\n")
                f.write("channel,1,1\n")
                f.write("direction,Forward,Forward\n")
                f.write("value,V,J\n")

                for i in range(len(data['Voltage (V)'])):
                    f.write(f",{data['Voltage (V)'][i]},{data['Current (A)'][i]}\n")

            logger.info(f"Calibration report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to write calibration report: {e}")

    def _update_progress_bar(self, value):
        self.progress_bar.setValue(int(value))

    # -------------------------------------------------------------------------
    # Calibration Results
    # -------------------------------------------------------------------------
    def _on_sweep_finished(self, experiment):
        self.progress_bar.setValue(100)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #e0e0e0; height: 4px; border-radius: 2px; }
            QProgressBar::chunk { background-color: #053a46; border-radius: 2px; }
        """)

        if self.manager.is_running():
            return

        self.run_button.setText("RESTART")
        self._check_readiness()

        # Warn if the sweep started inside the compliance clamp — those
        # points are the instrument's current ceiling, not the cell (a real
        # report showed J pinned at 0.0999... for the first 4 points of a
        # 0.7 V start with 0.1 A compliance). They are tagged COMPLIANCE in
        # the raw data and excluded from the analysed metrics.
        n_clipped = getattr(experiment.procedure, 'compliance_clipped_points', 0)
        if n_clipped:
            QtWidgets.QMessageBox.warning(
                self, "Compliance Limit Reached",
                f"{n_clipped} measurement point(s) were clamped at the "
                f"compliance limit "
                f"({float(experiment.procedure.compliance_current)} A) at "
                "the start of the sweep.\n\n"
                "These points are NOT valid measurements of the cell and "
                "were excluded from the analysis.\n\n"
                "Fix: lower the Start Voltage (the reference cell's Voc is "
                "well below it) or raise the Compliance Current."
            )

        try:
            results = experiment.procedure.analysis_results
            channel = int(experiment.procedure.active_channel)

            if not results or channel not in results:
                self._set_fail_state("No data", 0, 0, 0)
                return

            raw = results[channel]
            if isinstance(raw, dict) and any(isinstance(v, dict) for v in raw.values()):
                metrics = next(iter(raw.values()))
            else:
                metrics = raw

            measured_isc = metrics.get("Isc", 0.0)

            area_val = self.spin_area.value()
            measured_jsc = (measured_isc / area_val) * 1000 if area_val > 0 else 0

            target_isc = self.spin_target_isc.value()
            tolerance_pct = self.spin_tolerance.value()

            diff = abs(measured_isc - target_isc)
            limit = target_isc * (tolerance_pct / 100.0)

            self.lbl_measured_isc.setText(f"{measured_isc * 1000:.2f} mA")
            self.lbl_measured_jsc.setText(f"{measured_jsc:.2f} mA/cm²")
            self._update_measurement_card(measured_isc, measured_jsc)

            # Write permanent calibration report if we have a path
            if hasattr(self, '_calibration_file_path') and self._calibration_file_path:
                result_text = "PASS" if diff <= limit else "FAIL"
                self._write_calibration_report(
                    self._calibration_file_path,
                    experiment.procedure,
                    metrics,
                    result_text
                )

            if diff <= limit:
                self._set_pass_state()
            else:
                self._set_fail_state("Tolerance", measured_isc, target_isc, measured_jsc)

        except Exception as e:
            logger.error(f"Calibration evaluation error: {e}")
            self._set_fail_state("Error", 0, 0, 0)

    def _on_abort_complete(self):
        self.lbl_status_text.setText("ABORTED")
        self.lbl_status_text.setStyleSheet(
            "background: #ffdad6; color: #93000a; padding: 8px; border-radius: 8px;"
        )
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #e0e0e0; height: 4px; border-radius: 2px; }
            QProgressBar::chunk { background-color: #ef4444; border-radius: 2px; }
        """)
        self.run_button.setText("RESTART")
        self._check_readiness()

    def _set_pass_state(self):
        self.lbl_status_text.setText("PASS")
        self.lbl_status_text.setStyleSheet(
            "background: #d1fae5; color: #065f46; padding: 8px; border-radius: 8px;"
        )
        self.proceed_button.setEnabled(True)
        self.proceed_button.setText("PROCEED TO MAIN ➡")
        self.run_button.setText("RE-CALIBRATE")
        self.run_button.setStyleSheet("")
        self.run_button.setObjectName("RunButton")
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)

    def _set_fail_state(self, reason, measured_val, target_val, measured_jsc):
        self.lbl_status_text.setText("FAIL")
        self.lbl_status_text.setStyleSheet(
            "background: #ffdad6; color: #93000a; padding: 8px; border-radius: 8px;"
        )
        self.proceed_button.setEnabled(False)

        if target_val != 0:
            pct_diff = ((measured_val - target_val) / target_val) * 100
        else:
            pct_diff = 0.0

        if pct_diff > 0:
            hint = "Current is too HIGH → Move lamp UP."
        else:
            hint = "Current is too LOW → Move lamp DOWN."

        message = (
            f"<b>Measurement out of tolerance!</b><br><br>"
            f"Measured Isc: <b>{measured_val * 1000:.2f} mA</b><br>"
            f"Measured Jsc: <b>{measured_jsc:.2f} mA/cm²</b><br>"
            f"Target Isc: <b>{target_val * 1000:.2f} mA</b><br>"
            f"Diff: <b>{pct_diff:+.1f}%</b><br><br>"
            f"<i>Hint: {hint}</i>"
        )
        QtWidgets.QMessageBox.warning(self, "Calibration Failed", message)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
    def launch_checklist_dialog(self):
        dialog = CalibrationChecklistDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.checklist_confirmed = True
            # The checklist instructs the user to power the hardware ON —
            # so the initial connection attempt (made at window construction,
            # BEFORE the checklist) has typically already failed. Retry now
            # that the user has confirmed everything is powered; this also
            # refreshes the status lights and the run-button pill.
            self._connect_hardware()
        else:
            self.checklist_confirmed = False
        self._check_readiness()