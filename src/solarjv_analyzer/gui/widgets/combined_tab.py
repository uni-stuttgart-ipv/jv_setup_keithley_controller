"""
Combined JV + SPO Tab

Provides a single-tab workflow where the user configures JV sweep parameters
and SPO hold settings together, then clicks one button to run everything:
  1. JV sweeps on selected channels
  2. Auto-select the best channel by Efficiency (Pmpp as tie-breaker)
  3. Auto-start SPO on that channel using its Vmpp as the hold voltage

Layout order (top to bottom):
  JV Sweep Parameters (Start V, Stop V, Step, Sweep Rate, Compliance, Area, Arch)
  SPO Settings (Hold Duration, Sampling Interval, Pre-conditioning)
  Channel Selection (moved here from inside ParameterTab)
"""

from PyQt5 import QtWidgets, QtCore

from .parameter_tab import ParameterTab


class CombinedTab(QtWidgets.QWidget):
    """Single-tab widget combining JV sweep parameters + SPO settings.

    Embeds the existing ParameterTab for JV sweep / architecture configuration,
    then adds SPO Settings between the sweep params and channel selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # -------------------------------------------------------------------
    # UI Construction
    # -------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ---- JV Sweep Parameters (embedded, reused wholesale) ----------
        self.jv_params = ParameterTab()

        # Extract Channel Selection and Notes from ParameterTab's internal
        # QFormLayout so we can reorder them:
        #   sweep params → SPO Settings → Channel Selection → Notes
        form = self.jv_params.layout()  # QFormLayout
        channel_card = None
        notes_widget = None
        channel_row = -1
        notes_row = -1
        for i in range(form.rowCount()):
            field_item = form.itemAt(i, QtWidgets.QFormLayout.FieldRole)
            if field_item is None:
                continue
            w = field_item.widget()
            if w is None:
                continue
            if isinstance(w, QtWidgets.QGroupBox) and w.title() == "Channel Selection":
                channel_card = w
                channel_row = i
            elif isinstance(w, QtWidgets.QWidget) and notes_widget is None:
                # The Notes field contains a QTextEdit — find it by child widget
                if w.findChild(QtWidgets.QTextEdit, "notes_field") or \
                   w.findChild(QtWidgets.QTextEdit):
                    notes_widget = w
                    notes_row = i
        # Also try finding notes by label text
        if notes_widget is None:
            for i in range(form.rowCount()):
                label_item = form.itemAt(i, QtWidgets.QFormLayout.LabelRole)
                if label_item is not None and label_item.widget() is not None:
                    lbl = label_item.widget()
                    if isinstance(lbl, QtWidgets.QLabel) and lbl.text() == "Notes:":
                        field_item = form.itemAt(i, QtWidgets.QFormLayout.FieldRole)
                        if field_item is not None:
                            notes_widget = field_item.widget()
                            notes_row = i
                            break

        # Extract rows from bottom to top so indices don't shift
        for row, _name in sorted(
            [(channel_row, "channel"), (notes_row, "notes")],
            key=lambda x: x[0], reverse=True
        ):
            if row < 0:
                continue
            result = form.takeRow(row)
            # Delete empty label spacers
            if result.labelItem is not None:
                lbl_w = result.labelItem.widget()
                if lbl_w is not None:
                    lbl_w.deleteLater()

        layout.addWidget(self.jv_params)

        # ---- SPO Settings ----------------------------------------------
        spo_group = QtWidgets.QGroupBox("SPO Settings")
        spo_layout = QtWidgets.QFormLayout(spo_group)
        spo_layout.setVerticalSpacing(10)

        self.hold_duration = QtWidgets.QLineEdit("300")
        self.hold_duration.setToolTip("Total hold time in seconds")
        spo_layout.addRow("Hold Duration (s):", self.hold_duration)

        self.sampling_interval = QtWidgets.QLineEdit("1.0")
        self.sampling_interval.setToolTip("Time between samples in seconds")
        spo_layout.addRow("Sampling Interval (s):", self.sampling_interval)

        self.preconditioning = QtWidgets.QLineEdit("5.0")
        self.preconditioning.setToolTip(
            "Stabilisation time before logging begins"
        )
        spo_layout.addRow("Pre-conditioning (s):", self.preconditioning)

        layout.addWidget(spo_group)

        # ---- Channel Selection (reparented from ParameterTab) -----------
        if channel_card is not None:
            layout.addWidget(channel_card)

        # ---- Notes (reparented from ParameterTab, already has heading) -----
        if notes_widget is not None:
            layout.addWidget(notes_widget)

        layout.addStretch()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def get_jv_parameters(self) -> dict:
        """Return the full JV sweep parameter dict from the embedded tab."""
        return self.jv_params.get_parameters()

    def get_spo_parameters(self) -> dict:
        """Return SPO settings as a dict ready for SpoProcedure."""
        return {
            "hold_duration": float(self.hold_duration.text() or "300"),
            "sampling_interval": float(self.sampling_interval.text() or "1.0"),
            "preconditioning_time": float(self.preconditioning.text() or "5.0"),
        }

    def get_selected_channels(self) -> list:
        """Return the list of currently toggled-on channel numbers (1-indexed)."""
        return self.jv_params.get_selected_channels()

    def set_config_enabled(self, enabled: bool):
        """Enable or disable all inputs (locked during a combined run)."""
        self.jv_params.setEnabled(enabled)
        self.hold_duration.setEnabled(enabled)
        self.sampling_interval.setEnabled(enabled)
        self.preconditioning.setEnabled(enabled)

    @property
    def sweep_rate(self):
        """Expose the sweep_rate QLineEdit for NPLC preview connections."""
        return self.jv_params.sweep_rate

    @property
    def sweep_rate_unit(self):
        return self.jv_params.sweep_rate_unit

    @property
    def start_voltage(self):
        return self.jv_params.start_voltage

    @property
    def start_unit(self):
        return self.jv_params.start_unit

    @property
    def stop_voltage(self):
        return self.jv_params.stop_voltage

    @property
    def stop_unit(self):
        return self.jv_params.stop_unit

    @property
    def step_size(self):
        return self.jv_params.step_size

    @property
    def step_unit(self):
        return self.jv_params.step_unit
