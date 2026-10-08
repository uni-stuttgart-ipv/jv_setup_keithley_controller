"""
Application Controller for JV Analyzer

Manages the experiment lifecycle, file handling, and coordination between
the PyMeasure Manager and the GUI components.
"""

import logging
import math
import os
import io
import re
import tempfile
from datetime import datetime
from PyQt5 import QtCore, QtWidgets

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.experiment import Results

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.config import TIMESTAMP_FORMAT

# SPO is a self-contained, optional module: if the spo/ package is removed,
# the JV application must still compile and run (SPO features simply no-op).
try:
    from solarjv_analyzer.spo.spo_procedure import SpoProcedure, SpoWorker
    from solarjv_analyzer.spo.spo_analysis import compute_spo_metrics, SPO_METRICS_UNITS
    SPO_AVAILABLE = True
except ImportError:
    SpoProcedure = SpoWorker = compute_spo_metrics = None
    SPO_METRICS_UNITS = []
    SPO_AVAILABLE = False

logger = logging.getLogger(__name__)



# The file panel asks for a "Filename Prefix", so almost nobody types an
# extension — and `os.path.splitext("Test")` returns `("Test", "")`. That empty
# extension was carried straight through to the merged single-file output,
# which landed on disk with no extension at all and opened as "File" in
# Windows rather than as a spreadsheet. Per-channel files escaped it only
# because the forward+reverse branch hardcodes ".csv" downstream.
_EXTENSION_RE = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,4}$")


def _split_output_name(filename: str, default_ext: str = ".csv") -> tuple:
    """Split a user-typed output name into (base, extension).

    Defaults to `.csv` when no extension was given, and treats a trailing dot
    group that is not extension-shaped as part of the name — so "Sample_1.5cm"
    becomes ("Sample_1.5cm", ".csv") rather than ("Sample_1", ".5cm").
    """
    name = (filename or "").strip()
    match = _EXTENSION_RE.search(name)
    if match:
        return name[:match.start()], match.group(0)
    return name, default_ext


class AppController:
    """
    Coordinates experiment execution and data management.

    Responsibilities:
    - Queue experiments with user-specified parameters
    - Manage file output (single or multi-file modes)
    - Handle experiment lifecycle (start, abort, resume)
    - Process and format measurement data
    """

    def __init__(self, view):
        """
        Initialize the controller with a reference to the main window.

        Args:
            view: The main application window (JVAnalyzerWindow instance)
        """
        self.view = view
        self.finished_experiment_count = 0
        self.is_busy = False

        # File management state
        self.experiment_files = {}      # Maps channel number to file path
        self.is_single_file_mode = False
        self.processed_files = set()    # Tracks formatted files to avoid duplicates

        # Initialize PyMeasure Manager with display widgets
        self.manager = Manager(
            [self.view.plot_widget, self.view.analysis_panel],
            self.view.browser_widget.browser,
            log_level=logging.INFO,
            parent=self.view
        )
        self._connect_manager_signals()

        # Channel color mapping for consistent colors across forward and reverse curves
        # Single source of truth — see theme.tokens.CHANNEL_COLORS. The pens
        # here and the Channel Analysis rows must agree, so neither side
        # keeps its own copy of the palette.
        from solarjv_analyzer.gui.theme import tokens as _tokens
        self.CHANNEL_COLORS = {
            ch: pg.mkColor(hex_colour)
            for ch, hex_colour in _tokens.CHANNEL_COLORS.items()
        }

        self.view.abort_button.setEnabled(False)

        # ---------------------------------------------------------------
        # SPO (Set-Point Operation) state
        # ---------------------------------------------------------------
        self.spo_widget = self.view.spo_widget
        self.spo_running = False
        self._spo_worker = None
        self._spo_procedure = None
        self._spo_times = []
        self._spo_currents = []
        self._spo_voltages = []

        # ---------------------------------------------------------------
        # Combined JV + SPO state
        # ---------------------------------------------------------------
        self._combined_mode = False
        self._combined_jv_finished = False
        self._combined_best_channel = None
        self._combined_best_vmpp = 0.0
        self._combined_spo_params = {}
        self._combined_spo_powers = []

        # Every unlock in this class hangs off a finish handler; a handler
        # that never runs cannot unlock anything. See _idle_watchdog().
        self._stuck_ticks = 0
        self._idle_timer = QtCore.QTimer(self.view)
        self._idle_timer.setInterval(2000)
        self._idle_timer.timeout.connect(self._idle_watchdog)
        self._idle_timer.start()

    def _idle_watchdog(self):
        """Unlock the window when nothing is actually running any more.

        Every exit path re-enables the controls correctly — `_return_to_idle`,
        `_on_combined_spo_finished`, `on_abort_returned`, `on_failed` and both
        exits of `_start_spo_phase` all call `_set_combined_config_enabled(True)`,
        and a lock/unlock cycle restores the architecture toggle exactly as it
        should. The failure is that none of them RUN: pymeasure raises inside
        `Manager._finish()` *before* it emits `finished` (one empty results
        table is enough — see the 2026-10-01 logs), so the sweeps end, the
        queue empties, and the toggle, the voltages and Run stay greyed with
        nothing left to re-enable them.

        Rather than chase every way a handler can die, this asks the ground
        truth — is anything actually running? — and recovers. The state must
        look idle for THREE consecutive ticks (~6 s) before it acts: a queue
        being built briefly shows no running experiment, and unlocking inside
        that window would fight the run that is just starting.

        A paused queue (`has_next()`) is legitimately not idle, so it is left
        alone — Resume and Clear own that state.

        UI recovery only. If a finish handler died it may also have skipped
        the file merge, which this deliberately does not retry; the warning
        below is the signal to go and look.
        """
        try:
            if (self.spo_running or self.manager.is_running()
                    or self.manager.experiments.has_next()):
                self._stuck_ticks = 0
                return

            worker = getattr(self, "_spo_worker", None)
            if worker is not None and worker.isRunning():
                self._stuck_ticks = 0
                return

            try:
                unlocked = self.view.combined_tab.jv_params.isEnabled()
            except AttributeError:
                unlocked = True
            if not self.is_busy and unlocked:
                self._stuck_ticks = 0
                return

            self._stuck_ticks += 1
            if self._stuck_ticks < 3:
                return
            self._stuck_ticks = 0
            logger.warning(
                "Nothing is running but the window was still locked - a finish "
                "handler did not complete. Returning to idle; check that the "
                "last run's report was written."
            )
            self._return_to_idle()
        except Exception as exc:                       # noqa: BLE001
            logger.debug(f"Idle watchdog skipped: {exc}")

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_manager_signals(self):
        """Connect PyMeasure Manager signals to controller handlers."""
        self.manager.abort_returned.connect(self.on_abort_returned)
        self.manager.queued.connect(self.on_queued)
        self.manager.running.connect(self.on_running)
        self.manager.finished.connect(self.on_finished)
        # pymeasure's `_failed()` does NOT advance the queue the way
        # `_finish()` does, so without this the application is simply
        # never told a sweep died — see on_failed().
        self.manager.failed.connect(self.on_failed)
        self.manager.finished.connect(self.update_analysis_panel)

    # -------------------------------------------------------------------------
    # Experiment Queueing
    # -------------------------------------------------------------------------

    def queue_experiment(self):
        """
        Collect parameters from UI and queue experiments for selected channels.

        Handles both single-file (merged) and multi-file output modes.
        Clears the plots and the analysis table so the screen shows only
        this queue's channels.
        """
        if self.is_busy:
            self._refuse_busy("a new queue")
            return
        self.is_busy = True

        # Collect parameters from UI tabs
        params_dict = self.view.params_tab.get_parameters()
        analysis_dict = self.view.active_analysis_tab.get_parameters()
        instr_dict = self.view.active_instr_tab.get_parameters()

        procedure_params = {
            'user_name': self.view.username,
            **params_dict,
            **instr_dict,
            **analysis_dict,
        }

        # Keep a snapshot of the notes – they are not part of the procedure itself,
        # but we need them when writing the final report.
        self.current_notes_text = procedure_params.pop('notes_text', '')
        self.current_save_notes = procedure_params.pop('save_notes', False)
        file_params = self.view.file_panel.get_parameters()
        selected_channels = self.view.params_tab.get_selected_channels()

        if not selected_channels:
            logger.warning("No channels selected.")
            self.is_busy = False
            return

        # Reset state for this run
        self.experiment_files = {}
        self.processed_files = set()
        self.is_single_file_mode = file_params['single_file']

        # Connect hardware
        sim_mode = False  
        try:
            self.view.instrument_manager.connect_keithley(simulation=sim_mode)
            self.view.instrument_manager.connect_mux(simulation=sim_mode)
        except Exception as e:
            logger.error(f"Hardware connection failed: {e}")
            self.view.update_instrument_lights()
            self.is_busy = False
            return

        self.view.update_instrument_lights()

        if not self._keithley_usable("JV queue"):
            self.is_busy = False
            return

        # The run is certain to start, so the previous one now comes off the
        # screen. Purging earlier — before the hardware checks above — would
        # have discarded good results every time a run failed to start.
        self._reset_view_for_new_run(selected_channels)

        # Generate file paths with timestamp
        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        base, ext = _split_output_name(file_params['filename'])
        directory = file_params['directory']
        # Remember what the operator typed. `_process_multi_files()` writes the
        # FINAL per-channel reports and used to name them "Output_<ts>_chN.csv",
        # discarding the name entirely — so every run from every user produced
        # identically-named files and nobody could tell which experiment a file
        # belonged to without opening it.
        self._run_base_name = base
        self._run_ext = ext

        if self.is_single_file_mode:
            self._queue_single_file_experiment(
                directory, base, ext, filename_timestamp,
                selected_channels, procedure_params, sim_mode
            )
        else:
            self._queue_multi_file_experiment(
                directory, base, ext, filename_timestamp,
                selected_channels, procedure_params, sim_mode
            )

    def _reset_view_for_new_run(self, channels):
        """Wipe every trace of the previous run before queueing a new one.

        A run used to MERGE into whatever was already on screen. The analysis
        table was rebuilt for the UNION of the old and new channels and the
        old metrics copied back in, while the previous experiments — and so
        their curves — were left in the manager untouched. Measuring ch1-3
        and then ch4-6 showed all six, which was the intent; but the ordinary
        case was far more confusing. Pressing Run left the last run's curves
        on the plot and the last run's numbers in the table until each new
        channel happened to overwrite its own row, so for the length of a
        queue the screen showed two experiments at once with nothing to say
        which row belonged to which.

        A new queue now starts from a blank screen. Bringing earlier results
        back for comparison is what the Open button is for — and `load_files()`
        performs this same purge before loading them.

        Deliberately does NOT touch run state (`is_busy`, the run controls):
        this runs mid-start, with the run already claimed by the caller.
        """
        # Drops the experiments, their browser rows AND their plot curves.
        self.manager.clear()
        # Belt and braces: `clear_experiments()` re-arms after clearing for a
        # reason (see `_rearm_manager`), and a run that starts from a
        # dis-armed manager queues its sweeps and then never runs them.
        self._rearm_manager()
        self.view.browser_widget.browser.clear()
        self.finished_experiment_count = 0

        # Rebuild the table for THIS run's channels only, cells empty.
        ordered = sorted(channels)
        self.view.analysis_panel.reset_channels(
            ordered, JVProcedure.ANALYSIS_LABELS_UNITS)
        self.view.analysis_panel.clear_all()
        self._sync_channel_indicators(ordered)

        self._reset_combined_spo_view()

    def _reset_combined_spo_view(self):
        """Blank the JV+SPO tab's SPO trace and its three readouts.

        Clearing the J-V half of that tab while leaving the previous run's SPO
        curve and drift figures below it would be worse than clearing nothing:
        the two halves would be showing different experiments.
        """
        self._spo_times = []
        self._spo_currents = []
        self._spo_voltages = []
        self._combined_spo_powers = []
        try:
            self.view.combined_spo_curve.setData([], [])
            self.view.combined_spo_mean.setText("— mW")
            self.view.combined_spo_drift.setText("— %")
            self.view.combined_spo_elapsed.setText("— s")
        except AttributeError:
            # The combined tab's widgets are built lazily; nothing to blank.
            pass


    def _queue_single_file_experiment(self, directory, base, ext, timestamp,
                                  channels, params, sim_mode):
        """
        Queue experiments for single-file output mode.
        In single sweep mode, only forward sweeps are queued.
        """
        channel_list_str = "_".join(map(str, channels))
        self.merged_file_path = os.path.join(
            directory, f"{base}_{timestamp}_ch{channel_list_str}{ext}"
        )
        self.merged_data_written = False

        merged_filename = os.path.basename(self.merged_file_path)
        logger.info(f"Single file mode: merging to {merged_filename}")

        for channel_num in channels:
            base_params = params.copy()
            for i in range(1, 7):
                base_params[f'channel{i}'] = (i == channel_num)

            # ---------- Forward Sweep ----------
            forward_params = base_params.copy()
            forward_params['single_sweep_mode'] = True
            forward_params['sweep_direction'] = 'Forward'

            forward_temp_path = os.path.join(
                directory, f"{base}_{timestamp}_ch{channel_num}_forward_temp{ext}"
            )
            forward_key = f"{channel_num}_forward"
            self.experiment_files[forward_key] = forward_temp_path

            proc_forward = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim_mode,
                active_channel=channel_num,
                check_errors_between_points=False,
                **forward_params
            )

            results_forward = Results(proc_forward, forward_temp_path)
            proc_forward.results = results_forward

            display_name_forward = f"Ch {channel_num} (Fwd) - {merged_filename}"
            experiment_forward = self._create_experiment(
                results_forward, display_name_forward,
                channel=channel_num, is_reverse=False
            )
            self.manager.queue(experiment_forward)
            logger.info(f"Queued Channel {channel_num} (Forward)")

            # ---------- Reverse Sweep (only in dual‑sweep mode) ----------
            if not params.get('single_sweep_mode', False):
                reverse_params = base_params.copy()
                reverse_params['single_sweep_mode'] = True
                reverse_params['sweep_direction'] = 'Reverse'
                reverse_params['start_voltage'] = params.get('stop_voltage', -0.2)
                reverse_params['stop_voltage'] = params.get('start_voltage', 1.2)

                reverse_temp_path = os.path.join(
                    directory, f"{base}_{timestamp}_ch{channel_num}_reverse_temp{ext}"
                )
                reverse_key = f"{channel_num}_reverse"
                self.experiment_files[reverse_key] = reverse_temp_path

                proc_reverse = JVProcedure(
                    instrument=self.view.instrument_manager.keithley,
                    mux=self.view.instrument_manager.mux,
                    manager=self.view.instrument_manager,
                    simulation=sim_mode,
                    active_channel=channel_num,
                    check_errors_between_points=False,
                    **reverse_params
                )

                results_reverse = Results(proc_reverse, reverse_temp_path)
                proc_reverse.results = results_reverse

                display_name_reverse = f"Ch {channel_num} (Rev) - {merged_filename}"
                experiment_reverse = self._create_experiment(
                    results_reverse, display_name_reverse,
                    channel=channel_num, is_reverse=True
                )
                self.manager.queue(experiment_reverse)
                logger.info(f"Queued Channel {channel_num} (Reverse)")

    def _queue_multi_file_experiment(self, directory, base, ext, timestamp,
                                 channels, params, sim_mode):
        """
        Queue experiments for multi-file output mode.
        In single sweep mode, only forward sweeps are queued.
        """
        for channel_num in channels:
            # Set channel-specific parameters
            base_params = params.copy()
            for i in range(1, 7):
                base_params[f'channel{i}'] = (i == channel_num)

            # ---------- Forward Sweep (always queued) ----------
            forward_params = base_params.copy()
            forward_params['single_sweep_mode'] = True
            forward_params['sweep_direction'] = 'Forward'

            forward_path = os.path.join(
                directory, f"{base}_{timestamp}_ch{channel_num}_forward{ext}"
            )
            self.experiment_files[f"{channel_num}_forward"] = forward_path

            proc_forward = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim_mode,
                active_channel=channel_num,
                check_errors_between_points=False,
                **forward_params
            )

            results_forward = Results(proc_forward, forward_path)
            proc_forward.results = results_forward

            experiment_forward = self._create_experiment(
                results_forward, channel=channel_num, is_reverse=False
            )
            self.manager.queue(experiment_forward)
            logger.info(f"Queued Channel {channel_num} (Forward)")

            # ---------- Reverse Sweep (only in dual‑sweep mode) ----------
            if not params.get('single_sweep_mode', False):
                reverse_params = base_params.copy()
                reverse_params['single_sweep_mode'] = True
                reverse_params['sweep_direction'] = 'Reverse'
                reverse_params['start_voltage'] = params.get('stop_voltage', -0.2)
                reverse_params['stop_voltage'] = params.get('start_voltage', 1.2)

                reverse_path = os.path.join(
                    directory, f"{base}_{timestamp}_ch{channel_num}_reverse{ext}"
                )
                self.experiment_files[f"{channel_num}_reverse"] = reverse_path

                proc_reverse = JVProcedure(
                    instrument=self.view.instrument_manager.keithley,
                    mux=self.view.instrument_manager.mux,
                    manager=self.view.instrument_manager,
                    simulation=sim_mode,
                    active_channel=channel_num,
                    check_errors_between_points=False,
                    **reverse_params
                )

                results_reverse = Results(proc_reverse, reverse_path)
                proc_reverse.results = results_reverse

                experiment_reverse = self._create_experiment(
                    results_reverse, channel=channel_num, is_reverse=True
                )
                self.manager.queue(experiment_reverse)
                logger.info(f"Queued Channel {channel_num} (Reverse)")

    def _create_experiment(self, results: Results, display_filename: str = None, 
                       channel: int = None, is_reverse: bool = False) -> Experiment:
        """
        Create an Experiment object with associated plot curve and browser item.

        Args:
            results: PyMeasure Results object
            display_filename: Optional custom filename for browser display
            channel: Channel number (for consistent color mapping)
            is_reverse: True for reverse sweep (dashed line), False for forward (solid line)

        Returns:
            Experiment: Configured experiment ready for queueing
        """
        browser = self.view.browser_widget.browser
        
        # Use channel-specific color if provided, otherwise default
        if channel and channel in self.CHANNEL_COLORS:
            base_color = self.CHANNEL_COLORS[channel]
        else:
            base_color = pg.intColor(browser.topLevelItemCount() % 8)
        
        if is_reverse:
            # Reverse curve: dashed line, semi-transparent
            color = pg.mkColor(base_color)
            color.setAlpha(180)
            pen = pg.mkPen(color=color, width=2, style=QtCore.Qt.DashLine)
            curve = self.view.plot_widget.new_curve(results, pen=pen)
        else:
            # Forward curve: solid line
            pen = pg.mkPen(color=base_color, width=2, style=QtCore.Qt.SolidLine)
            curve = self.view.plot_widget.new_curve(results, pen=pen)

        browser_item = BrowserItem(results, base_color)
        if display_filename:
            browser_item.setText(1, display_filename)

        return Experiment(results, [curve], browser_item)

    # -------------------------------------------------------------------------
    # File Loading
    # -------------------------------------------------------------------------

    def _load_spo_report(self, filename: str) -> bool:
        """Show `filename` in the SPO view if it is an SPO report.

        Returns True when the file was handled, so the J-V loader is skipped.
        """
        # Whichever SPO view the operator is already in is the one that
        # should show the file. Opening from the JV+SPO tab must not throw
        # them over to Advanced.
        if getattr(self.view, "in_combined_view", lambda: False)():
            if self.view.load_spo_report_into_combined(filename):
                return True
            return False

        widget = getattr(self.view, "spo_widget", None)
        if widget is None or not hasattr(widget, "load_report"):
            return False
        if not widget.load_report(filename):
            return False
        # Bring the operator to what they just opened. The SPO display lives
        # inside ADVANCED mode with the SPO toggle selected, so showing it
        # takes both steps: `_show_spo_mode()` alone only un-hides a widget
        # on a page the main display stack is not currently showing, which
        # looked exactly like "the file loaded but no plot appeared".
        try:
            if hasattr(self.view, "_on_mode_button_click"):
                self.view._on_mode_button_click(1)          # Advanced
            if hasattr(self.view, "spo_mode_button"):
                self.view.spo_mode_button.setChecked(True)
            if hasattr(self.view, "_show_spo_mode"):
                self.view._show_spo_mode()
            # Make sure the SPO area is on its Plot page, not its Log page.
            for attr, index in (("spo_graph_tab_bar", 0), ("spo_graph_stack", 0)):
                target = getattr(self.view, attr, None)
                if target is not None:
                    target.setCurrentIndex(index)
        except Exception as exc:
            logger.warning(f"Could not switch to the SPO view: {exc}")
        return True

    def load_files(self, filenames: list):
        """
        Load previously saved measurement files into the browser and plot.

        Args:
            filenames: List of file paths to load
        """
        logger.info(f"Loading {len(filenames)} file(s)")

        # ---- purge all stale data before loading new files ----
        self.clear_experiments()
        self.view.analysis_panel.clear_all()
        self.view.browser_widget.browser.clear()

        all_channels = []
        newly_loaded_items = []

        for filename in filenames:
            try:
                # An SPO report is power-versus-time, not current-versus-
                # voltage, so it cannot become a curve in the J-V plot. Hand
                # it to the SPO view instead, which owns that axis pair and
                # the matching metrics card. Detection is by content, not by
                # filename, because both kinds are plain .csv.
                if self._load_spo_report(filename):
                    continue
                experiments, _, channels = self._parse_and_load_file(filename)
                newly_loaded_items.extend(experiments)
                all_channels.extend(channels)
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                from PyQt5 import QtWidgets
                QtWidgets.QMessageBox.warning(
                    self.view, "Load Error",
                    f"Failed to load {os.path.basename(filename)}\n{e}"
                )

        # Update analysis panel with all loaded channels
        active_channels = set()
        browser = self.view.browser_widget.browser
        root = browser.invisibleRootItem()

        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'active_channel'):
                try:
                    active_channels.add(int(exp.procedure.active_channel))
                except (ValueError, TypeError):
                    pass

        # ===== 1. Determine sweep mode (single vs dual) BEFORE reset =====
        has_reverse = False
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'sweep_direction') and exp.procedure.sweep_direction == 'Reverse':
                has_reverse = True
                break
        self.view.analysis_panel.set_single_sweep_mode(not has_reverse)
        # =================================================================

        if active_channels:
            self.view.analysis_panel.reset_channels(
                sorted(active_channels), JVProcedure.ANALYSIS_LABELS_UNITS
            )
            self._sync_channel_indicators(sorted(active_channels))

        # ===== 2. Restore analysis data (now handles nested direction dicts) =====
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'analysis_results'):
                results = exp.procedure.analysis_results
                if results:
                    for channel, metrics in results.items():
                        if isinstance(metrics, dict):
                            # Nested: e.g., {1: {"Forward": {...}, "Reverse": {...}}}
                            for direction, dir_metrics in metrics.items():
                                self.view.analysis_panel.analysis({
                                    'Channel': channel,
                                    'Direction': direction,
                                    **dir_metrics
                                })
                        else:
                            # Legacy flat dict
                            self.view.analysis_panel.analysis({
                                'Channel': channel,
                                'Direction': 'Forward',
                                **metrics
                            })
        # ==========================================================================

        # Enable Show/Hide/Clear after loading
        self.view.browser_widget.show_button.setEnabled(True)
        self.view.browser_widget.hide_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(True)

        # Re-strip BrowserWidget internal margins (layout may have been
        # re-created during load, undoing the zero-margin fix from init).
        if self.view.browser_widget.layout():
            self.view.browser_widget.layout().setContentsMargins(0, 0, 0, 0)

        # Select the first loaded experiment
        if newly_loaded_items:
            first_exp = newly_loaded_items[0]
            for i in range(root.childCount()):
                item = root.child(i)
                if self.manager.experiments.with_browser_item(item) == first_exp:
                    item.setSelected(True)
                    if hasattr(first_exp.procedure, 'active_channel'):
                        self.view.analysis_panel.set_active_channel(
                            int(first_exp.procedure.active_channel)
                        )
                    break

    def _parse_and_load_file(self, filename: str) -> tuple:
        """
        Parse a saved file and create experiment objects (one per direction per channel).
        """
        import uuid  # for unique temp file names

        with open(filename, 'r') as f:
            content = f.read()

        blocks = content.split('[[')
        params_dict = {}
        analysis_data = []
        measurement_df = pd.DataFrame()

        for block in blocks:
            if not block.strip():
                continue
            if "EXPERIMENTAL PARAMETERS" in block:
                lines = block.split(']]')[1].strip()
                if lines:
                    try:
                        params_df = pd.read_csv(io.StringIO(lines))
                        params_dict = params_df.to_dict(orient='records')[0]
                    except Exception:
                        pass
            elif "ANALYSIS SUMMARY" in block:
                lines = block.split(']]')[1].strip()
                if lines and "No analysis" not in lines:
                    analysis_df = pd.read_csv(io.StringIO(lines))
                    # Strip units from column headers
                    metric_units_map = {
                        "EFF (%)": "EFF", "FF (%)": "FF", "Voc (mV)": "Voc",
                        "Jsc (mA/cm2)": "Jsc",
                        "Vmax (mV)": "Vmpp", "Vmpp (mV)": "Vmpp",
                        "Jmax (mA/cm2)": "Jmpp", "Jmpp (mA/cm2)": "Jmpp",
                        "Isc (A)": "Isc",
                        "Rsh (Ohm)": "Rsh", "Rs (Ohm)": "Rs",
                        "Area (cm2)": "A", "Incd. Pwr (mW/cm2)": "Incd. Pwr",
                        "Pmpp (mW)": "Pmpp",
                    }
                    analysis_df.rename(columns=metric_units_map, inplace=True)
                    analysis_data = analysis_df.to_dict(orient='records')
            elif "MEASUREMENT DATA" in block:
                lines = block.split(']]')[1].strip()
                if lines:
                    measurement_df = pd.read_csv(io.StringIO(lines), header=[0, 1, 2])

        if measurement_df.empty:
            logger.warning(f"No measurement data found in {filename}")
            return [], [], []

        experiments = []
        channels = []
        display_name = os.path.basename(filename)

        # Extract device area
        area_str = params_dict.get("Device Area (cm^2)", "0.089")
        if isinstance(area_str, str):
            area = float(area_str.split()[0]) if area_str else 0.089
        else:
            area = float(area_str)

        # Pre-process analysis: parse "Channel" -> (channel_int, direction, metrics)
        parsed_analysis = []
        for row in analysis_data:
            ch_str = str(row.get('Channel', ''))
            channel_num = None
            direction = 'Forward'
            try:
                if '_' in ch_str:
                    parts = ch_str.split('_', 1)
                    channel_num = int(parts[0])
                    direction = parts[1] if parts[1] in ('Forward', 'Reverse') else 'Forward'
                else:
                    channel_num = int(ch_str)
            except ValueError:
                continue
            if channel_num is not None:
                parsed_analysis.append({
                    'Channel': channel_num,
                    'Direction': direction,
                    'metrics': {k: v for k, v in row.items() if k != 'Channel'}
                })

        channel_cols = measurement_df.columns.get_level_values(0).unique()

        for ch_str in channel_cols:
            if not ch_str:
                continue
            try:
                channel_num = int(ch_str)
            except ValueError:
                continue

            channel_df = measurement_df[ch_str]
            directions = channel_df.columns.get_level_values(0).unique()

            for direction in directions:
                if direction not in ('Forward', 'Reverse'):
                    continue
                data_subset = channel_df[direction]
                if 'V' not in data_subset.columns or 'J' not in data_subset.columns:
                    continue

                # Extract voltage and current (already in Amperes)
                v_raw = data_subset['V'].values
                i_raw = data_subset['J'].values

                # Sort by voltage to guarantee monotonic line
                sort_idx = np.argsort(v_raw)
                v_sorted = v_raw[sort_idx]
                i_sorted = i_raw[sort_idx]

                plot_df = pd.DataFrame({
                    "Channel": channel_num,
                    "Voltage (V)": v_sorted,
                    "Current (A)": i_sorted,
                    "Time (s)": np.nan,
                    "Status": "Loaded"
                })

                # Write to unique temporary CSV
                # newline="" is load-bearing on Windows, not decoration.
                # pandas writes rows terminated with os.linesep (\r\n there),
                # and a text-mode handle then translates the \n again — so
                # every row ended \r\r\n and read back as a BLANK LINE
                # between each data point. macOS, where os.linesep is \n and
                # nothing is translated, was unaffected, which is why an
                # opened report plotted correctly there and showed spurious
                # straight segments on Windows. Both report writers already
                # pass newline=""; this one was missed.
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode='w', newline="",
                    prefix=f"ch{channel_num}_{direction}_"
                )
                temp_file.write("Channel,Voltage (V),Current (A),Time (s),Status\n")
                plot_df.to_csv(temp_file, index=False, header=False)
                temp_file.close()

                logger.debug(f"Loaded {len(plot_df)} points for Ch{channel_num} {direction}")

                procedure = JVProcedure()
                procedure.active_channel = channel_num
                procedure.sweep_direction = direction
                procedure.architecture = params_dict.get(
                    "Device Architecture", "n-i-p"
                )

                results = Results(procedure, temp_file.name)

                display_filename = f"Ch {channel_num} ({direction[:3].capitalize()}) - {display_name}"
                experiment = self._create_experiment(
                    results, display_filename,
                    channel=channel_num,
                    is_reverse=(direction == 'Reverse')
                )

                # Restore analysis for this channel & direction
                ch_analysis = next(
                    (pa for pa in parsed_analysis
                    if pa['Channel'] == channel_num and pa['Direction'] == direction),
                    None
                )
                if ch_analysis:
                    experiment.procedure.analysis_results = {
                        channel_num: {direction: ch_analysis['metrics']}
                    }
                else:
                    experiment.procedure.analysis_results = {}

                self.manager.load(experiment)
                experiments.append(experiment)
                channels.append(channel_num)

        return experiments, analysis_data, channels

    # -------------------------------------------------------------------------
    # Browser Selection
    # -------------------------------------------------------------------------

    def on_browser_selection_changed(self):
        """Update analysis panel when user selects a different experiment."""
        try:
            items = self.view.browser_widget.browser.selectedItems()
            if not items:
                return

            item = items[0]
            experiment = self.manager.experiments.with_browser_item(item)

            if not experiment:
                return

            # Get channel and direction from experiment
            channel = None
            direction = "Forward"
            
            if hasattr(experiment.procedure, 'active_channel'):
                try:
                    channel = int(experiment.procedure.active_channel)
                except (ValueError, TypeError):
                    pass
            
            if hasattr(experiment.procedure, 'sweep_direction'):
                direction = experiment.procedure.sweep_direction

            # Update analysis panel with stored results
            if hasattr(experiment.procedure, 'analysis_results'):
                results = experiment.procedure.analysis_results
                if results:
                    for ch, metrics in results.items():
                        if isinstance(metrics, dict) and "Forward" in metrics:
                            # Dual sweep mode
                            for dir_name, dir_metrics in metrics.items():
                                self.view.analysis_panel.analysis({
                                    'Channel': ch,
                                    'Direction': dir_name,
                                    **dir_metrics
                                })
                        else:
                            # Single sweep mode
                            self.view.analysis_panel.analysis({
                                'Channel': ch,
                                'Direction': 'Forward',
                                **metrics
                            })

            # Switch to the active channel tab
            if channel:
                self.view.analysis_panel.set_active_channel(channel, direction)

        except Exception as e:
            logger.error(f"Selection handler error: {e}")

    # -------------------------------------------------------------------------
    # Experiment Lifecycle
    # -------------------------------------------------------------------------

    def on_finished(self):
        """Handle post-experiment tasks after a sweep completes."""
        logger.info("Experiment finished")

        if not self.manager.experiments.has_next():
            # All sweeps are done
            if self._combined_mode and not self._combined_jv_finished:
                # Combined mode: transition from JV phase to SPO phase.
                # Skip file merge and disconnect — SPO needs instruments.
                # _start_spo_phase() handles channel selection + SPO start.
                self.view.combined_abort_button.setEnabled(True)
                self._start_spo_phase()
            else:
                self._finalize_run()
                self._return_to_idle()
        else:
            # More experiments in queue – keep instruments connected
            self.view.queue_button.setEnabled(False)
            self.is_busy = True

    # -- run-control button -------------------------------------------------
    # The one button is Abort while a sweep runs and Resume while the queue is
    # paused, so its label and its `clicked` connection have to move together.
    # They used to be set in five different places and drifted apart: after an
    # abort that emptied the queue the label said "Abort" while the click still
    # ran resume_experiment().
    RUN_CONTROL_TEXT = {"abort": "Abort", "resume": "Resume",
                        "aborting": "Aborting...", "idle": "Abort"}

    def _set_run_control(self, mode: str):
        """Put the VISIBLE run control into `mode`.

        `mode` is "abort", "resume", "aborting" or "idle".

        The label and the click handler must move together — they used to be
        set in five places and drifted apart. Just as important, they must be
        applied to whichever button the operator can actually see: each view
        has its own pair, so writing unconditionally to `abort_button` left the
        combined view showing "Abort" while the controller was waiting to
        resume. `_run_control_mode` is remembered so a view switch can repaint
        the newly-visible button into the same state.
        """
        self._run_control_mode = mode
        self._apply_run_control()

    def _apply_run_control(self):
        """Paint the current mode onto the control that is on screen."""
        mode = getattr(self, "_run_control_mode", "idle")
        try:
            _start, button = self.view.active_run_controls()
        except Exception:                         # noqa: BLE001 - very early init
            button = self.view.abort_button
        if button is None:
            return

        handler = (self.resume_experiment if mode == "resume"
                   else self.abort_experiment)
        # The combined view's own abort has extra work to do (it may be in the
        # SPO phase), so keep its handler unless we are offering Resume.
        if button is getattr(self.view, "combined_abort_button", None) \
                and mode != "resume":
            handler = self.abort_combined

        button.setText(self.RUN_CONTROL_TEXT[mode])
        button.setEnabled(mode in ("abort", "resume"))
        try:
            button.clicked.disconnect()
        except TypeError:
            pass                                  # nothing connected yet
        button.clicked.connect(handler)

    def _rearm_manager(self):
        """Let the manager start queued experiments again.

        `Manager.abort()` clears `_start_on_add` and `_is_continuous`, and the
        only thing in pymeasure that restores them is `Manager.resume()`. Any
        abort the operator did not follow with a Resume therefore left the
        manager unable to start anything for the rest of the session:
        `Manager.queue()` appended the row and returned without calling
        `next()`, so the browser filled with QUEUED sweeps and nothing ran.
        Every path back to idle goes through here.
        """
        self.manager._start_on_add = True
        self.manager._is_continuous = True

    def _discard_aborted_sweep(self, experiment):
        """Drop the partial file an aborted sweep left behind.

        An aborted sweep is discarded, never re-run — the same thing that
        happens to it when there are more sweeps behind it in the queue. Its
        temp file holds a truncated J-V curve and no `[[ANALYSIS]]` block
        (that is written only on completion), so leaving it in
        `experiment_files` merged a half-measured curve into the report as if
        it were a real measurement, silently and without metrics.
        """
        data_file = None
        for attr in ("data_filename", "data_path", "filename"):
            value = getattr(getattr(experiment, "results", None), attr, None)
            if isinstance(value, str) and value:
                data_file = os.path.normcase(os.path.abspath(value))
                break
        if data_file is None:
            return
        for key, path in list(self.experiment_files.items()):
            if os.path.normcase(os.path.abspath(path)) != data_file:
                continue
            self.experiment_files.pop(key, None)
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning(f"Could not remove the aborted sweep's "
                               f"partial file {os.path.basename(path)}: {exc}")
            logger.info(f"Discarded the aborted sweep's partial data ({key})")
            return

    def _finalize_run(self):
        """Merge, publish and release the hardware at the end of a run.

        Reachable from `on_finished()` AND from `on_abort_returned()`: aborting
        the last sweep of a queue used to skip this entirely, so the sweeps
        that had already completed were left on disk as `_temp` files that
        nothing merged, nothing published, and the next run's
        `queue_experiment()` forgot about.
        """
        try:
            if self.is_single_file_mode:
                self._merge_channel_files()
            else:
                self._process_multi_files()
        except Exception as e:
            logger.error(f"File post-processing error: {e}")

        self._disconnect_instruments()

    def abort_experiment(self):
        """Abort the currently running experiment."""
        if not self.manager.is_running():
            # `Manager.abort()` raises when nothing is running, and an
            # unhandled exception in a Qt slot is not survivable in general.
            # Reaching here means the UI thought a sweep was running and none
            # was, so put the window back into a state the operator can use.
            if self.manager.experiments.has_next():
                # Nothing is running but sweeps are still QUEUED: this is the
                # paused state, and "Abort" here means "give up on the rest".
                # Returning to idle without clearing them left the window
                # looking idle while the manager still held pending work.
                logger.warning(
                    "Abort with no sweep running — discarding the queued "
                    "sweeps that were still pending.")
                self.clear_experiments()
                return
            logger.warning("Abort ignored: no experiment is running")
            self._return_to_idle()
            return

        logger.info("Abort requested")
        self.view.queue_button.setEnabled(False)
        self._set_run_control("aborting")
        try:
            self.manager.abort()
        except Exception as exc:
            logger.error(f"Abort failed: {exc}")
            self._return_to_idle()

    def resume_experiment(self):
        """Continue the queue after an abort, with the NEXT sweep.

        The aborted sweep is not re-run: `ExperimentQueue.next()` only returns
        experiments still marked QUEUED, and this button means "carry on with
        what is left".
        """
        logger.info("Resuming experiment queue")

        # Ensure output is off before resuming
        try:
            self.view.instrument_manager.keithley.write(":OUTP OFF")
            self.view.instrument_manager.keithley.write(":ABOR")
        except Exception:
            pass

        start, _run = self.view.active_run_controls()
        if start is not None:
            start.setEnabled(False)
        self._set_run_control("abort")

        if self.manager.experiments.has_next():
            self.manager.resume()
        else:
            self._finalize_run()
            self._return_to_idle()

    def _return_to_idle(self):
        """Nothing is running and nothing is queued: accept a new run."""
        self._rearm_manager()
        self._set_run_control("idle")
        # Re-enable BOTH the generic Queue button and whichever start control
        # the active view shows, so returning to idle from the combined tab
        # does not leave "Run JV + SPO" greyed out.
        self.view.queue_button.setEnabled(True)
        try:
            start, _run = self.view.active_run_controls()
            if start is not None:
                start.setEnabled(True)
        except Exception:                         # noqa: BLE001
            pass
        self.view.browser_widget.clear_button.setEnabled(True)
        self.view.browser_widget.show_button.setEnabled(True)
        self.view.browser_widget.hide_button.setEnabled(True)

        # Unlock the configuration inputs. `start_combined_run()` disables the
        # whole JV parameter block — which is ONE shared ParameterTab, so the
        # architecture toggle, channels and voltages all freeze with it. Every
        # combined path re-enabled it except the one that goes through Clear:
        # abort mid-queue, then Clear, and the run button came back while the
        # n-i-p / p-i-n toggle stayed dead. Idle means editable, so this
        # belongs here rather than in each exit path.
        self._set_combined_config_enabled(True)

        self.is_busy = False

    def _refuse_busy(self, action: str) -> None:
        """Explain a refusal instead of only writing it to the log.

        A paused queue looks idle — no sweep is running and the plot is
        static — so a silent `logger.warning` reads to the operator as "the
        button does nothing". Name the state and the two ways out of it.
        """
        if self.spo_running:
            detail = "An SPO measurement is still running."
            remedy = "Abort it first."
        else:
            detail = ("A J-V queue is paused: sweeps from the last run are "
                      "still waiting.")
            remedy = ("Press Resume to finish them, or Clear to discard "
                      "them, and then try again.")
        logger.warning(f"{action} refused — {detail}")
        from PyQt5 import QtWidgets
        QtWidgets.QMessageBox.information(
            self.view, f"Cannot start {action}", f"{detail}\n\n{remedy}")

    def clear_experiments(self):
        """Clear all experiments from the manager, and go back to idle.

        Clearing is the operator's way OUT of a paused queue, so it has to
        leave the controller genuinely idle. It used to clear the queue and
        re-arm the manager but leave `is_busy` True, which meant every later
        "already busy" guard still fired: after aborting part-way through a
        multi-channel run, Clear appeared to work and yet Run JV+SPO, a new
        queue and SPO all silently refused to start, with no way back short of
        restarting the application.
        """
        self.manager.clear()
        self._rearm_manager()      # clearing after an abort must not stay dead
        self.finished_experiment_count = 0
        self._sync_channel_indicators([])
        if not self.spo_running:
            self._return_to_idle()

    def _sync_channel_indicators(self, channels):
        """Push the active channel list and architecture to the window badges."""
        if hasattr(self.view, 'update_channel_indicators'):
            self.view.update_channel_indicators(channels)
        # Also push the architecture from the first browser experiment
        arch = self._find_first_architecture()
        if hasattr(self.view, 'update_architecture_badge'):
            self.view.update_architecture_badge(arch)

    def _find_first_architecture(self) -> str:
        """Return the architecture string from the first browser experiment,
        or empty string if none found."""
        root = self.view.browser_widget.browser.invisibleRootItem()
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'architecture'):
                return exp.procedure.architecture or ""
        return ""

    def _keithley_usable(self, context: str) -> bool:
        """True when the Keithley's VISA session can actually be written to.

        `connect_keithley()` returning without raising is not proof of a
        usable instrument: a handle whose session was closed elsewhere used to
        short-circuit the connect and then fail on the procedure's very first
        :OUTP OFF, deep inside the worker thread, as a raw pyvisa traceback.
        Checking here fails fast, with a message the operator can act on, and
        before the MUX is told to select a channel for a run that cannot start.
        """
        if self.view.instrument_manager.is_keithley_alive():
            return True
        logger.error(
            f"{context}: Keithley VISA session is closed — cannot start."
        )
        self.view.update_instrument_lights()
        QtWidgets.QMessageBox.critical(
            self.view, "Keithley Not Available",
            "The Keithley 2400 connection is no longer usable (its VISA "
            "session has been closed).\n\n"
            "Reconnect the instrument, then try again."
        )
        return False

    def _disconnect_instruments(self):
        """Disconnect from hardware instruments."""
        try:
            self.view.instrument_manager.disconnect_keithley()
            self.view.instrument_manager.disconnect_mux()
        except Exception:
            pass
        finally:
            self.view.update_instrument_lights()

    # -------------------------------------------------------------------------
    # Combined JV + SPO Workflow
    # -------------------------------------------------------------------------

    def start_combined_run(self):
        """Run JV sweeps on selected channels, then auto-transition to SPO.

        1. Collects JV + SPO params from the CombinedTab.
        2. Queues JV sweeps normally (reusing existing queue_experiment logic).
        3. When JV finishes (on_finished), _start_spo_phase() runs SPO on the
           best channel using its Vmpp as the hold voltage.
        """
        if self.is_busy or self.spo_running:
            self._refuse_busy("Run JV + SPO")
            return

        combined_tab = self.view.combined_tab

        # ---- Collect parameters ------------------------------------------
        params_dict = combined_tab.get_jv_parameters()
        analysis_dict = self.view.active_analysis_tab.get_parameters()
        instr_dict = self.view.active_instr_tab.get_parameters()
        spo_params = combined_tab.get_spo_parameters()

        selected_channels = combined_tab.get_selected_channels()
        if not selected_channels:
            logger.warning("No channels selected.")
            return

        self._combined_mode = True
        self._combined_jv_finished = False
        self._combined_best_channel = None
        self._combined_best_vmpp = 0.0
        self._show_combined_spo_setpoint()      # blank until the JV phase picks
        self._combined_spo_params = spo_params
        # Snapshot shared measurement settings NOW: the SPO phase must run
        # with the parameters that were active when the user started the run,
        # not whatever the (still editable) widgets contain later.
        self._combined_spo_params['device_area'] = params_dict.get('device_area', 0.089)
        self._combined_spo_params['compliance_current'] = params_dict.get('compliance_current', 0.18)
        self._combined_spo_params['incident_power'] = analysis_dict.get('incident_power', 100.0)

        # Build procedure params (same as queue_experiment)
        procedure_params = {
            'user_name': self.view.username,
            **params_dict,
            **instr_dict,
            **analysis_dict,
        }
        self.current_notes_text = procedure_params.pop('notes_text', '')
        self.current_save_notes = procedure_params.pop('save_notes', False)
        file_params = self.view.file_panel.get_parameters()

        # ---- Reset state for this run ------------------------------------
        self.experiment_files = {}
        self.processed_files = set()
        self.is_single_file_mode = file_params['single_file']

        # ---- Connect hardware --------------------------------------------
        try:
            self.view.instrument_manager.connect_keithley(simulation=False)
            self.view.instrument_manager.connect_mux(simulation=False)
        except Exception as e:
            logger.error(f"Hardware connection failed: {e}")
            self.view.update_instrument_lights()
            self._combined_mode = False
            return
        self.view.update_instrument_lights()

        if not self._keithley_usable("Combined JV+SPO run"):
            self._combined_mode = False
            return

        # Same as queue_experiment: clear the previous run only once this one
        # is certain to start.
        self._reset_view_for_new_run(selected_channels)

        # ---- Generate file paths -----------------------------------------
        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        # Same extension handling as queue_experiment: a bare "Test" must not
        # produce an extension-less file, and "Sample_1.5cm" must not be split
        # at the dot.
        base, ext = _split_output_name(file_params['filename'])
        self._run_base_name = base
        self._run_ext = ext
        directory = file_params['directory']

        # JV file path (same as single-file mode)
        channel_list_str = "_".join(map(str, selected_channels))
        self.merged_file_path = os.path.join(
            directory, f"{base}_{filename_timestamp}_ch{channel_list_str}{ext}"
        )
        self.merged_data_written = False

        # SPO file paths (SPO_ prefix, shared timestamp).
        # Raw CSV and formatted report MUST use different filenames —
        # SpoReport.finalize() opens with "w" and would overwrite the raw data.
        #
        # They go in the SPO folder, NOT alongside the JV report: `directory`
        # is the file panel's path, which is always the "JV" mode folder, so
        # a combined run used to file its SPO data under Main. A standalone SPO
        # run resolves the SPO folder itself (see SpoProcedure), but in
        # combined mode the controller passes an explicit csv_path, which
        # bypasses that — hence resolving it here.
        #
        # DirectoryManager is a process-wide singleton: switch its mode, read
        # the path, and ALWAYS restore it. An exception here (permissions, a
        # missing network drive) must not strand the singleton in "SPO" mode
        # for every subsequent JV run.
        spo_directory = directory
        dir_manager = getattr(self.view, 'dir_manager', None)
        if dir_manager is not None:
            previous_mode = dir_manager.mode
            try:
                dir_manager.set_mode("SPO")
                spo_directory = dir_manager.get_current_directory(create=True)
            except Exception as e:
                logger.error(
                    f"[Combined] Could not resolve the SPO output folder "
                    f"({e}); falling back to {directory}"
                )
                spo_directory = directory
            finally:
                dir_manager.set_mode(previous_mode)

        spo_filename = f"SPO_{base}_{filename_timestamp}.csv"
        self._combined_spo_csv_path = os.path.join(spo_directory, spo_filename)
        self._combined_spo_report_path = os.path.join(
            spo_directory, f"SPO_report_{base}_{filename_timestamp}.csv"
        )

        # ---- Queue JV sweeps (reuse existing per-channel logic) ----------
        for channel_num in selected_channels:
            base_params = procedure_params.copy()
            for i in range(1, 7):
                base_params[f'channel{i}'] = (i == channel_num)

            # Forward Sweep
            fwd_params = base_params.copy()
            fwd_params['single_sweep_mode'] = True
            fwd_params['sweep_direction'] = 'Forward'

            fwd_temp_path = os.path.join(
                directory,
                f"{base}_{filename_timestamp}_ch{channel_num}_forward_temp{ext}"
            )
            self.experiment_files[f"{channel_num}_forward"] = fwd_temp_path

            proc_fwd = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=False,
                active_channel=channel_num,
                check_errors_between_points=False,
                **fwd_params
            )
            results_fwd = Results(proc_fwd, fwd_temp_path)
            proc_fwd.results = results_fwd

            display_name_fwd = (
                f"Ch {channel_num} (Fwd) - {os.path.basename(self.merged_file_path)}"
            )
            exp_fwd = self._create_experiment(
                results_fwd, display_name_fwd, channel=channel_num, is_reverse=False
            )
            self.manager.queue(exp_fwd)
            logger.info(f"[Combined] Queued Ch {channel_num} (Forward)")

            # Reverse Sweep (only in dual-sweep mode). The Single Sweep Mode
            # checkbox lives in the ANALYSIS settings tab, not the parameter
            # tab — reading it from params_dict silently always queued
            # reverse sweeps.
            if not analysis_dict.get('single_sweep_mode', False):
                rev_params = base_params.copy()
                rev_params['single_sweep_mode'] = True
                rev_params['sweep_direction'] = 'Reverse'
                rev_params['start_voltage'] = params_dict.get('stop_voltage', -0.2)
                rev_params['stop_voltage'] = params_dict.get('start_voltage', 1.2)

                rev_temp_path = os.path.join(
                    directory,
                    f"{base}_{filename_timestamp}_ch{channel_num}_reverse_temp{ext}"
                )
                self.experiment_files[f"{channel_num}_reverse"] = rev_temp_path

                proc_rev = JVProcedure(
                    instrument=self.view.instrument_manager.keithley,
                    mux=self.view.instrument_manager.mux,
                    manager=self.view.instrument_manager,
                    simulation=False,
                    active_channel=channel_num,
                    check_errors_between_points=False,
                    **rev_params
                )
                results_rev = Results(proc_rev, rev_temp_path)
                proc_rev.results = results_rev

                display_name_rev = (
                    f"Ch {channel_num} (Rev) - {os.path.basename(self.merged_file_path)}"
                )
                exp_rev = self._create_experiment(
                    results_rev, display_name_rev,
                    channel=channel_num, is_reverse=True
                )
                self.manager.queue(exp_rev)
                logger.info(f"[Combined] Queued Ch {channel_num} (Reverse)")

        # Disable the combined run button, enable abort; lock the config
        # inputs so mid-run edits cannot leak into the SPO phase.
        self.view.combined_run_button.setEnabled(False)
        self.view.combined_abort_button.setEnabled(True)
        self._set_combined_config_enabled(False)
        self.is_busy = True
        logger.info("[Combined] JV sweeps queued; waiting for completion...")

    def _set_combined_config_enabled(self, enabled: bool):
        """Lock/unlock the combined tab's configuration inputs during a run."""
        try:
            self.view.combined_tab.set_config_enabled(enabled)
        except Exception as e:
            logger.debug(f"set_config_enabled({enabled}) unavailable: {e}")

    def _start_spo_phase(self):
        """Transition from JV phase to SPO phase within a combined run.

        Selects the best channel from JV results, then starts SPO on that
        channel using the auto-determined Vmpp as the hold voltage.
        """
        if not self._combined_mode:
            return

        # ---- Find the best channel from all finished experiments ----------
        root = self.view.browser_widget.browser.invisibleRootItem()
        experiments = []
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp is not None:
                experiments.append(exp)

        try:
            best_ch, best_vmpp = self._select_best_channel(experiments)
        except ValueError:
            logger.error("[Combined] No valid JV data — SPO aborted.")
            QtWidgets.QMessageBox.warning(
                self.view, "JV + SPO",
                "No valid J‑V data — SPO aborted.\n\n"
                "All channels failed to produce analysable results."
            )
            self._reset_combined_state()
            self._disconnect_instruments()
            self.view.combined_run_button.setEnabled(True)
            self.view.combined_abort_button.setEnabled(False)
            self._set_combined_config_enabled(True)
            self.is_busy = False
            return

        self._combined_best_channel = best_ch
        self._combined_best_vmpp = best_vmpp
        self._combined_jv_finished = True

        # Tell the operator what the SPO phase decided. Vmpp is SIGNED and is
        # used directly as the hold voltage, so show the signed value — a
        # p-i-n cell legitimately holds at a negative voltage and an
        # unsigned display here would look like a bug.
        self._show_combined_spo_setpoint(best_ch, best_vmpp)

        logger.info(
            f"[Combined] JV sweeps complete. Best channel: Ch {best_ch} "
            f"(Vmpp = {best_vmpp * 1000:.1f} mV)"
        )

        # ---- Configure SPO parameters -------------------------------------
        # Use the snapshot taken at run start — never re-read live widgets
        # mid-run (the user could have edited them during the JV phase).
        spo_params = self._combined_spo_params
        device_area = spo_params.get('device_area', 0.089)
        compliance_current = spo_params.get('compliance_current', 0.18)
        incident_power = spo_params.get('incident_power', 100.0)

        # ---- Build SpoProcedure with custom output paths ------------------
        self._spo_times = []
        self._spo_currents = []
        self._spo_voltages = []
        self._combined_spo_powers = []

        # Clear the combined SPO plot for the new run
        self.view.combined_spo_curve.setData([], [])
        self.view.combined_spo_mean.setText("— mW")
        self.view.combined_spo_drift.setText("— %")
        self.view.combined_spo_elapsed.setText("— s")

        self._spo_procedure = SpoProcedure(
            instrument=self.view.instrument_manager.keithley,
            mux=self.view.instrument_manager.mux,
            manager=self.view.instrument_manager,
            simulation=False,
            active_channel=best_ch,
            hold_voltage=best_vmpp,
            hold_duration=spo_params['hold_duration'],
            sampling_interval=spo_params['sampling_interval'],
            preconditioning_time=spo_params['preconditioning_time'],
            device_area=device_area,
            incident_power=incident_power,
            compliance_current=compliance_current,
            nplc=1.0,
            user_name=self.view.username,
            username=self.view.username,
            check_errors_between_points=False,
            csv_path=self._combined_spo_csv_path,
            report_path=self._combined_spo_report_path,
        )

        self._spo_worker = SpoWorker(self._spo_procedure)
        self._spo_worker.results_ready.connect(self._on_spo_results)
        self._spo_worker.status_changed.connect(self._on_combined_spo_status)
        self._spo_worker.run_finished.connect(self._on_combined_spo_finished)
        self._spo_worker.run_failed.connect(self._on_combined_spo_failed)

        self.spo_running = True
        self._spo_worker.start()
        logger.info(
            f"[Combined] SPO started on Ch {best_ch} "
            f"(hold = {best_vmpp:.4f} V, duration = {spo_params['hold_duration']} s)"
        )

    def abort_combined(self):
        """Abort the combined run — kills JV queue or SPO worker, whichever
        is active, and returns the system to idle."""
        logger.info("[Combined] Abort requested by user.")

        if self.spo_running and self._spo_worker is not None:
            # Abort the SPO phase. Do NOT reset combined state here — the
            # worker's run_finished signal drives _on_combined_spo_finished,
            # which saves the collected data and performs the cleanup. The
            # bounded wait is safe: the hold loop checks the stop flag every
            # 0.1 s (abort-aware sleep).
            self._spo_worker.abort()
            self._spo_worker.wait(15000)
        elif self.is_busy:
            # Abort the JV phase. Cleanup happens in on_abort_returned's
            # combined branch, which needs _combined_mode still True —
            # resetting state synchronously here made that branch
            # unreachable and wedged the combined view.
            try:
                self.manager.abort()
            except Exception as e:
                logger.error(f"[Combined] JV abort failed: {e}")

    def _final_report_name(self, timestamp: str, channel: int) -> str:
        """`<operator's name>_<timestamp>_ch<N>.csv` for a per-channel report.

        Falls back to "Output" only when nothing was typed, which the filename
        validation should already prevent.
        """
        base = getattr(self, "_run_base_name", "") or "Output"
        ext = getattr(self, "_run_ext", "") or ".csv"
        return f"{base}_{timestamp}_ch{channel}{ext}"

    @staticmethod
    def _finite(value):
        """Return `value` as a float, or None if it is missing or not finite.

        NaN is the whole problem this guards against. `compute_jv_metrics()`
        reports NaN — deliberately — when a sweep never crosses V=0 or I=0, so
        a dark, truncated or disconnected channel yields NaN metrics rather
        than a fabricated number. NaN then poisons every ordinary test:
        `NaN or 0` evaluates to NaN because NaN is truthy, `NaN <= 0` is False
        so a "skip the bad ones" guard passes it through, and sorting is
        undefined because every comparison with NaN is False. A NaN channel
        could therefore win the ranking and hand SPO a NaN hold voltage.
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _select_best_channel(self, experiments: list) -> tuple:
        """Select the best channel for the SPO hold, by EFF then |Vmpp| then Jsc.

        Considers BOTH sweep directions. Previously only "Forward" was read, so
        a channel whose forward sweep failed was discarded even when its
        reverse sweep was perfectly good — and, worse, its NaN metrics were
        still ranked. Each (channel, direction) that produced finite, usable
        numbers competes; the winner's SIGNED Vmpp becomes the hold voltage.

        Returns:
            (channel_number, vmpp_volts) — Vmpp signed, in volts.

        Raises:
            ValueError: if no channel produced a finite, positive-efficiency
                result. Refusing is correct: holding at a made-up voltage
                would produce a confidently wrong degradation curve.
        """
        candidates = []
        rejected = []

        for exp in experiments:
            results = getattr(getattr(exp, "procedure", None),
                              "analysis_results", None)
            if not results:
                continue

            for ch, metrics in results.items():
                try:
                    ch = int(ch)
                except (ValueError, TypeError):
                    continue
                if not isinstance(metrics, dict):
                    continue

                # Either {"Forward": {...}, "Reverse": {...}} or a flat dict.
                first = next(iter(metrics.values()), None)
                by_direction = (metrics if isinstance(first, dict)
                                else {"Forward": metrics})

                for direction, values in by_direction.items():
                    if not isinstance(values, dict):
                        continue
                    eff = self._finite(values.get("EFF"))
                    vmpp_mv = self._finite(values.get("Vmpp"))
                    jsc = self._finite(values.get("Jsc"))

                    if vmpp_mv is None or eff is None:
                        rejected.append(f"Ch{ch} {direction}: non-finite metrics")
                        continue
                    if eff <= 0 or vmpp_mv == 0:
                        rejected.append(
                            f"Ch{ch} {direction}: EFF={eff:.3g}, Vmpp={vmpp_mv:.3g}")
                        continue

                    candidates.append(
                        (eff, abs(vmpp_mv), jsc if jsc is not None else 0.0,
                         ch, vmpp_mv, direction))

        if rejected:
            logger.warning(
                "Channels excluded from SPO selection — %s", "; ".join(rejected))

        if not candidates:
            raise ValueError("No valid JV analysis results found")

        # EFF desc, then |Vmpp| desc, then Jsc desc. Every value here is
        # finite, so the ordering is well defined.
        candidates.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
        eff, _abs_vmpp, jsc, best_ch, vmpp_mv, direction = candidates[0]
        logger.info(
            "SPO channel selected: Ch%d (%s) — EFF %.2f%%, Vmpp %.1f mV, "
            "Jsc %.3f mA/cm2, from %d usable candidate(s)",
            best_ch, direction, eff, vmpp_mv, jsc, len(candidates))
        return best_ch, vmpp_mv / 1000.0      # signed mV -> signed V

    def _on_combined_spo_status(self, status: str):
        logger.info(f"[Combined] SPO status: {status}")

    def _on_combined_spo_finished(self, proc):
        """Handle SPO completion within a combined run."""
        self.spo_running = False
        self.is_busy = False

        # Merge JV files (normal path)
        try:
            if self.is_single_file_mode:
                self._merge_channel_files()
            else:
                self._process_multi_files()
        except Exception as e:
            logger.error(f"[Combined] JV file merge failed: {e}")

        # Finalise SPO report with custom path
        report_path = None
        metrics = {}
        try:
            if self._spo_times:
                metrics = compute_spo_metrics(
                    self._spo_times, self._spo_currents, self._spo_voltages
                )
                report = proc.report if proc is not None else None
                if report is not None:
                    metrics_units = dict(SPO_METRICS_UNITS)
                    # The same notes the J-V half of this run recorded, in
                    # the same section of the file. A combined run is one
                    # experiment to the operator, so both reports carry it.
                    report_path = report.finalize(
                        metrics, metrics_units,
                        output_path=self._combined_spo_report_path,
                        notes_text=getattr(self, 'current_notes_text', ''),
                        save_notes=getattr(self, 'current_save_notes', False),
                    )
                    # The raw journal is discarded by finalize() once the
                    # report is verified, so it is not listed here — logging a
                    # path that no longer exists sends people looking for it.
                    logger.info(f"[Combined] SPO report saved: {report_path}")
        except Exception as e:
            logger.error(f"[Combined] SPO report finalization failed: {e}")

        # Capture BEFORE resetting state — _reset_combined_state() nulls it.
        best_channel = self._combined_best_channel

        self._disconnect_instruments()
        self._reset_combined_state()
        self.view.combined_run_button.setEnabled(True)
        self.view.combined_abort_button.setEnabled(False)
        self._set_combined_config_enabled(True)
        self._spo_worker = None
        self._spo_procedure = None

        QtWidgets.QMessageBox.information(
            self.view, "JV + SPO Complete",
            f"Combined run finished.\n\n"
            f"Best channel: Ch {best_channel}\n"
            f"JV file: {os.path.basename(getattr(self, 'merged_file_path', ''))}\n"
            f"SPO file: {os.path.basename(report_path) if report_path else 'N/A'}"
        )

    def _on_combined_spo_failed(self, message: str):
        logger.error(f"[Combined] SPO run failed: {message}")
        # SPO failure still saves JV data — fall through to finished handler
        # which is always emitted after failed.

    def _show_combined_spo_setpoint(self, channel=None, hold_voltage=None):
        """Display the SPO channel and hold voltage on the combined SPO panel.

        Called with no arguments to blank the chips at the start of a run and
        when combined state is reset.
        """
        channel_label = getattr(self.view, 'combined_spo_channel', None)
        hold_label = getattr(self.view, 'combined_spo_hold', None)
        if channel_label is None or hold_label is None:
            return          # older layout without the chips — nothing to do
        if channel is None:
            channel_label.setText("—")
            hold_label.setText("— mV")
            return
        channel_label.setText(f"Ch {int(channel)}")
        hold_label.setText(f"{float(hold_voltage) * 1000.0:+.1f} mV")

    def _reset_combined_state(self):
        """Clear all combined-mode state flags."""
        self._show_combined_spo_setpoint()
        self._combined_mode = False
        self._combined_jv_finished = False
        self._combined_best_channel = None
        self._combined_best_vmpp = 0.0
        self._combined_spo_params = {}
        self._combined_spo_csv_path = None
        self._combined_spo_report_path = None
        self._combined_spo_powers = []

    # -------------------------------------------------------------------------
    # SPO (Set-Point Operation) Lifecycle
    # -------------------------------------------------------------------------

    def start_spo(self):
        """Collect parameters from spo_widget, create a SpoProcedure, and run it."""
        if not SPO_AVAILABLE:
            logger.warning("Cannot start SPO: SPO module is not available.")
            return

        if self.is_busy or self.spo_running:
            self._refuse_busy("an SPO run")
            return

        if not self.spo_widget.has_valid_hold_voltage():
            logger.warning("Cannot start SPO: no valid Vmpp has been set.")
            return

        spo_params = self.spo_widget.get_parameters()

        # Reuse existing parameter tabs for shared measurement settings.
        params_dict = self.view.params_tab.get_parameters()
        analysis_dict = self.view.analysis_settings_tab.get_parameters()

        device_area = params_dict.get('device_area', 0.089)
        compliance_current = params_dict.get('compliance_current', 0.18)
        incident_power = analysis_dict.get('incident_power', 100.0)

        sim_mode = False
        try:
            self.view.instrument_manager.connect_keithley(simulation=sim_mode)
            self.view.instrument_manager.connect_mux(simulation=sim_mode)
        except Exception as e:
            logger.error(f"SPO hardware connection failed: {e}")
            self.view.update_instrument_lights()
            return
        self.view.update_instrument_lights()

        if not self._keithley_usable("SPO run"):
            return

        self._spo_times = []
        self._spo_currents = []
        self._spo_voltages = []

        self._spo_procedure = SpoProcedure(
            instrument=self.view.instrument_manager.keithley,
            mux=self.view.instrument_manager.mux,
            manager=self.view.instrument_manager,
            simulation=sim_mode,
            active_channel=spo_params['active_channel'],
            hold_voltage=spo_params['hold_voltage'],
            hold_duration=spo_params['hold_duration'],
            sampling_interval=spo_params['sampling_interval'],
            preconditioning_time=spo_params['preconditioning_time'],
            device_area=device_area,
            incident_power=incident_power,
            compliance_current=compliance_current,
            nplc=1.0,  # stable DC reading; SPO does not use sweep-rate-derived NPLC
            user_name=self.view.username,
            username=self.view.username,
            check_errors_between_points=False,
        )

        self._spo_worker = SpoWorker(self._spo_procedure)
        self._spo_worker.results_ready.connect(self._on_spo_results)
        self._spo_worker.status_changed.connect(self._on_spo_status)
        self._spo_worker.run_finished.connect(self._on_spo_run_finished)
        self._spo_worker.run_failed.connect(self._on_spo_run_failed)

        self.spo_running = True
        self.is_busy = True
        self.spo_widget.start_spo()
        self.view.spo_start_button.setEnabled(False)
        self.view.spo_abort_button.setEnabled(True)

        self._spo_worker.start()
        logger.info(f"SPO started on Channel {spo_params['active_channel']}")

    def abort_spo(self):
        """Abort the currently running SPO test via the SpoWorker."""
        if not SPO_AVAILABLE or not self.spo_running or self._spo_worker is None:
            return
        logger.info("SPO abort requested")
        self.view.spo_abort_button.setEnabled(False)
        self.spo_widget.abort_spo()
        self._spo_worker.abort()

    def _on_spo_results(self, record: dict):
        """Handle each live SPO sample: accumulate for metrics and update the plot."""
        self._spo_times.append(record["Time (s)"])
        self._spo_currents.append(record["Current (A)"])
        self._spo_voltages.append(record["Voltage (V)"])

        if self._combined_mode:
            # Combined mode: feed data to the side-by-side SPO plot.
            # Display GENERATED power (-V*I, positive while generating) to
            # match the sign convention of the report metrics.
            power_mw = -record["Power (W)"] * 1000.0
            self._combined_spo_powers.append(power_mw)
            self.view.combined_spo_curve.setData(
                self._spo_times, self._combined_spo_powers
            )
            self.view.combined_spo_elapsed.setText(
                f"{record['Time (s)']:.1f} s"
            )
            if len(self._spo_times) % 5 == 0:
                metrics = compute_spo_metrics(
                    self._spo_times, self._spo_currents, self._spo_voltages
                )
                self.view.combined_spo_mean.setText(
                    f"{metrics.get('mean_power_mw', 0):.2f} mW"
                )
                self.view.combined_spo_drift.setText(
                    f"{metrics.get('drift_percent', 0):.2f} %"
                )
        else:
            # Plot generated power (-V*I): positive while the cell delivers.
            self.spo_widget.update_plot(record["Time (s)"], -record["Power (W)"])
            if len(self._spo_times) % 5 == 0:
                metrics = compute_spo_metrics(
                    self._spo_times, self._spo_currents, self._spo_voltages
                )
                self.spo_widget.update_metrics(metrics)

    def _on_spo_status(self, status: str):
        logger.info(f"SPO status: {status}")

    def _on_spo_run_finished(self, proc):
        """Compute final metrics, finalize the report, and reset SPO UI state.

        Args:
            proc: The SpoProcedure instance that just finished, emitted by
                SpoWorker.run_finished so we can read proc.csv_path / proc.report.
        """
        self.spo_running = False
        self.is_busy = False
        self.view.spo_abort_button.setEnabled(False)

        report_path = None
        metrics = {}
        try:
            if self._spo_times:
                metrics = compute_spo_metrics(
                    self._spo_times, self._spo_currents, self._spo_voltages
                )
                report = proc.report if proc is not None else None
                if report is not None:
                    metrics_units = dict(SPO_METRICS_UNITS)
                    report_path = report.finalize(metrics, metrics_units)
                    logger.info(f"SPO report saved: {report_path}")
        except Exception as e:
            logger.error(f"Failed to finalize SPO report: {e}")

        self.spo_widget.on_spo_finished(metrics, report_path)
        self._disconnect_instruments()
        # Route through the window's gating logic so the start button honours
        # BOTH conditions (valid hold voltage AND valid filename) — setting
        # enabled directly bypassed the filename gate.
        self.view._on_spo_vmpp_ready(self.spo_widget.has_valid_hold_voltage())

        self._spo_worker = None
        self._spo_procedure = None

    def _on_spo_run_failed(self, message: str):
        """Log the failure; cleanup and report finalization happen in
        `_on_spo_run_finished`, which SpoWorker always emits afterward."""
        logger.error(f"SPO run failed: {message}")

    def on_abort_returned(self, experiment=None):
        """Handle post-abort state.

        `experiment` is what pymeasure's `abort_returned` signal carries — the
        sweep that was cut short. Its partial file is discarded here so it can
        never reach a report; the browser row stays, marked Aborted, so the
        operator can still see what they stopped.
        """
        if experiment is not None:
            self._discard_aborted_sweep(experiment)

        # The manager disarms itself on abort. Re-arm on every path out of
        # here, or the next run queues sweeps that never start.
        self._rearm_manager()

        if self._combined_mode:
            # An abort during the SPO phase ends the run: there is no queue
            # left to resume into.
            if self._combined_jv_finished or not self.manager.experiments.has_next():
                self._finalize_run()
                self._reset_combined_state()
                self.view.combined_run_button.setEnabled(True)
                self.view.combined_abort_button.setEnabled(False)
                self._set_combined_config_enabled(True)
                self.is_busy = False
                self._set_run_control("idle")
                logger.info("[Combined] Aborted by user.")
                return

            # Aborting one sweep of the J-V phase behaves like it does in
            # Advanced: the rest of the queue is still there, so offer Resume.
            # Combined mode stays ON, so draining the queue still hands over to
            # the SPO phase — the operator loses one channel, not the run.
            self._set_run_control("resume")
            self.view.combined_run_button.setEnabled(False)
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = True
            logger.info(
                "[Combined] Sweep aborted — press Resume to continue with "
                "the rest of the queue.")
            return

        if self.manager.experiments.has_next():
            # Instruments stay connected: the operator can carry on with the
            # rest of the queue.
            self._set_run_control("resume")
            self.view.queue_button.setEnabled(False)
            # Re-enable Clear. `on_running()` disabled it for the duration of
            # the sweep, and a paused queue with Clear still greyed out is a
            # dead end: Queue is disabled, the combined run refuses because
            # the controller is busy, and Resume is the only door left.
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = True
        else:
            self._finalize_run()
            self._return_to_idle()

    def on_failed(self, experiment=None):
        """Handle a sweep that died mid-run, and keep the queue usable.

        This is the deadlock path. pymeasure's `BaseManager._finish()` calls
        `next()` when the queue is continuous, but `_failed()` deliberately
        does not — it cleans up and emits `failed`, leaving the decision to the
        application. Nothing was connected to that signal, so when a sweep
        raised (a MUX write failing on a yanked USB adapter, say) the run
        simply stopped being tracked: `is_busy` stayed True, the run-control
        button stayed on "Abort" while `manager.is_running()` was False, and
        every experiment still queued sat at QUEUED forever with no way to
        start them.

        A failure is treated exactly like an abort, because that is what it is
        from the operator's point of view — this sweep produced no usable data,
        and what matters is whether anything is left to run. The partial file
        is discarded so a truncated curve can never reach a report, and the
        manager is re-armed so the queue can move again.
        """
        status = ""
        if experiment is not None:
            procedure = getattr(experiment, "procedure", None)
            status = getattr(procedure, "status", "")
            self._discard_aborted_sweep(experiment)
        logger.error(
            "Sweep failed%s — its partial data has been discarded. "
            "See the traceback above for the cause.",
            f" (status {status})" if status else "",
        )

        # `_failed()` does not disarm the manager the way `abort()` does, but
        # re-arming is harmless and makes this path identical to the abort one
        # rather than subtly different.
        self._rearm_manager()

        if self._combined_mode:
            if self._combined_jv_finished or not self.manager.experiments.has_next():
                self._finalize_run()
                self._reset_combined_state()
                self.view.combined_run_button.setEnabled(True)
                self.view.combined_abort_button.setEnabled(False)
                self._set_combined_config_enabled(True)
                self.is_busy = False
                self._set_run_control("idle")
                logger.error("[Combined] Run failed.")
                return
            # One sweep died but the queue is not empty: same offer as an
            # abort. Losing a channel should not cost the whole run.
            self._set_run_control("resume")
            self.view.combined_run_button.setEnabled(False)
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = True
            logger.error(
                "[Combined] Sweep failed; press Resume to continue with "
                "the rest of the queue.")
            return

        if self.manager.experiments.has_next():
            # Sweeps remain: offer Resume, exactly as after an abort, so the
            # operator can carry on with the rest instead of requeueing.
            self._set_run_control("resume")
            self.view.queue_button.setEnabled(False)
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = True
            # Deliberately no count here: this runs inside a failure handler,
            # and poking at the queue's internals to produce a nicer message is
            # exactly where a second exception would strand the UI again.
            logger.info("Sweeps remain in the queue — press Resume to continue.")
        else:
            self._finalize_run()
            self._return_to_idle()

    def on_queued(self):
        """Handle experiment queued state."""
        self.view.queue_button.setEnabled(False)
        self._set_run_control("abort")
        self.view.browser_widget.show_button.setEnabled(True)
        self.view.browser_widget.hide_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(True)

    def on_running(self):
        """Handle experiment running state."""
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(False)

    def update_analysis_panel(self):
        """Update the analysis panel with newly computed results."""
        try:
            browser = self.view.browser_widget.browser
            root = browser.invisibleRootItem()

            # First, collect all channels that have experiments
            all_channels = set()
            channel_directions = {}  # Store which directions exist per channel
            
            for i in range(root.childCount()):
                item = root.child(i)
                exp = self.manager.experiments.with_browser_item(item)
                if exp and hasattr(exp.procedure, 'active_channel'):
                    try:
                        channel = int(exp.procedure.active_channel)
                        all_channels.add(channel)
                        
                        # Check if this is a reverse experiment
                        if hasattr(exp.procedure, 'sweep_direction'):
                            direction = exp.procedure.sweep_direction
                            if channel not in channel_directions:
                                channel_directions[channel] = set()
                            channel_directions[channel].add(direction)
                    except (ValueError, TypeError):
                        pass

            # Determine if we're in single sweep mode (no reverse experiments)
            has_reverse = any("Reverse" in dirs for dirs in channel_directions.values())
            self.view.analysis_panel.set_single_sweep_mode(not has_reverse)

            # Reset analysis panel with all channels
            if all_channels:
                self.view.analysis_panel.reset_channels(
                    sorted(all_channels), JVProcedure.ANALYSIS_LABELS_UNITS
                )
                self._sync_channel_indicators(sorted(all_channels))

            # Clear the 'analysis_shown' flag on all items so they will be re-populated
            # after the panel has been rebuilt. This handles transitions from single
            # to dual sweep mode when reverse sweeps complete after forward sweeps.
            for i in range(root.childCount()):
                item = root.child(i)
                if hasattr(item, 'analysis_shown'):
                    del item.analysis_shown

            # Now process results
            for i in range(root.childCount()):
                item = root.child(i)
                experiment = self.manager.experiments.with_browser_item(item)

                if (experiment and hasattr(experiment.procedure, 'analysis_results') and
                        not hasattr(item, 'analysis_shown')):
                    results = experiment.procedure.analysis_results
                    if results:
                        for channel, metrics in results.items():
                            if isinstance(metrics, dict):
                                # Handle both single-direction keys ("Forward" or "Reverse")
                                # and dual-direction keys ({"Forward": ..., "Reverse": ...})
                                for direction, dir_metrics in metrics.items():
                                    self.view.analysis_panel.analysis({
                                        'Channel': channel,
                                        'Direction': direction,
                                        **dir_metrics
                                    })
                                    logger.debug(f"Analysis updated: Ch{channel} {direction}")
                            else:
                                # Legacy format: single metrics dict without direction nesting
                                self.view.analysis_panel.analysis({
                                    'Channel': channel,
                                    'Direction': 'Forward',
                                    **metrics
                                })
                    item.analysis_shown = True

            self.finished_experiment_count = root.childCount()

            # Auto-switch to the Channel Analysis view so the user sees results
            if hasattr(self.view, 'bottom_stack'):
                self.view.bottom_stack.setCurrentIndex(1)
                self.view.bottom_tab_bar.setCurrentIndex(1)

        except Exception as e:
            logger.error(f"Analysis update error: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------------
    # File Processing and Formatting
    # -------------------------------------------------------------------------

    def _merge_channel_files(self):
        """Combine temporary channel files into a single merged report."""
        if self.merged_data_written:
            return

        try:
            logger.info("Merging channel files...")
            
            # Group by channel number
            channel_data_map = {}
            analysis_summary = []
            experiment_params = None

            for key, file_path in self.experiment_files.items():
                if not os.path.exists(file_path):
                    continue

                # Parse key: format "channel_direction" (e.g., "1_forward")
                parts = key.split("_")
                channel_num = int(parts[0])
                direction = parts[1].capitalize()  # "Forward" or "Reverse"

                channel_data, channel_analysis, channel_params = self._parse_temp_file(file_path)

                if not experiment_params and channel_params:
                    experiment_params = channel_params

                if channel_analysis:
                    channel_analysis['Channel'] = f"{channel_num}_{direction}"
                    analysis_summary.append(channel_analysis)

                # Store data for combining
                if channel_num not in channel_data_map:
                    channel_data_map[channel_num] = {}
                channel_data_map[channel_num][direction] = channel_data

                try:
                    os.remove(file_path)
                except OSError:
                    pass

            # Combine data for each channel
            all_channel_dfs = []
            for channel_num, data in sorted(channel_data_map.items()):
                if 'Forward' in data and 'Reverse' in data:
                    formatted_df = self._combine_forward_reverse_data(channel_num, data['Forward'], data['Reverse'])
                elif 'Forward' in data:
                    formatted_df = self._format_channel_dataframe(channel_num, data['Forward'], {})
                elif 'Reverse' in data:
                    formatted_df = self._format_channel_dataframe(channel_num, data['Reverse'], {})
                else:
                    continue
                all_channel_dfs.append(formatted_df)

            if not all_channel_dfs:
                return

            final_df = pd.concat(all_channel_dfs, axis=1)
            self._write_formatted_report(self.merged_file_path, experiment_params,
                                        analysis_summary, final_df,
                                        notes_text=getattr(self, 'current_notes_text', ''),
                                        save_notes=getattr(self, 'current_save_notes', False))
            self.merged_data_written = True
            logger.info(f"Merged report saved: {self.merged_file_path}")

        except Exception as e:
            logger.error(f"Merge failed: {e}")
            import traceback
            traceback.print_exc()

    def _process_multi_files(self):
        """Merge forward and reverse files into a single file per channel."""
        try:
            logger.info("Merging forward/reverse files per channel...")
            
            # Group by channel number
            channel_files = {}
            for key, file_path in self.experiment_files.items():
                if not os.path.exists(file_path):
                    continue
                    
                if isinstance(key, str) and "_" in key:
                    parts = key.split("_")
                    channel = int(parts[0])
                    direction = parts[1]
                    channel_files.setdefault(channel, {})[direction] = (file_path, key)
                else:
                    # Fallback for legacy keys (should not happen)
                    channel = int(key)
                    channel_files.setdefault(channel, {})['single'] = (file_path, key)
            
            from solarjv_analyzer.config import TIMESTAMP_FORMAT
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT).replace(":", "-").replace(" ", "_")
            
            for channel, files in channel_files.items():
                final_path = None
                analysis_summary = []
                experiment_params = None
                combined_df = None
                
                if 'single' in files:
                    # Legacy single-file handling
                    file_path, key = files['single']
                    if os.path.exists(file_path):
                        channel_data, channel_analysis, channel_params = self._parse_temp_file(file_path)
                        if channel_analysis:
                            channel_analysis['Channel'] = f"{channel}"
                            analysis_summary.append(channel_analysis)
                        if not experiment_params and channel_params:
                            experiment_params = channel_params
                        combined_df = self._format_channel_dataframe(channel, channel_data, channel_analysis)
                        final_path = file_path.replace("_temp", "")
                        self.processed_files.add(key)
                
                elif 'forward' in files and 'reverse' in files:
                    # Dual sweep mode
                    forward_path, forward_key = files['forward']
                    reverse_path, reverse_key = files['reverse']
                    
                    forward_data, forward_analysis, forward_params = self._parse_temp_file(forward_path)
                    if forward_analysis:
                        forward_analysis['Channel'] = f"{channel}_Forward"
                        analysis_summary.append(forward_analysis)
                    if not experiment_params and forward_params:
                        experiment_params = forward_params
                    
                    reverse_data, reverse_analysis, reverse_params = self._parse_temp_file(reverse_path)
                    if reverse_analysis:
                        reverse_analysis['Channel'] = f"{channel}_Reverse"
                        analysis_summary.append(reverse_analysis)
                    
                    combined_df = self._combine_forward_reverse_data(channel, forward_data, reverse_data)
                    
                    base_dir = os.path.dirname(forward_path)
                    final_path = os.path.join(
                        base_dir, self._final_report_name(timestamp, channel))
                    
                    # Delete temp files after merging
                    for p in [forward_path, reverse_path]:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    
                    self.processed_files.add(forward_key)
                    self.processed_files.add(reverse_key)
                
                elif 'forward' in files or 'reverse' in files:
                    # One direction only. Forward-only is Single-Sweep-Mode;
                    # reverse-only is what aborting the forward sweep and
                    # resuming leaves behind, and it used to match no branch
                    # at all — the completed reverse sweep was dropped with a
                    # "No data to write" warning.
                    direction = 'forward' if 'forward' in files else 'reverse'
                    single_path, single_key = files[direction]

                    single_data, single_analysis, single_params = self._parse_temp_file(single_path)
                    if single_analysis:
                        single_analysis['Channel'] = f"{channel}"
                        analysis_summary.append(single_analysis)
                    if not experiment_params and single_params:
                        experiment_params = single_params

                    combined_df = self._format_channel_dataframe(channel, single_data, single_analysis)

                    # Create a clean output file name (remove the direction)
                    base_dir = os.path.dirname(single_path)
                    final_path = os.path.join(
                        base_dir, self._final_report_name(timestamp, channel))

                    try:
                        os.remove(single_path)
                    except OSError:
                        pass

                    self.processed_files.add(single_key)
                
                # Write the formatted report
                if final_path and combined_df is not None and not combined_df.empty:
                    self._write_formatted_report(final_path, experiment_params, analysis_summary, combined_df,
                                                notes_text=getattr(self, 'current_notes_text', ''),
                                                save_notes=getattr(self, 'current_save_notes', False))
                    logger.info(f"Formatted Channel {channel} -> {final_path}")
                else:
                    logger.warning(f"No data to write for Channel {channel}")
            
            logger.info(f"Multi-file formatting complete")
            
        except Exception as e:
            logger.error(f"File formatting failed: {e}")
            import traceback
            traceback.print_exc()

    def _combine_forward_reverse_data(self, channel_num, forward_df, reverse_df):
        """
        Combine forward and reverse data into a single multi-index DataFrame.
        
        Args:
            channel_num: Channel number
            forward_df: DataFrame with forward sweep data
            reverse_df: DataFrame with reverse sweep data
        
        Returns:
            pd.DataFrame: Combined DataFrame with both directions
        """
        if forward_df.empty and reverse_df.empty:
            return pd.DataFrame()
        
        data_map = {}
        
        # Process forward data
        if not forward_df.empty:
            curr_col = 'Current (A)' if 'Current (A)' in forward_df.columns else 'Current'
            volt_col = 'Voltage (V)' if 'Voltage (V)' in forward_df.columns else 'Voltage'
            
            if curr_col in forward_df.columns and volt_col in forward_df.columns:
                data_map[(channel_num, "Forward", 'V')] = forward_df[volt_col].values
                data_map[(channel_num, "Forward", 'J')] = forward_df[curr_col].values
        
        # Process reverse data
        if not reverse_df.empty:
            curr_col = 'Current (A)' if 'Current (A)' in reverse_df.columns else 'Current'
            volt_col = 'Voltage (V)' if 'Voltage (V)' in reverse_df.columns else 'Voltage'
            
            if curr_col in reverse_df.columns and volt_col in reverse_df.columns:
                data_map[(channel_num, "Reverse", 'V')] = reverse_df[volt_col].values
                data_map[(channel_num, "Reverse", 'J')] = reverse_df[curr_col].values
        
        if not data_map:
            return pd.DataFrame()
        
        # Align lengths (pad with NaN for uneven data)
        max_len = max((len(arr) for arr in data_map.values()), default=0)
        aligned_data = {}
        
        for key, arr in data_map.items():
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                aligned_data[key] = padded
            else:
                aligned_data[key] = arr
        
        # Create MultiIndex DataFrame
        multi_index = pd.MultiIndex.from_tuples(
            aligned_data.keys(), names=["channel", "direction", "value"]
        )
        final_df = pd.DataFrame(aligned_data)
        final_df.columns = multi_index
        
        return final_df

    def _write_formatted_report(self, filepath, parameters, analysis_summary, final_df,
                                notes_text='', save_notes=False):
        """
        Write a formatted report with experimental parameters, analysis, and data.

        The analysis summary will have column headers with units included.
        """
        try:
            with open(filepath, 'w', newline='') as f:
                # Write experimental parameters
                if parameters:
                    f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                    exclude_keys = {
                        "Parameter", "Parameters", "Procedure", "Active Channel",
                        "GPIB Address", "Measurement Range", "MUX Object"
                    }
                    filtered_params = [
                        (k, v) for k, v in parameters
                        if k not in exclude_keys and not k.startswith("Channel ")
                    ]
                    if filtered_params:
                        param_dict = dict(filtered_params)
                        param_df = pd.DataFrame([param_dict])
                        param_df.to_csv(f, index=False, sep=',')
                    f.write("\n")

                # Write analysis summary
                f.write("[[ ANALYSIS SUMMARY ]]\n")
                if analysis_summary:
                    summary_df = pd.DataFrame(analysis_summary)

                    # Ensure 'Channel' column exists
                    if 'Channel' not in summary_df.columns:
                        for col in summary_df.columns:
                            if 'channel' in col.lower():
                                summary_df.rename(columns={col: 'Channel'}, inplace=True)
                                break
                        else:
                            summary_df['Channel'] = range(1, len(summary_df) + 1)

                    # Reorder columns to put Channel first
                    cols = ['Channel'] + [c for c in summary_df.columns if c != 'Channel']
                    summary_df = summary_df[cols]

                    # Add units to metric column headers
                    metric_units = {
                        "EFF": "EFF (%)",
                        "FF": "FF (%)",
                        "Voc": "Voc (mV)",
                        "Jsc": "Jsc (mA/cm2)",
                        "Vmpp": "Vmpp (mV)",
                        "Jmpp": "Jmpp (mA/cm2)",
                        "Pmpp": "Pmpp (mW)",
                        "Isc": "Isc (A)",
                        "Rsh": "Rsh (Ohm)",
                        "Rs": "Rs (Ohm)",
                        "A": "Area (cm2)",
                        "Incd. Pwr": "Incd. Pwr (mW/cm2)",
                    }
                    summary_df.rename(columns=metric_units, inplace=True)
                    summary_df.to_csv(f, index=False, sep=',')
                else:
                    f.write("No analysis data available.\n")

                f.write("\n")

                # Optional notes
                if save_notes and notes_text.strip():
                    f.write("[[ NOTES ]]\n")
                    f.write(notes_text.strip())
                    f.write("\n\n")
                    
                f.write("[[ MEASUREMENT DATA ]]\n")

                if not final_df.empty:
                    final_df = final_df.round(6)
                    final_df.index = [''] * len(final_df)

                    header_ch = ["channel"] + [str(col[0]) for col in final_df.columns]
                    f.write(",".join(header_ch) + "\n")
                    header_dir = ["direction"] + [str(col[1]) for col in final_df.columns]
                    f.write(",".join(header_dir) + "\n")
                    header_type = ["value"] + [str(col[2]) for col in final_df.columns]
                    f.write(",".join(header_type) + "\n")

                    final_df.to_csv(f, header=False, index=True)

            logger.info(f"Formatted report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to write formatted report: {e}")
            raise

    def _parse_temp_file(self, filepath: str):
        """
        Parse a temporary measurement file.

        Returns:
            tuple: (dataframe, analysis_dict, parameters_list)
        """
        data_lines = []
        analysis_dict = {}
        parameters = []
        in_analysis = False

        with open(filepath, 'r') as f:
            for line in f:
                stripped = line.strip()

                if stripped == "[[ANALYSIS]]":
                    in_analysis = True
                    continue
                if stripped == "[[/ANALYSIS]]":
                    in_analysis = False
                    continue

                if in_analysis:
                    parts = stripped.split('\t')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        try:
                            analysis_dict[key] = float(parts[1])
                        except ValueError:
                            analysis_dict[key] = parts[1]
                else:
                    if stripped.startswith("#"):
                        content = stripped.lstrip("#").strip()
                        if content.endswith(":") and " " not in content:
                            continue
                        if ":" in content:
                            key, val = content.split(":", 1)
                            parameters.append((key.strip(), val.strip()))
                    elif stripped:
                        data_lines.append(line)

        from io import StringIO
        if data_lines:
            csv_data = StringIO("".join(data_lines))
            try:
                df = pd.read_csv(csv_data)
            except Exception as e:
                logger.warning(f"Failed to parse CSV data: {e}")
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        return df, analysis_dict, parameters

    def _format_channel_dataframe(self, channel_num, df, analysis_dict):
        """
        Format a channel's data into a multi-index DataFrame for merging.

        Args:
            channel_num: Channel number
            df: Raw DataFrame with Current (A) and Voltage (V) columns
            analysis_dict: Analysis metrics for this channel (may contain direction info)

        Returns:
            pd.DataFrame: Multi-index DataFrame with channel/direction/value levels
        """
        if df.empty:
            return pd.DataFrame()

        curr_col = 'Current (A)' if 'Current (A)' in df.columns else 'Current'
        volt_col = 'Voltage (V)' if 'Voltage (V)' in df.columns else 'Voltage'

        if curr_col not in df.columns:
            return pd.DataFrame()

        area = analysis_dict.get("Area", 1.0) if analysis_dict else 1.0
        df['J'] = (df[curr_col] / area) * 1000.0
        df['V'] = df[volt_col]

        voltages = df['V'].values
        data_map = {}

        if len(voltages) > 2:
            diff = np.diff(voltages)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            if len(sign_changes) > 0:
                split_idx = sign_changes[0] + 1
                is_increasing = voltages[1] > voltages[0]

                dir1 = "Forward" if is_increasing else "Reverse"
                dir2 = "Reverse" if is_increasing else "Forward"

                df1 = df.iloc[:split_idx].reset_index(drop=True)
                df2 = df.iloc[split_idx:].reset_index(drop=True)

                data_map = {
                    (channel_num, dir1, 'V'): df1['V'],
                    (channel_num, dir1, 'J'): df1['J'],
                    (channel_num, dir2, 'V'): df2['V'],
                    (channel_num, dir2, 'J'): df2['J'],
                }
            else:
                direction = "Forward"
                if analysis_dict and "Channel" in analysis_dict:
                    channel_label = analysis_dict.get("Channel", "")
                    if "_Reverse" in str(channel_label) or "_reverse" in str(channel_label):
                        direction = "Reverse"
                data_map = {
                    (channel_num, direction, 'V'): df['V'],
                    (channel_num, direction, 'J'): df['J']
                }
        else:
            direction = "Forward"
            if analysis_dict and "Channel" in analysis_dict:
                channel_label = analysis_dict.get("Channel", "")
                if "_Reverse" in str(channel_label) or "_reverse" in str(channel_label):
                    direction = "Reverse"
            data_map = {
                (channel_num, direction, 'V'): df['V'],
                (channel_num, direction, 'J'): df['J']
            }

        max_len = max((len(arr) for arr in data_map.values()), default=0)
        aligned_data = {}

        for key, arr in data_map.items():
            arr_values = arr.values
            if len(arr_values) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr_values)] = arr_values
                aligned_data[key] = padded
            else:
                aligned_data[key] = arr_values

        multi_index = pd.MultiIndex.from_tuples(
            aligned_data.keys(), names=["channel", "direction", "value"]
        )
        final_df = pd.DataFrame(aligned_data)
        final_df.columns = multi_index

        return final_df
