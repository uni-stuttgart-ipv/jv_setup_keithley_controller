"""
Application Controller for JV Analyzer

Manages the experiment lifecycle, file handling, and coordination between
the PyMeasure Manager and the GUI components.
"""

import logging
import os
import io
import tempfile
from datetime import datetime
from PyQt5 import QtCore

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.experiment import Results

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.config import TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


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
            [self.view.plot_widget, self.view.log_widget, self.view.analysis_panel],
            self.view.browser_widget.browser,
            log_level=logging.INFO,
            parent=self.view
        )
        self._connect_manager_signals()

        # Channel color mapping for consistent colors across forward and reverse curves
        self.CHANNEL_COLORS = {
            1: pg.mkColor('#0984e3'),  # Blue
            2: pg.mkColor('#00b894'),  # Green
            3: pg.mkColor('#e17055'),  # Orange
            4: pg.mkColor('#a29bfe'),  # Purple
            5: pg.mkColor('#fdcb6e'),  # Yellow
            6: pg.mkColor('#e84393'),  # Pink
        }

        self.view.abort_button.setEnabled(False)

    # -------------------------------------------------------------------------
    # Signal Connections
    # -------------------------------------------------------------------------

    def _connect_manager_signals(self):
        """Connect PyMeasure Manager signals to controller handlers."""
        self.manager.abort_returned.connect(self.on_abort_returned)
        self.manager.queued.connect(self.on_queued)
        self.manager.running.connect(self.on_running)
        self.manager.finished.connect(self.on_finished)
        self.manager.finished.connect(self.update_analysis_panel)

    # -------------------------------------------------------------------------
    # Experiment Queueing
    # -------------------------------------------------------------------------

    def queue_experiment(self):
        """
        Collect parameters from UI and queue experiments for selected channels.

        Handles both single-file (merged) and multi-file output modes.
        Preserves existing channel data in the analysis panel.
        """
        if self.is_busy:
            logger.warning("Cannot queue: Operation in progress.")
            return
        self.is_busy = True

        # Collect parameters from UI tabs
        params_dict = self.view.params_tab.get_parameters()
        analysis_dict = self.view.analysis_settings_tab.get_parameters()
        instr_dict = self.view.instr_tab.get_parameters()

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

        # Preserve existing channel data in the analysis panel
        self._preserve_existing_channel_data(selected_channels)

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

        # Generate file paths with timestamp
        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        base, ext = os.path.splitext(file_params['filename'])
        directory = file_params['directory']

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

    def _preserve_existing_channel_data(self, new_channels):
        """
        Preserve analysis data for channels not being re-measured.
        """
        # Collect existing channels and experiments
        existing_channels = set()
        existing_experiments = []
        root = self.view.browser_widget.browser.invisibleRootItem()

        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp:
                existing_experiments.append(exp)
                if hasattr(exp.procedure, 'active_channel'):
                    try:
                        existing_channels.add(int(exp.procedure.active_channel))
                    except (ValueError, TypeError):
                        pass

        # Combine old and new channels
        all_channels = sorted(existing_channels.union(set(new_channels)))
        self.view.analysis_panel.reset_channels(all_channels, JVProcedure.ANALYSIS_LABELS_UNITS)

        # Restore data for channels not being measured
        for exp in existing_experiments:
            if hasattr(exp.procedure, 'analysis_results') and exp.procedure.analysis_results:
                results = exp.procedure.analysis_results
                for channel, metrics in results.items():
                    if channel not in new_channels:
                        if isinstance(metrics, dict):
                            # Handle both single-direction keys ("Forward" or "Reverse")
                            # and dual-direction keys ({"Forward": ..., "Reverse": ...})
                            for direction, dir_metrics in metrics.items():
                                self.view.analysis_panel.analysis({
                                    'Channel': channel,
                                    'Direction': direction,
                                    **dir_metrics
                                })
                        else:
                            # Legacy format: single metrics dict without direction nesting
                            self.view.analysis_panel.analysis({
                                'Channel': channel,
                                'Direction': 'Forward',
                                **metrics
                            })

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

    def load_files(self, filenames: list):
        """
        Load previously saved measurement files into the browser and plot.

        Args:
            filenames: List of file paths to load
        """
        logger.info(f"Loading {len(filenames)} file(s)")

        all_channels = []
        newly_loaded_items = []

        for filename in filenames:
            try:
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
                        "Jsc (mA/cm2)": "Jsc", "Vmax (mV)": "Vmax",
                        "Jmax (mA/cm2)": "Jmax", "Isc (A)": "Isc",
                        "Rsh (Ohm)": "Rsh", "Rs (Ohm)": "Rs",
                        "Area (cm2)": "A", "Incd. Pwr (mW/cm2)": "Incd. Pwr",
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
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode='w',
                    prefix=f"ch{channel_num}_{direction}_"
                )
                temp_file.write("Channel,Voltage (V),Current (A),Time (s),Status\n")
                plot_df.to_csv(temp_file, index=False, header=False)
                temp_file.close()

                logger.debug(f"Loaded {len(plot_df)} points for Ch{channel_num} {direction}")

                procedure = JVProcedure()
                procedure.active_channel = channel_num
                procedure.sweep_direction = direction

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
            self.view.abort_button.setEnabled(False)
            self.view.abort_button.setText("Abort")
            self.view.browser_widget.clear_button.setEnabled(True)
            self.view.browser_widget.show_button.setEnabled(True)
            self.view.browser_widget.hide_button.setEnabled(True)

            # Merge / process files when all sweeps are done
            try:
                if self.is_single_file_mode:
                    self._merge_channel_files()
                else:
                    self._process_multi_files()
            except Exception as e:
                logger.error(f"File post-processing error: {e}")

            self._disconnect_instruments()
            self.view.queue_button.setEnabled(True)
            self.is_busy = False
        else:
            # More experiments in queue – keep instruments connected
            self.view.queue_button.setEnabled(False)
            self.is_busy = True

    def abort_experiment(self):
        """Abort the currently running experiment."""
        logger.info("Abort requested")
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(False)
        self.view.abort_button.setText("Aborting...")
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.resume_experiment)
        self.manager.abort()

    def resume_experiment(self):
        """Resume the experiment queue after an abort."""
        logger.info("Resuming experiment queue")

        # Ensure output is off before resuming
        try:
            self.view.instrument_manager.keithley.write(":OUTP OFF")
            self.view.instrument_manager.keithley.write(":ABOR")
        except Exception:
            pass

        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setText("Abort")
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.abort_experiment)

        if self.manager.experiments.has_next():
            self.manager.resume()
        else:
            self.view.abort_button.setEnabled(False)
            self.view.queue_button.setEnabled(True)
            self.is_busy = False

    def clear_experiments(self):
        """Clear all experiments from the manager."""
        self.manager.clear()
        self.finished_experiment_count = 0

    def _disconnect_instruments(self):
        """Disconnect from hardware instruments."""
        try:
            self.view.instrument_manager.disconnect_keithley()
            self.view.instrument_manager.disconnect_mux()
        except Exception:
            pass
        finally:
            self.view.update_instrument_lights()

    def on_abort_returned(self):
        """Handle post-abort state - instruments remain connected."""
        # Note: Instruments remain connected to allow resume functionality
        if self.manager.experiments.has_next():
            self.view.abort_button.setText("Resume")
            self.view.abort_button.setEnabled(True)
            self.view.queue_button.setEnabled(False)
            self.is_busy = True
        else:
            self.view.abort_button.setText("Abort")
            self.view.abort_button.setEnabled(False)
            self.view.queue_button.setEnabled(True)
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = False

    def on_queued(self):
        """Handle experiment queued state."""
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.abort_button.setText("Abort")
        try:
            self.view.abort_button.clicked.disconnect()
        except TypeError:
            pass
        self.view.abort_button.clicked.connect(self.abort_experiment)
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
                    final_path = os.path.join(base_dir, f"Output_{timestamp}_ch{channel}.csv")
                    
                    # Delete temp files after merging
                    for p in [forward_path, reverse_path]:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    
                    self.processed_files.add(forward_key)
                    self.processed_files.add(reverse_key)
                
                elif 'forward' in files:
                    # Single sweep forward mode
                    forward_path, forward_key = files['forward']
                    
                    forward_data, forward_analysis, forward_params = self._parse_temp_file(forward_path)
                    if forward_analysis:
                        forward_analysis['Channel'] = f"{channel}"
                        analysis_summary.append(forward_analysis)
                    if not experiment_params and forward_params:
                        experiment_params = forward_params
                    
                    # Format with multi-index including Forward direction
                    combined_df = self._format_channel_dataframe(channel, forward_data, forward_analysis)
                    
                    # Create a clean output file name (remove "_forward")
                    base_dir = os.path.dirname(forward_path)
                    final_path = os.path.join(base_dir, f"Output_{timestamp}_ch{channel}.csv")
                    
                    # Delete the temp forward file
                    try:
                        os.remove(forward_path)
                    except OSError:
                        pass
                    
                    self.processed_files.add(forward_key)
                
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
                        "Vmax": "Vmax (mV)",
                        "Jmax": "Jmax (mA/cm2)",
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