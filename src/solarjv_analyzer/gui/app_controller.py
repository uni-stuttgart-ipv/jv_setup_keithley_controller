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
        sim_mode = file_params['simulation']
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

        Args:
            new_channels: List of channel numbers being queued for measurement
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
                for channel, metrics in exp.procedure.analysis_results.items():
                    if channel not in new_channels:
                        self.view.analysis_panel.analysis({'Channel': channel, **metrics})

    def _queue_single_file_experiment(self, directory, base, ext, timestamp,
                                      channels, params, sim_mode):
        """
        Queue experiments for single-file output mode.

        Each channel writes to a temporary file; files are merged later.
        """
        channel_list_str = "_".join(map(str, channels))
        self.merged_file_path = os.path.join(
            directory, f"{base}_{timestamp}_ch{channel_list_str}{ext}"
        )
        self.merged_data_written = False

        merged_filename = os.path.basename(self.merged_file_path)
        logger.info(f"Single file mode: merging to {merged_filename}")

        for channel_num in channels:
            temp_file_path = os.path.join(
                directory, f"{base}_{timestamp}_ch{channel_num}_temp{ext}"
            )
            self.experiment_files[channel_num] = temp_file_path

            # Set channel-specific parameters
            single_ch_params = params.copy()
            for i in range(1, 7):
                single_ch_params[f'channel{i}'] = (i == channel_num)

            procedure = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim_mode,
                active_channel=channel_num,
                check_errors_between_points=False,
                **single_ch_params
            )

            results = Results(procedure, temp_file_path)
            procedure.results = results

            display_name = f"Ch {channel_num} - {merged_filename}"
            experiment = self._create_experiment(results, display_name)
            self.manager.queue(experiment)

    def _queue_multi_file_experiment(self, directory, base, ext, timestamp,
                                     channels, params, sim_mode):
        """
        Queue experiments for multi-file output mode.

        Each channel writes directly to its own final file.
        """
        for channel_num in channels:
            file_path = os.path.join(directory, f"{base}_{timestamp}_ch{channel_num}{ext}")
            self.experiment_files[channel_num] = file_path

            # Set channel-specific parameters
            single_ch_params = params.copy()
            for i in range(1, 7):
                single_ch_params[f'channel{i}'] = (i == channel_num)

            procedure = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim_mode,
                active_channel=channel_num,
                check_errors_between_points=False,
                **single_ch_params
            )

            results = Results(procedure, file_path)
            procedure.results = results

            experiment = self._create_experiment(results)
            self.manager.queue(experiment)
            logger.info(f"Queued Channel {channel_num} to {os.path.basename(file_path)}")

    def _create_experiment(self, results: Results, display_filename: str = None) -> Experiment:
        """
        Create an Experiment object with associated plot curve and browser item.

        Args:
            results: PyMeasure Results object
            display_filename: Optional custom filename for browser display

        Returns:
            Experiment: Configured experiment ready for queueing
        """
        browser = self.view.browser_widget.browser
        color = pg.intColor(browser.topLevelItemCount() % 8)
        curve = self.view.plot_widget.new_curve(results, color=color)

        browser_item = BrowserItem(results, color)
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

        if active_channels:
            self.view.analysis_panel.reset_channels(
                sorted(active_channels), JVProcedure.ANALYSIS_LABELS_UNITS
            )

        # Restore analysis data for all experiments
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'analysis_results'):
                for channel, metrics in exp.procedure.analysis_results.items():
                    self.view.analysis_panel.analysis({'Channel': channel, **metrics})

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
        Parse a saved file and create experiment objects.

        Args:
            filename: Path to the saved measurement file

        Returns:
            tuple: (experiments, analysis_data, channels)
        """
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
        area = float(params_dict.get("Device Area (cm^2)", 0.089))

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
            direction = "Forward" if "Forward" in directions else directions[0]
            data_subset = channel_df[direction]

            if 'V' not in data_subset.columns or 'J' not in data_subset.columns:
                continue

            # Create plot data
            plot_df = pd.DataFrame()
            plot_df["Voltage (V)"] = data_subset['V']
            plot_df["Current (A)"] = data_subset['J'] * area / 1000.0
            plot_df["Channel"] = channel_num
            plot_df["Time (s)"] = np.nan
            plot_df["Status"] = "Loaded"

            # Write to temporary CSV
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
            temp_file.write(",".join(plot_df.columns) + "\n")
            plot_df.to_csv(temp_file, index=False, header=False)
            temp_file.close()

            # Create procedure and results
            procedure = JVProcedure()
            procedure.active_channel = channel_num

            results = Results(procedure, temp_file.name)

            display_filename = f"Ch {channel_num} - {display_name}"
            experiment = self._create_experiment(results, display_filename)

            # Restore analysis data if available
            channel_metrics = next(
                (a for a in analysis_data if a.get('Channel') == channel_num), None
            )
            if channel_metrics:
                experiment.procedure.analysis_results = {channel_num: channel_metrics}

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

            # Update analysis panel with stored results
            if hasattr(experiment.procedure, 'analysis_results'):
                results = experiment.procedure.analysis_results
                if results:
                    for channel, metrics in results.items():
                        self.view.analysis_panel.analysis({'Channel': channel, **metrics})

            # Switch to the active channel tab
            if hasattr(experiment.procedure, 'active_channel'):
                try:
                    channel = int(experiment.procedure.active_channel)
                    self.view.analysis_panel.set_active_channel(channel)
                except (ValueError, TypeError):
                    pass

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
            self.view.queue_button.setEnabled(False)
            self.is_busy = True

        self.view.update_instrument_lights()

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

            for i in range(root.childCount()):
                item = root.child(i)
                experiment = self.manager.experiments.with_browser_item(item)

                if (experiment and experiment.procedure.analysis_results and
                        not hasattr(item, 'analysis_shown')):
                    for channel, metrics in experiment.procedure.analysis_results.items():
                        self.view.analysis_panel.analysis({'Channel': channel, **metrics})
                    item.analysis_shown = True

            self.finished_experiment_count = root.childCount()

        except Exception as e:
            logger.error(f"Analysis update error: {e}")

    # -------------------------------------------------------------------------
    # File Processing and Formatting
    # -------------------------------------------------------------------------

    def _merge_channel_files(self):
        """Combine temporary channel files into a single merged report."""
        if self.merged_data_written:
            return

        try:
            logger.info("Merging channel files...")
            all_channel_dfs = []
            analysis_summary = []
            experiment_params = []

            for channel_num, file_path in sorted(self.experiment_files.items()):
                if not os.path.exists(file_path):
                    continue

                channel_data, channel_analysis, channel_params = self._parse_temp_file(file_path)

                if not experiment_params and channel_params:
                    experiment_params = channel_params

                if channel_analysis:
                    channel_analysis['Channel'] = channel_num
                    analysis_summary.append(channel_analysis)

                formatted_df = self._format_channel_dataframe(channel_num, channel_data, channel_analysis)
                all_channel_dfs.append(formatted_df)

                try:
                    os.remove(file_path)
                except OSError:
                    pass

            if not all_channel_dfs:
                return

            final_df = pd.concat(all_channel_dfs, axis=1)
            self._write_formatted_report(self.merged_file_path, experiment_params,
                                         analysis_summary, final_df)

            self.merged_data_written = True
            logger.info(f"Merged report saved: {self.merged_file_path}")

        except Exception as e:
            logger.error(f"Merge failed: {e}")

    def _process_multi_files(self):
        """Format individual channel files (multi-file mode)."""
        try:
            logger.info("Formatting individual channel files...")

            for channel_num, file_path in sorted(self.experiment_files.items()):
                if file_path in self.processed_files or not os.path.exists(file_path):
                    continue

                channel_data, channel_analysis, channel_params = self._parse_temp_file(file_path)

                analysis_summary = []
                if channel_analysis:
                    channel_analysis['Channel'] = channel_num
                    analysis_summary.append(channel_analysis)

                formatted_df = self._format_channel_dataframe(channel_num, channel_data, channel_analysis)
                self._write_formatted_report(file_path, channel_params, analysis_summary, formatted_df)

                self.processed_files.add(file_path)
                logger.info(f"Formatted: {file_path}")

        except Exception as e:
            logger.error(f"File formatting failed: {e}")

    def _write_formatted_report(self, filepath, parameters, analysis_summary, final_df):
        """
        Write a formatted report with experimental parameters, analysis, and data.

        Args:
            filepath: Output file path
            parameters: List of (key, value) parameter tuples
            analysis_summary: List of analysis dictionaries
            final_df: Multi-index DataFrame with measurement data
        """
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
                cols = ['Channel'] + [c for c in summary_df.columns if c != 'Channel']
                summary_df = summary_df[cols]
                summary_df.to_csv(f, index=False, sep=',')
            else:
                f.write("No analysis data available.\n")

            f.write("\n")
            f.write("[[ MEASUREMENT DATA ]]\n")

            if not final_df.empty:
                final_df = final_df.round(6)
                final_df.index = [''] * len(final_df)

                # Multi-level header
                header_ch = ["channel"] + [str(col[0]) for col in final_df.columns]
                f.write(",".join(header_ch) + "\n")
                header_dir = ["direction"] + [str(col[1]) for col in final_df.columns]
                f.write(",".join(header_dir) + "\n")
                header_type = ["value"] + [str(col[2]) for col in final_df.columns]
                f.write(",".join(header_type) + "\n")

                final_df.to_csv(f, header=False, index=True)

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
                        try:
                            analysis_dict[parts[0]] = float(parts[1])
                        except ValueError:
                            pass
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
            except Exception:
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
            analysis_dict: Analysis metrics for this channel

        Returns:
            pd.DataFrame: Multi-index DataFrame with channel/direction/value levels
        """
        if df.empty:
            return pd.DataFrame()

        # Identify current and voltage columns
        curr_col = 'Current (A)' if 'Current (A)' in df.columns else 'Current'
        volt_col = 'Voltage (V)' if 'Voltage (V)' in df.columns else 'Voltage'

        if curr_col not in df.columns:
            return pd.DataFrame()

        area = analysis_dict.get("A", 1.0) if analysis_dict else 1.0
        df['J'] = (df[curr_col] / area) * 1000.0
        df['V'] = df[volt_col]

        voltages = df['V'].values
        data_map = {}

        # Detect sweep direction change for forward/reverse splitting
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
                data_map = {
                    (channel_num, direction, 'V'): df['V'],
                    (channel_num, direction, 'J'): df['J']
                }
        else:
            data_map = {
                (channel_num, "Forward", 'V'): df['V'],
                (channel_num, "Forward", 'J'): df['J']
            }

        # Align lengths (pad with NaN for uneven data)
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