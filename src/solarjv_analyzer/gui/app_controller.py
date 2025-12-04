import logging
import os
import io
import tempfile
from datetime import datetime
import pyqtgraph as pg
import pandas as pd
import numpy as np

from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.experiment import Results

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.config import TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


class AppController:
    """
    Manages the application's state and orchestrates the experiment lifecycle.
    """

    def __init__(self, view):
        self.view = view
        self.finished_experiment_count = 0
        self.is_busy = False
        
        # State tracking for files
        self.experiment_files = {}  # Maps channel_num -> file_path
        self.is_single_file_mode = False
        self.processed_files = set() # Track which files we've already formatted

        # Initialize PyMeasure Manager
        self.manager = Manager(
            [self.view.plot_widget, self.view.log_widget, self.view.analysis_panel],
            self.view.browser_widget.browser,
            log_level=logging.INFO,
            parent=self.view
        )
        self._connect_manager_signals()
        
        self.view.abort_button.setEnabled(False)

    def _connect_manager_signals(self) -> None:
        self.manager.abort_returned.connect(self.on_abort_returned)
        self.manager.queued.connect(self.on_queued)
        self.manager.running.connect(self.on_running)
        self.manager.finished.connect(self.on_finished)
        self.manager.finished.connect(self.update_analysis_panel)

    def queue_experiment(self) -> None:
        """Collects parameters and queues the experiment."""
        if self.is_busy:
            logger.warning("Cannot queue: Operation in progress.")
            return
        self.is_busy = True
        
        self.view.update_save_directory()

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
        channels_selected = self.view.params_tab.get_selected_channels()
        
        if not channels_selected:
            logger.warning("No channels selected.")
            self.is_busy = False
            return

        self.experiment_files = {}
        self.processed_files = set()
        self.is_single_file_mode = file_params['single_file']

        self.view.analysis_panel.reset_channels(
            channels_selected, JVProcedure.ANALYSIS_LABELS_UNITS
        )

        sim_mode = file_params['simulation']
        
        try:
            self.view.instrument_manager.connect_keithley(simulation=sim_mode)
            self.view.instrument_manager.connect_mux(simulation=sim_mode)
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.view.update_instrument_lights()
            self.is_busy = False
            return

        self.view.update_instrument_lights()

        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        base, ext = os.path.splitext(file_params['filename'])
        directory = file_params['directory']

        if self.is_single_file_mode:
            self._queue_single_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )
        else:
            self._queue_multi_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )

    def _queue_single_file_experiment(self, directory, base, ext, timestamp, channels, params, sim):
        ch_list_str = "_".join(map(str, channels))
        self.merged_file_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_list_str}{ext}")
        self.merged_data_written = False
        
        merged_filename = os.path.basename(self.merged_file_path)
        logger.info(f"Single file mode: Will merge to {merged_filename}")
        
        for ch_num in channels:
            temp_file_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_num}_temp{ext}")
            self.experiment_files[ch_num] = temp_file_path
            
            single_ch_params = params.copy()
            for i in range(1, 7): single_ch_params[f'channel{i}'] = (i == ch_num)

            proc = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim,
                active_channel=ch_num,
                **single_ch_params
            )
            
            results = Results(proc, temp_file_path)
            proc.results = results
            
            display_name = f"Ch {ch_num} - {merged_filename}"
            experiment = self._new_experiment(results, display_name)
            self.manager.queue(experiment)

    def _queue_multi_file_experiment(self, directory, base, ext, timestamp, channels, params, sim):
        for ch_num in channels:
            full_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_num}{ext}")
            self.experiment_files[ch_num] = full_path

            single_ch_params = params.copy()
            for i in range(1, 7): single_ch_params[f'channel{i}'] = (i == ch_num)

            proc = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim,
                active_channel=ch_num,
                **single_ch_params
            )

            results = Results(proc, full_path)
            proc.results = results

            self.manager.queue(self._new_experiment(results))
            logger.info(f"Queued Ch {ch_num} to {os.path.basename(full_path)}")

    def _new_experiment(self, results: Results, display_filename: str = None) -> Experiment:
        browser = self.view.browser_widget.browser
        color = pg.intColor(browser.topLevelItemCount() % 8)
        curve = self.view.plot_widget.new_curve(results, color=color)
        browser_item = BrowserItem(results, color)
        if display_filename:
            browser_item.setText(1, display_filename)
        return Experiment(results, [curve], browser_item)

    # --- File Loading Logic ---

    def load_files(self, filenames: list) -> None:
        """Loads custom format files into the browser and plot."""
        logger.info(f"Loading {len(filenames)} files...")
        
        all_channels_found = []
        all_analysis_data = [] 
        newly_loaded_items = []
        
        for filename in filenames:
            try:
                experiments, file_analysis_data, channels = self._parse_and_load_single_file(filename)
                
                newly_loaded_items.extend(experiments)
                all_channels_found.extend(channels)
                all_analysis_data.extend(file_analysis_data)
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}", exc_info=True)
                QtWidgets.QMessageBox.warning(self.view, "Load Error", f"Failed to load {os.path.basename(filename)}\n{e}")

        all_active_channels = set()
        browser = self.view.browser_widget.browser
        root = browser.invisibleRootItem()
        
        for i in range(root.childCount()):
            item = root.child(i)
            exp = self.manager.experiments.with_browser_item(item)
            if exp and hasattr(exp.procedure, 'active_channel'):
                try:
                    ch = int(exp.procedure.active_channel)
                    all_active_channels.add(ch)
                except (ValueError, TypeError): pass
        
        if all_active_channels:
            self.view.analysis_panel.reset_channels(
                sorted(list(all_active_channels)), 
                JVProcedure.ANALYSIS_LABELS_UNITS
            )

        for analysis_entry in all_analysis_data:
            ch = analysis_entry.get('Channel')
            if ch:
                self.view.analysis_panel.analysis({'Channel': ch, **analysis_entry})

        if newly_loaded_items:
            first_exp = newly_loaded_items[0]
            for i in range(root.childCount()):
                item = root.child(i)
                if self.manager.experiments.with_browser_item(item) == first_exp:
                    item.setSelected(True)
                    # Trigger the selection logic manually for the first item
                    self.on_browser_selection_changed()
                    break

    def _parse_and_load_single_file(self, filename: str) -> tuple:
        """Parses a single file and creates experiments."""
        with open(filename, 'r') as f:
            content = f.read()
        
        blocks = content.split('[[')
        params_dict = {}
        analysis_data = []
        measurement_df = pd.DataFrame()
        
        for block in blocks:
            if not block.strip(): continue
            if "EXPERIMENTAL PARAMETERS" in block:
                lines = block.split(']]')[1].strip()
                if lines:
                    try:
                        p_df = pd.read_csv(io.StringIO(lines))
                        params_dict = p_df.to_dict(orient='records')[0]
                    except: pass
            elif "ANALYSIS SUMMARY" in block:
                lines = block.split(']]')[1].strip()
                if lines and "No analysis" not in lines:
                    a_df = pd.read_csv(io.StringIO(lines))
                    analysis_data = a_df.to_dict(orient='records')
            elif "MEASUREMENT DATA" in block:
                lines = block.split(']]')[1].strip()
                if lines:
                    measurement_df = pd.read_csv(io.StringIO(lines), header=[0, 1, 2])

        if measurement_df.empty:
            logger.warning(f"No measurement data found in {filename}")
            return [], [], []

        loaded_experiments = [] 
        loaded_channels = []
        display_name = os.path.basename(filename)
        area = float(params_dict.get("Device Area (cm^2)", 0.089))

        channels = measurement_df.columns.get_level_values(0).unique()
        
        for ch_str in channels:
            if not ch_str: continue 
            try: ch_num = int(ch_str)
            except ValueError: continue
            
            ch_df = measurement_df[ch_str]
            directions = ch_df.columns.get_level_values(0).unique()
            direction = "Forward" if "Forward" in directions else directions[0]
            data_subset = ch_df[direction]
            
            if 'V' not in data_subset.columns or 'J' not in data_subset.columns:
                continue

            plot_df = pd.DataFrame()
            plot_df["Voltage (V)"] = data_subset['V']
            plot_df["Current (A)"] = data_subset['J'] * area / 1000.0
            plot_df["Channel"] = ch_num
            plot_df["Time (s)"] = np.nan
            plot_df["Status"] = "Loaded"

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w')
            temp.write(",".join(plot_df.columns) + "\n")
            plot_df.to_csv(temp, index=False, header=False)
            temp.close()
            
            proc = JVProcedure()
            proc.active_channel = ch_num 
            
            results = Results(proc, temp.name)
            
            formatted_name = f"Ch {ch_num} - {display_name}"
            exp = self._new_experiment(results, display_filename=formatted_name)
            
            ch_metrics = next((a for a in analysis_data if a.get('Channel') == ch_num), None)
            if ch_metrics:
                exp.procedure.analysis_results = {ch_num: ch_metrics}
            
            self.manager.load(exp)
            loaded_experiments.append(exp)
            loaded_channels.append(ch_num)
            
        return loaded_experiments, analysis_data, loaded_channels

    # --- Browser Selection Handler ---

    def on_browser_selection_changed(self):
        """Called when user clicks an item in the browser."""
        try:
            items = self.view.browser_widget.browser.selectedItems()
            if not items: return
            
            item = items[0]
            experiment = self.manager.experiments.with_browser_item(item)
            
            if experiment:
                # 1. Update Analysis Panel if data exists
                if hasattr(experiment.procedure, 'analysis_results'):
                    results = experiment.procedure.analysis_results
                    if results:
                        for ch, metrics in results.items():
                            self.view.analysis_panel.analysis({'Channel': ch, **metrics})
                
                # 2. FIX: Switch the active tab to match the channel
                if hasattr(experiment.procedure, 'active_channel'):
                    try:
                        ch = int(experiment.procedure.active_channel)
                        self.view.analysis_panel.set_active_channel(ch)
                    except (ValueError, TypeError):
                        pass
                        
        except Exception as e:
            logger.error(f"Selection handler error: {e}")

    # --- Experiment Lifecycle Methods ---

    def on_finished(self) -> None:
        """Handles post-experiment logic."""
        logger.info("Experiment finished.")
        
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
                logger.error(f"Error in file post-processing: {e}", exc_info=True)
            
            self._disconnect_instruments()
            self.view.queue_button.setEnabled(True)
            self.is_busy = False
        else:
            self.view.queue_button.setEnabled(False)
            self.is_busy = True
        
        self.view.update_instrument_lights()

    def abort_experiment(self) -> None:
        """Aborts the currently running experiment."""
        logger.info("Abort requested.")
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(False)
        self.view.abort_button.setText("Aborting...")
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.resume_experiment)
        self.manager.abort()

    def resume_experiment(self) -> None:
        """Resumes the experiment queue."""
        logger.info("Resuming.")
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

    def clear_experiments(self) -> None:
        self.manager.clear()
        self.finished_experiment_count = 0

    def _disconnect_instruments(self) -> None:
        try:
            self.view.instrument_manager.disconnect_keithley()
            self.view.instrument_manager.disconnect_mux()
        except Exception: pass
        finally: self.view.update_instrument_lights()

    def on_abort_returned(self) -> None:
        self._disconnect_instruments()
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

    def on_queued(self) -> None:
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.abort_button.setText("Abort")
        try: self.view.abort_button.clicked.disconnect()
        except TypeError: pass
        self.view.abort_button.clicked.connect(self.abort_experiment)
        self.view.browser_widget.show_button.setEnabled(True)
        self.view.browser_widget.hide_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(True)

    def on_running(self) -> None:
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(False)

    def update_analysis_panel(self) -> None:
        try:
            browser = self.view.browser_widget.browser
            root = browser.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                experiment = self.manager.experiments.with_browser_item(item)
                if (experiment and experiment.procedure.analysis_results and 
                    not hasattr(item, 'analysis_shown')):
                    for ch, metrics in experiment.procedure.analysis_results.items():
                        self.view.analysis_panel.analysis({'Channel': ch, **metrics})
                    item.analysis_shown = True
            self.finished_experiment_count = root.childCount()
        except Exception as e:
            logger.error(f"Analysis update error: {e}", exc_info=True)

    # --- File Processing Methods ---

    def _merge_channel_files(self) -> None:
        """Combines temp files into one merged report."""
        if self.merged_data_written: return

        try:
            logger.info("Merging files with advanced formatting...")
            all_channel_dfs = []
            analysis_summary = []
            exp_parameters = [] 

            for ch_num, file_path in sorted(self.experiment_files.items()):
                if not os.path.exists(file_path): continue

                ch_data, ch_analysis, ch_params = self._parse_temp_file(file_path)
                
                if not exp_parameters and ch_params:
                    exp_parameters = ch_params

                if ch_analysis:
                    ch_analysis['Channel'] = ch_num
                    analysis_summary.append(ch_analysis)

                formatted_df = self._format_channel_dataframe(ch_num, ch_data, ch_analysis)
                all_channel_dfs.append(formatted_df)

                try: os.remove(file_path)
                except OSError: pass

            if not all_channel_dfs: return

            final_df = pd.concat(all_channel_dfs, axis=1)
            self._write_formatted_report(self.merged_file_path, exp_parameters, analysis_summary, final_df)
            
            self.merged_data_written = True
            logger.info(f"Saved merged report: {self.merged_file_path}")

        except Exception as e:
            logger.error(f"Merge failed: {e}", exc_info=True)

    def _process_multi_files(self) -> None:
        """Rewrites individual channel files with new formatting."""
        try:
            logger.info("Formatting individual channel files...")
            
            for ch_num, file_path in sorted(self.experiment_files.items()):
                if file_path in self.processed_files: continue
                if not os.path.exists(file_path): continue

                ch_data, ch_analysis, ch_params = self._parse_temp_file(file_path)
                
                analysis_summary = []
                if ch_analysis:
                    ch_analysis['Channel'] = ch_num
                    analysis_summary.append(ch_analysis)
                
                formatted_df = self._format_channel_dataframe(ch_num, ch_data, ch_analysis)
                
                self._write_formatted_report(file_path, ch_params, analysis_summary, formatted_df)
                self.processed_files.add(file_path)
                
                logger.info(f"Formatted file: {file_path}")

        except Exception as e:
            logger.error(f"File formatting failed: {e}", exc_info=True)

    def _write_formatted_report(self, filepath, parameters, analysis_summary, final_df):
        with open(filepath, 'w', newline='') as f:
            if parameters:
                f.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
                exclude_keys = {
                    "Parameter", "Parameters", "Procedure", 
                    "Active Channel", "GPIB Address", 
                    "Measurement Range", "MUX Object"
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

                header_ch = ["channel"] + [str(col[0]) for col in final_df.columns]
                f.write(",".join(header_ch) + "\n")
                header_dir = ["direction"] + [str(col[1]) for col in final_df.columns]
                f.write(",".join(header_dir) + "\n")
                header_type = ["value"] + [str(col[2]) for col in final_df.columns]
                f.write(",".join(header_type) + "\n")

                final_df.to_csv(f, header=False, index=True)

    def _parse_temp_file(self, filepath: str):
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
                        except ValueError: pass
                else:
                    if stripped.startswith("#"):
                        content = stripped.lstrip("#").strip()
                        if content.endswith(":") and " " not in content: continue
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

    def _format_channel_dataframe(self, ch_num, df, analysis_dict):
        if df.empty: return pd.DataFrame()

        curr_col = 'Current (A)' if 'Current (A)' in df.columns else 'Current'
        volt_col = 'Voltage (V)' if 'Voltage (V)' in df.columns else 'Voltage'
        
        if curr_col not in df.columns: return pd.DataFrame()

        area = analysis_dict.get("A", 1.0) if analysis_dict else 1.0
        df['J'] = (df[curr_col] / area) * 1000.0
        df['V'] = df[volt_col]

        v = df['V'].values
        data_map = {}
        
        if len(v) > 2:
            diff = np.diff(v)
            sign_changes = np.where(np.diff(np.sign(diff)))[0]
            
            if len(sign_changes) > 0:
                split_idx = sign_changes[0] + 1
                is_increasing = (v[1] > v[0])
                dir1 = "Forward" if is_increasing else "Reverse"
                dir2 = "Reverse" if is_increasing else "Forward"

                df1 = df.iloc[:split_idx].reset_index(drop=True)
                df2 = df.iloc[split_idx:].reset_index(drop=True)
                
                data_map = {
                    (ch_num, dir1, 'V'): df1['V'], (ch_num, dir1, 'J'): df1['J'],
                    (ch_num, dir2, 'V'): df2['V'], (ch_num, dir2, 'J'): df2['J'],
                }
            else:
                direction = "Forward" 
                data_map = { (ch_num, direction, 'V'): df['V'], (ch_num, direction, 'J'): df['J'] }
        else:
            data_map = { (ch_num, "Forward", 'V'): df['V'], (ch_num, "Forward", 'J'): df['J'] }

        max_len = max((len(s) for s in data_map.values()), default=0)
        aligned_data = {}
        for k, arr in data_map.items():
            arr = arr.values
            if len(arr) < max_len:
                padded = np.full(max_len, np.nan)
                padded[:len(arr)] = arr
                aligned_data[k] = padded
            else:
                aligned_data[k] = arr

        multi_index = pd.MultiIndex.from_tuples(aligned_data.keys(), names=["channel", "direction", "value"])
        final_df = pd.DataFrame(aligned_data)
        final_df.columns = multi_index
        return final_df