import logging
import os
from datetime import datetime
import pyqtgraph as pg

from pymeasure.display.manager import Manager, Experiment
from pymeasure.display.browser import BrowserItem
from pymeasure.experiment import Results

from solarjv_analyzer.procedures.jv_procedure import JVProcedure
from solarjv_analyzer.config import TIMESTAMP_FORMAT

logger = logging.getLogger(__name__)


class AppController:
    """
    Manages the application's state and orchestrates the experiment lifecycle.
    
    This controller handles all experiment operations including queuing, running,
    aborting, and managing the UI state during these operations. It coordinates
    between the view components and the PyMeasure experiment manager.
    """

    def __init__(self, view):
        """
        Initializes the AppController.

        Args:
            view: A reference to the main JVAnalyzerWindow instance.
        """
        self.view = view
        self.finished_experiment_count = 0
        self.is_busy = False  # State lock for managing UI

        # Initialize PyMeasure Manager with view components
        self.manager = Manager(
            [self.view.plot_widget, self.view.log_widget, self.view.analysis_panel],
            self.view.browser_widget.browser,
            log_level=logging.INFO,
            parent=self.view
        )
        self._connect_manager_signals()
        
        # Set initial button state
        self.view.abort_button.setEnabled(False)

    def _connect_manager_signals(self) -> None:
        """Connects signals from the PyMeasure Manager to controller methods."""
        self.manager.abort_returned.connect(self.on_abort_returned)
        self.manager.queued.connect(self.on_queued)
        self.manager.running.connect(self.on_running)
        self.manager.finished.connect(self.on_finished)
        self.manager.finished.connect(self.update_analysis_panel)

    def queue_experiment(self) -> None:
        """
        Gathers settings from the view and queues experiments in the manager.
        
        This method:
        - Validates that no operation is already in progress
        - Collects parameters from all UI tabs
        - Connects to instruments
        - Queues experiments based on single-file or multi-file mode
        """
        # Prevent queueing if already busy
        if self.is_busy:
            logger.warning("Cannot queue experiment: An operation is already in progress or paused.")
            return
        self.is_busy = True  # Lock the state
        
        self.view.update_save_directory()

        # Collect parameters from all UI tabs
        procedure_params = {
            'user_name': self.view.username,
            **self.view.params_tab.get_parameters(),
            **self.view.instr_tab.get_parameters(),
            **self.view.analysis_settings_tab.get_parameters(),
        }

        file_params = self.view.file_panel.get_parameters()
        channels_selected = self.view.params_tab.get_selected_channels()
        
        # Validate channel selection
        if not channels_selected:
            logger.warning("No channels selected for measurement.")
            self.is_busy = False  # Release lock if no channels selected
            return

        # Initialize analysis panel with selected channels
        self.view.analysis_panel.reset_channels(
            channels_selected, JVProcedure.ANALYSIS_LABELS_UNITS
        )

        sim_mode = file_params['simulation']
        
        # Connect instruments before queuing experiments
        try:
            self.view.instrument_manager.connect_keithley(simulation=sim_mode)
            self.view.instrument_manager.connect_mux(simulation=sim_mode)
        except Exception as e:
            logger.error(f"Failed to connect to instruments: {e}")
            self.view.update_instrument_lights()
            self.is_busy = False  # Release lock on connection failure
            return  # Stop if connection fails

        self.view.update_instrument_lights()

        # Generate timestamp for filenames
        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        base, ext = os.path.splitext(file_params['filename'])
        directory = file_params['directory']

        # Queue experiments based on file mode selection
        if file_params['single_file']:
            self._queue_single_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )
        else:
            self._queue_multi_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )

    def _queue_single_file_experiment(self, directory: str, base: str, ext: str, timestamp: str, 
                                    channels: list, params: dict, sim: bool) -> None:
        """
        Queues multiple experiments that will be merged into a single file after completion.
        
        This method:
        - Creates temporary files for each channel experiment
        - Queues individual experiments for each channel
        - Sets up tracking for later file merging
        
        Args:
            directory: Output directory for files
            base: Base filename without extension
            ext: File extension
            timestamp: Timestamp string for filename
            channels: List of channel numbers to measure
            params: Procedure parameters dictionary
            sim: Simulation mode flag
        """
        # Create the final merged file path
        ch_list_str = "_".join(map(str, channels))
        self.merged_file_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_list_str}{ext}")
        
        # Initialize tracking variables for file merging
        self.channels_to_merge = channels.copy()
        self.merged_data_written = False
        self.temp_files = {}  # Store temp file paths by channel
        
        logger.info(f"Single file mode: Experiments will be merged into {self.merged_file_path}")
        
        # Get the merged filename for display
        merged_filename = os.path.basename(self.merged_file_path)
        
        # Queue one experiment per channel with temporary files
        for ch_num in channels:
            # Create unique temporary file for each channel
            temp_file_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_num}_temp{ext}")
            self.temp_files[ch_num] = temp_file_path
            
            # Configure parameters for this specific channel
            single_ch_params = params.copy()
            for i in range(1, 7):
                single_ch_params[f'channel{i}'] = (i == ch_num)

            # Create procedure instance
            proc = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim,
                active_channel=ch_num,
                **single_ch_params
            )

            # Create results with temporary file
            results = Results(proc, temp_file_path)
            
            # Link results to procedure
            proc.results = results

            # Create experiment with custom display name
            experiment = self._new_experiment(results, merged_filename)
            
            # Queue the experiment
            self.manager.queue(experiment)
            logger.info(f"Queued experiment for channel {ch_num} (displayed as: {merged_filename})")

    def _queue_multi_file_experiment(self, directory: str, base: str, ext: str, timestamp: str, 
                                   channels: list, params: dict, sim: bool) -> None:
        """
        Queues one experiment per channel, each with its own output file.
        
        Args:
            directory: Output directory for files
            base: Base filename without extension
            ext: File extension
            timestamp: Timestamp string for filename
            channels: List of channel numbers to measure
            params: Procedure parameters dictionary
            sim: Simulation mode flag
        """
        for ch_num in channels:
            # Create unique file for each channel
            full_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_num}{ext}")

            # Configure parameters for this specific channel
            single_ch_params = params.copy()
            for i in range(1, 7):
                single_ch_params[f'channel{i}'] = (i == ch_num)

            # Create procedure instance
            proc = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim,
                active_channel=ch_num,
                **single_ch_params
            )

            # Create results with unique file
            results = Results(proc, full_path)
            
            # Link results to procedure
            proc.results = results

            # Queue the experiment (no custom display filename for multi-file mode)
            self.manager.queue(self._new_experiment(results))
            logger.info(f"Queued experiment for channel {ch_num} to file: {full_path}")

    def _new_experiment(self, results: Results, display_filename: str = None) -> Experiment:
        """
        Creates a new Experiment object for the manager.
        
        Args:
            results: Results object containing procedure and data file info
            display_filename: Optional filename to display in browser (for single file mode)
            
        Returns:
            Experiment: Configured experiment object with curve and browser item
        """
        browser = self.view.browser_widget.browser
        color = pg.intColor(browser.topLevelItemCount() % 8)
        curve = self.view.plot_widget.new_curve(results, color=color)
        
        # Create browser item
        browser_item = BrowserItem(results, color)
        
        # Override displayed filename if provided (for single file mode)
        if display_filename:
            browser_item.setText(1, display_filename)  # Column 1 is typically the filename
        
        return Experiment(results, [curve], browser_item)

    def _merge_channel_files(self) -> None:
        """
        Merges all temporary channel files into a single final file.
        
        This method:
        - Combines data from all temporary files into the final merged file
        - Preserves headers and data formatting
        - Cleans up temporary files after successful merge
        """
        if self.merged_data_written:
            return
            
        try:
            logger.info("Starting file merge process...")
            
            # Create the merged file
            with open(self.merged_file_path, 'w', encoding='utf-8') as merged_file:
                header_written = False
                files_processed = 0
                
                # Process each channel's temporary file
                for ch_num in sorted(self.channels_to_merge):
                    temp_file_path = self.temp_files.get(ch_num)
                    
                    if not temp_file_path or not os.path.exists(temp_file_path):
                        logger.warning(f"Temporary file not found for channel {ch_num}")
                        continue
                        
                    try:
                        with open(temp_file_path, 'r', encoding='utf-8') as temp_file:
                            lines = temp_file.readlines()
                            
                            if not lines:
                                logger.warning(f"Empty temporary file for channel {ch_num}")
                                continue
                            
                            # Write header only once
                            if not header_written:
                                merged_file.write(lines[0])  # Write header
                                header_written = True
                                start_index = 1
                            else:
                                start_index = 1  # Skip header for subsequent files
                            
                            # Write data lines
                            for line in lines[start_index:]:
                                if line.strip():  # Skip empty lines
                                    merged_file.write(line)
                            
                            # Add separator between channels for readability
                            merged_file.write("\n")
                            
                        # Delete the temporary file after successful merge
                        os.remove(temp_file_path)
                        files_processed += 1
                        logger.info(f"Successfully merged data for channel {ch_num}")
                        
                    except Exception as e:
                        logger.error(f"Error processing temporary file for channel {ch_num}: {e}")
                        continue
            
            self.merged_data_written = True
            
            if files_processed > 0:
                logger.info(f"Successfully merged {files_processed} channel files into: {self.merged_file_path}")
            else:
                logger.warning("No files were successfully merged")
                
        except Exception as e:
            logger.error(f"Error during file merging process: {e}")

    def abort_experiment(self) -> None:
        """Aborts the currently running experiment."""
        logger.info("Experiment abort requested.")
        
        # Update UI state for aborting
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(False)
        self.view.abort_button.setText("Aborting...")
        
        # Reconfigure abort button for resume functionality
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.resume_experiment)
        
        # Signal manager to abort current experiment
        self.manager.abort()

    def resume_experiment(self) -> None:
        """Resumes the experiment queue if any experiments remain."""
        logger.info("Resuming experiment queue.")
        
        # Update UI state for resuming
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setText("Abort")
        
        # Reconfigure abort button for abort functionality
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.abort_experiment)
        
        # Resume or clean up based on queue state
        if self.manager.experiments.has_next():
            self.manager.resume()
        else:
            self.view.abort_button.setEnabled(False)
            self.view.queue_button.setEnabled(True)
            self.is_busy = False

    def clear_experiments(self) -> None:
        """Clears all experiments from the manager and resets counters."""
        self.manager.clear()
        self.finished_experiment_count = 0

    def _disconnect_instruments(self) -> None:
        """Helper function to disconnect instruments and log errors."""
        try:
            self.view.instrument_manager.disconnect_keithley()
            self.view.instrument_manager.disconnect_mux()
            logger.info("Instruments disconnected.")
        except Exception as e:
            logger.warning(f"Error disconnecting instruments: {e}")
        finally:
            self.view.update_instrument_lights()

    def on_abort_returned(self) -> None:
        """Handles UI state after an abort is complete."""
        logger.info("Abort completed.")
        
        # Disconnect instruments on abort
        self._disconnect_instruments()

        if self.manager.experiments.has_next():
            # PAUSED state - experiments remain in queue
            self.view.abort_button.setText("Resume")
            self.view.abort_button.setEnabled(True)
            self.view.queue_button.setEnabled(False)  # Can't queue while paused
            self.is_busy = True  # Still busy, just paused
        else:
            # IDLE state - no experiments remaining
            self.view.abort_button.setText("Abort")
            self.view.abort_button.setEnabled(False)
            self.view.queue_button.setEnabled(True)  # Can queue again
            self.view.browser_widget.clear_button.setEnabled(True)
            self.is_busy = False  # No longer busy

    def on_queued(self) -> None:
        """Handles UI state when an experiment is successfully queued."""
        logger.info("Experiment successfully queued.")
        
        # Update UI for busy state
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.abort_button.setText("Abort")
        
        # Ensure abort button is properly connected
        try:
            self.view.abort_button.clicked.disconnect()
        except TypeError:
            pass  # No connections to disconnect
        self.view.abort_button.clicked.connect(self.abort_experiment)

        # Enable browser controls
        browser_widget = self.view.browser_widget
        browser_widget.show_button.setEnabled(True)
        browser_widget.hide_button.setEnabled(True)
        browser_widget.clear_button.setEnabled(True)

    def on_running(self) -> None:
        """Handles UI state when an experiment starts running."""
        logger.info("Experiment started.")
        
        # Ensure UI reflects busy state
        self.view.queue_button.setEnabled(False)
        self.view.abort_button.setEnabled(True)
        self.view.browser_widget.clear_button.setEnabled(False)

    def on_finished(self) -> None:
        """Handles UI state when an experiment finishes."""
        logger.info("Experiment finished.")
        
        if not self.manager.experiments.has_next():
            # All experiments completed - transition to IDLE state
            self.view.abort_button.setEnabled(False)
            self.view.abort_button.setText("Abort")
            self.view.browser_widget.clear_button.setEnabled(True)
            
            # Merge files if this was a single-file experiment
            if hasattr(self, 'merged_file_path') and not getattr(self, 'merged_data_written', True):
                self._merge_channel_files()
            
            # Clean up and disconnect instruments
            logger.info("All queued experiments finished.")
            self._disconnect_instruments()
            
            self.view.queue_button.setEnabled(True)  # Can queue again
            self.is_busy = False  # No longer busy
        else:
            # More experiments remain in queue - stay in BUSY state
            self.view.queue_button.setEnabled(False)  # More items in queue
            self.is_busy = True
        
        self.view.update_instrument_lights()

    def update_analysis_panel(self) -> None:
        """
        Updates the analysis panel with results from a finished experiment.
        
        This method extracts analysis results from completed experiments
        and updates the analysis panel display with the calculated metrics.
        """
        try:
            browser = self.view.browser_widget.browser
            root = browser.invisibleRootItem()
            
            # Instead of using a counter, find the actual completed experiments
            # Process all browser items and check if they have analysis results
            for i in range(root.childCount()):
                browser_item = root.child(i)
                experiment = self.manager.experiments.with_browser_item(browser_item)
                
                if (experiment and 
                    hasattr(experiment.procedure, 'analysis_results') and 
                    experiment.procedure.analysis_results):
                    
                    # Only process if we haven't shown this analysis yet
                    # We can track processed experiments by their browser item
                    if not hasattr(browser_item, 'analysis_shown'):
                        all_results = experiment.procedure.analysis_results
                        for channel, metrics in all_results.items():
                            data_to_display = {'Channel': channel, **metrics}
                            self.view.analysis_panel.analysis(data_to_display)
                        
                        # Mark this browser item as processed
                        browser_item.analysis_shown = True
            
            # Reset the counter approach since we're now using a different method
            self.finished_experiment_count = root.childCount()
            
        except Exception as e:
            logger.error(f"Error during post-experiment analysis update: {e}", exc_info=True)