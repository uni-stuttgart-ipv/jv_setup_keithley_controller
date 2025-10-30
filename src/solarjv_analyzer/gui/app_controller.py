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
    """

    def __init__(self, view):
        """
        Initializes the AppController.

        Args:
            view: A reference to the main JVAnalyzerWindow instance.
        """
        self.view = view
        self.finished_experiment_count = 0

        self.manager = Manager(
            [self.view.plot_widget, self.view.log_widget, self.view.analysis_panel],
            self.view.browser_widget.browser,
            log_level=logging.INFO,
            parent=self.view
        )
        self._connect_manager_signals()

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
        """
        self.view.update_save_directory()

        procedure_params = {
            'user_name': self.view.username,
            **self.view.params_tab.get_parameters(),
            **self.view.instr_tab.get_parameters(),
            **self.view.analysis_settings_tab.get_parameters(),
        }

        file_params = self.view.file_panel.get_parameters()
        channels_selected = self.view.params_tab.get_selected_channels()
        if not channels_selected:
            logger.warning("No channels selected for measurement.")
            return

        self.view.analysis_panel.reset_channels(
            channels_selected, JVProcedure.ANALYSIS_LABELS_UNITS
        )

        sim_mode = file_params['simulation']
        self.view.instrument_manager.connect_keithley(simulation=sim_mode)
        self.view.instrument_manager.connect_mux(simulation=sim_mode)
        self.view.update_instrument_lights()

        timestamp_str = datetime.now().strftime(TIMESTAMP_FORMAT)
        filename_timestamp = timestamp_str.replace(":", "-").replace(" ", "_")
        base, ext = os.path.splitext(file_params['filename'])
        directory = file_params['directory']

        if file_params['single_file']:
            self._queue_single_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )
        else:
            self._queue_multi_file_experiment(
                directory, base, ext, filename_timestamp, channels_selected, procedure_params, sim_mode
            )

    def _queue_single_file_experiment(self, directory, base, ext, timestamp, channels, params, sim):
        """Helper to queue one experiment for all channels."""
        ch_list_str = "_".join(map(str, channels))
        full_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_list_str}{ext}")

        for i in range(1, 7):
            params[f'channel{i}'] = (i in channels)

        # NECESSARY WIRING CHANGE: Pass the instrument and mux objects directly,
        # just like the original, working code did.
        proc = JVProcedure(
            instrument=self.view.instrument_manager.keithley,
            mux=self.view.instrument_manager.mux,
            manager=self.view.instrument_manager,
            simulation=sim,
            **params
        )
        
        results = Results(proc, full_path)
        self.manager.queue(self._new_experiment(results))
        logger.info(f"Queued combined experiment for channels {channels} to file: {full_path}")

    def _queue_multi_file_experiment(self, directory, base, ext, timestamp, channels, params, sim):
        """Helper to queue one experiment per channel."""
        for ch_num in channels:
            full_path = os.path.join(directory, f"{base}_{timestamp}_ch{ch_num}{ext}")

            single_ch_params = params.copy()
            for i in range(1, 7):
                single_ch_params[f'channel{i}'] = (i == ch_num)

            # NECESSARY WIRING CHANGE: Pass the instrument and mux objects directly,
            # just like the original, working code did.
            proc = JVProcedure(
                instrument=self.view.instrument_manager.keithley,
                mux=self.view.instrument_manager.mux,
                manager=self.view.instrument_manager,
                simulation=sim,
                active_channel=ch_num,
                **single_ch_params
            )

            results = Results(proc, full_path)
            self.manager.queue(self._new_experiment(results))
            logger.info(f"Queued experiment for channel {ch_num} to file: {full_path}")

    def abort_experiment(self) -> None:
        """Aborts the currently running experiment."""
        logger.info("Experiment abort requested.")
        self.view.abort_button.setEnabled(False)
        self.view.abort_button.setText("Resume")
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.resume_experiment)
        self.manager.abort()

    def resume_experiment(self) -> None:
        """Resumes the experiment queue if any experiments remain."""
        logger.info("Resuming experiment queue.")
        self.view.abort_button.setText("Abort")
        self.view.abort_button.clicked.disconnect()
        self.view.abort_button.clicked.connect(self.abort_experiment)
        if self.manager.experiments.has_next():
            self.manager.resume()
        else:
            self.view.abort_button.setEnabled(False)

    def clear_experiments(self) -> None:
        """Clears all experiments from the manager and resets counters."""
        self.manager.clear()
        self.finished_experiment_count = 0

    def _new_experiment(self, results) -> Experiment:
        """Creates a new Experiment object for the manager."""
        browser = self.view.browser_widget.browser
        color = pg.intColor(browser.topLevelItemCount() % 8)
        curve = self.view.plot_widget.new_curve(results, color=color)
        browser_item = BrowserItem(results, color)
        return Experiment(results, [curve], browser_item)

    def on_abort_returned(self) -> None:
        """Handles UI state after an abort is complete."""
        logger.info("Abort completed.")
        if self.manager.experiments.has_next():
            self.view.abort_button.setText("Resume")
            self.view.abort_button.setEnabled(True)
        else:
            self.view.browser_widget.clear_button.setEnabled(True)

    def on_queued(self) -> None:
        """Handles UI state when an experiment is successfully queued."""
        logger.info("Experiment successfully queued.")
        self.view.abort_button.setEnabled(True)
        self.view.abort_button.setText("Abort")
        try:
            self.view.abort_button.clicked.disconnect()
        except TypeError:
            pass
        self.view.abort_button.clicked.connect(self.abort_experiment)

        browser_widget = self.view.browser_widget
        browser_widget.show_button.setEnabled(True)
        browser_widget.hide_button.setEnabled(True)
        browser_widget.clear_button.setEnabled(True)

    def on_running(self) -> None:
        """Handles UI state when an experiment starts running."""
        logger.info("Experiment started.")
        self.view.browser_widget.clear_button.setEnabled(False)

    def on_finished(self) -> None:
        """Handles UI state when an experiment finishes."""
        logger.info("Experiment finished.")
        if not self.manager.experiments.has_next():
            self.view.abort_button.setEnabled(False)
            self.view.abort_button.setText("Abort")
            self.view.browser_widget.clear_button.setEnabled(True)
        self.view.update_instrument_lights()

    def update_analysis_panel(self):
        """Updates the analysis panel with results from a finished experiment."""
        try:
            browser = self.view.browser_widget.browser
            root = browser.invisibleRootItem()
            if self.finished_experiment_count >= root.childCount():
                return

            finished_item = root.child(self.finished_experiment_count)
            experiment = self.manager.experiments.with_browser_item(finished_item)

            if experiment and hasattr(experiment.procedure, 'analysis_results'):
                all_results = experiment.procedure.analysis_results
                for channel, metrics in all_results.items():
                    data_to_display = {'Channel': channel, **metrics}
                    self.view.analysis_panel.analysis(data_to_display)

            self.finished_experiment_count += 1
        except Exception as e:
            logger.error(f"Error during post-experiment analysis update: {e}", exc_info=True)