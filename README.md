# SolarJv Analyzer

A Python-based application for performing multi-channel J–V (current–voltage) measurements on solar cells using PyMeasure. This tool provides a GUI for configuring experiments, executing sweeps, and viewing results, with support for both real and simulated instruments.

---

## Project Structure

```
solarjv-analyzer/          # Root project directory
├── src/solarjv_analyzer/   # Python package
│   ├── config.py           # Configuration constants and defaults
│   ├── main.py             # Entry point: initializes and runs the GUI
│   ├── procedures/         # Experiment logic
│   │   └── jv_procedure.py # JVProcedure: defines parameters and measurement steps
│   ├── instruments/        # Hardware abstraction layer
│   │   ├── mux_controller.py       # MuxController & get_mux()
│   │   └── instrument_manager.py   # get_keithley() factory
│   ├── gui/                # Graphical user interface
│   │   ├── jv_analyzer_window.py   # MainWindow subclass: custom PyMeasure UI
│   │   └── login_dialog.py         # LoginDialog for user authentication
│   └── utils/              # Utility modules (e.g., database, file paths)
│       └── database.py     # check_credentials for login
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # Project overview and documentation (this file)
```

---

## Key Components

### config.py

* Defines a frozen dataclass `_Config` with hardware addresses, file paths, and default behaviors.
* Exposes module-level constants for easy import: `MUX_PORT`, `GPIB_ADDRESS`, `RESULTS_ROOT`, etc.

### procedures/jv\_procedure.py

* **`JVProcedure`**: Subclasses `pymeasure.experiment.Procedure` to implement the J–V sweep logic.

  * **Parameters**: User Name, start/stop voltage, step size, compliance current, GPIB address, channel enable flags.
  * **`DATA_COLUMNS`**: Specifies output CSV columns: Channel, Voltage, Current, Time, Status.
  * **`startup()`**: Initializes MUX and Keithley instruments (real or simulated), records start time, configures sourcing.
  * **`execute()`**: Loops over enabled channels and voltage points, switching the MUX, sourcing voltage, measuring current, and emitting results + progress.
  * **`shutdown()`**: Cleans up instruments, turns off outputs, and reserves space for appending metadata.

### instruments/mux\_controller.py

* **`MuxController`**: Opens a serial port to control a 6-channel multiplexer, with `select_channel(channel: int)` to route the desired cell.
* **`get_mux(port: str)`**: Factory that tries real MUX, falls back to `SimulatedMux` on failure.

### instruments/instrument\_manager.py

* **`get_keithley(address: str, ...)`**: Factory that returns either a real `Keithley2400` driver or a `SimulatedKeithley2400`, based on `CONFIG.SIMULATION_MODE`.
* Provides a consistent interface: `apply_voltage`, `enable_source`, `source_voltage` setter, `measure_current()`, `shutdown()`, etc.

### gui/login\_dialog.py

* **`LoginDialog`**: A simple PyQt5 `QDialog` that prompts for username/password.
* Uses `check_credentials(username, password)` from `utils/database.py` to validate and calls `accept()` on success.

### gui/jv\_analyzer\_window\.py

* **`JVAnalyzerWindow`**: Subclasses PyMeasure’s `ManagedWindow` or a custom base to assemble a tailored GUI:

  * **Left Dock**: Tab widget with:

    * **Parameters Tab**: Start voltage, stop voltage, step size, compliance, and enable flags for each channel.
    * **Instrument Tab**: GPIB address, MUX port, and channel count settings.
  * **File Save Section**: File name prefix, root directory, options for separate vs. combined files.
  * **Controls**: Queue, Abort buttons exactly like PyMeasure’s managed window.
  * **Main Area**: A `QTabWidget` housing the live `PlotWidget` and `LogWidget`, plus an experiment queue table below.
  * **Signal Connections**: Ties GUI buttons to `Manager.queue()` and `Manager.abort()`, and handles progress updates and log streaming.

### main.py

* Imports `JVAnalyzerWindow` (aliased as `MainWindow`).
* Initializes a `QApplication`, instantiates `MainWindow`, and starts the Qt event loop.

### utils/database.py

* **`check_credentials(username: str, password: str) -> bool`**: Placeholder implementation that checks a user database (e.g., SQLite) for valid login.

---

## Installation & Usage

1. **Install dependencies**:

   ```bash
   pip install .
   ```
2. **Configure hardware settings** in `config.py` (MUX\_PORT, GPIB\_ADDRESS, SIMULATION\_MODE).
3. **Run the application**:

   ```bash
   hatch run start   # or: python -m solarjv_analyzer.main
   ```
4. **Login Dialog** appears. Enter credentials.
5. **Configure experiment** in the left panel and click **Queue** to start.
6. **View live plot**, **log**, and **experiment queue** in the main area.
7. **Data files** are saved under `RESULTS_ROOT/<DD-MM-YYYY>/…` with a timestamp.

---

## Future Extensions

* **Data analysis tab**: Compute J<sub>SC</sub>, V<sub>OC</sub>, FF, and efficiency automatically.
* **Sequence Editor**: Automate multi-condition experiments (lamp on/off, temperature sweeps).
* **Database logging**: Store metadata and results in a central database for easy retrieval.
* **Plugin support**: Add new instrument drivers (e.g., other SMUs or multiplexers).

---

For detailed API, refer to inline docstrings in each module. Happy measuring!
