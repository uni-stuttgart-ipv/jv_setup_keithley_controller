# SolarJV Analyzer

A Python-based GUI application for performing multi-channel J–V (current–voltage) measurements on solar cells using PyMeasure and PyQt5. The tool supports both real and simulated instruments and can be packaged into native macOS and Windows installers.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Development Installation](#development-installation)
- [Running the Application](#running-the-application)
- [Building Standalone Applications](#building-standalone-applications)
  - [macOS](#macos)
  - [Windows](#windows)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Future Extensions](#future-extensions)
- [License](#license)

---

## Features

- Comprehensive Analysis: Automatically calculates key solar cell metrics (Voc, Isc, Fill Factor, Efficiency, etc.) after each measurement.
- GUI for configuring and running JV sweeps with PyMeasure
- Real or simulated instrument backends (Keithley SMU and serial MUX)
- Live plot and log display
- CSV data export with timestamped directories
- Cross-platform packaging into `.app`/`.dmg` for macOS and `.msi` for Windows

---

## Requirements

- Python 3.8+
- [PyMeasure](https://github.com/ralph-tice/pymeasure)
- [PyQt5](https://pypi.org/project/PyQt5/)
- [pyserial](https://pypi.org/project/pyserial/)
- [pyvisa](https://pypi.org/project/PyVISA/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [pandas](https://pypi.org/project/pandas/)
- [reportlab](https://pypi.org/project/reportlab/)

Optional for development:

- [hatch](https://hatch.pypa.io/) for environment management
- [Briefcase](https://briefcase.readthedocs.io/) for native app packaging

---

## Development Installation

```bash
git clone https://github.com/uni-stuttgart-ipv/jv_setup_keithley_controller.git
cd jv_setup_keithley_controller
git checkout v2
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

---

## Running the Application

Launch in development mode:

```bash
python -m solarjv_analyzer.main
# or, if using hatch:
hatch run start
```

---

## Building Standalone Applications

### macOS

1. Install Briefcase and prerequisites:
   ```bash
   pip install briefcase
   ```
2. Create, build, and package:
   ```bash
   briefcase create macOS
   briefcase build   macOS
   briefcase package macOS
   ```
3. Distribute the resulting `.dmg` in `dist/macOS/`.

### Windows

1. On a Windows host, install Briefcase:
   ```powershell
   pip install briefcase
   ```
2. Create, build, and package:
   ```powershell
   briefcase create windows
   briefcase build   windows
   briefcase package windows
   ```
3. Distribute the resulting `.msi` in `dist\Windows\`.

---

## Configuration

Edit `src/solarjv_analyzer/config.py` to set:

- `MUX_PORT` (serial port for multiplexer)
- `GPIB_ADDRESS` (address of Keithley SMU)
- `RESULTS_ROOT` (directory for saving CSV data)
- `SIMULATION_MODE` (toggle between real or simulated instruments)

---

## Project Structure

```
jv_setup_keithley_controller/
├── LICENSE.txt
├── pyproject.toml
├── setup.py
├── README.md
├── src/
│   └── solarjv_analyzer/
│       ├── __init__.py
│       ├── __main__.py
│       ├── main.py
│       ├── config.py
│       ├── analysis/
│       │   ├── efficiency.py
│       │   ├── fill_factor.py
│       │   ├── isc.py
│       │   ├── main_calculator.py
│       │   ├── mpp.py
│       │   ├── resistances.py
│       │   └── voc.py
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── app_controller.py
│       │   ├── jv_analyzer_window.py
│       │   ├── login_dialog.py
│       │   └── widgets/
│       │       ├── analysis_panel.py
│       │       ├── analysis_settings_tab.py
│       │       ├── file_panel.py
│       │       ├── instrument_tab.py
│       │       └── parameter_tab.py
│       ├── instruments/
│       │   ├── __init__.py
│       │   ├── instrument_manager.py
│       │   ├── mux_controller.py
│       │   └── simulated/
│       │       ├── simulated_keithley.py
│       │       └── simulated_mux.py
│       ├── procedures/
│       │   ├── __init__.py
│       │   └── jv_procedure.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── database.py
│       └── users.db
├── data/
├── logs/
└── reports/
```

---

## Future Extensions

- Sequence editor for complex experiment workflows
- Centralized database logging
- Plugin architecture for additional instruments

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
