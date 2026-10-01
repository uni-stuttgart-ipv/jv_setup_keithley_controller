# Installing on a fresh Windows PC

For a lab machine with nothing on it — no Python, no drivers. Allow about an
hour, most of it downloads.

Everything below is run in **PowerShell**. Where a step needs administrator
rights it says so: right-click the Start button → *Terminal (Admin)* or
*Windows PowerShell (Admin)*.

---

## Step 0 — Collect four things from the PC that already works

Do this first. It turns most of the guide from guesswork into copying, and
takes two minutes. On the working machine, run:

```powershell
py --version
py -m pip freeze
py -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm); print(rm.list_resources())"
```

Write down:

1. **The Python version** (e.g. `3.13.2`) — install the same one on the new PC.
2. **The package list** — the versions that are known to work.
3. **The VISA backend and resource list.** The printed object says which backend
   is in use: `ResourceManager(<IVIVisaLibrary...>)` means **NI-VISA** is
   installed; `ResourceManager(<PyVisaLibrary...>)` means the pure-Python
   **pyvisa-py** backend. You want the same on the new PC. The resource list
   shows the instrument's address (e.g. `ASRL3::INSTR`).
4. **The Keithley's serial settings**, from its own front panel:
   `MENU → COMMUNICATION → RS-232`, then note **baud rate**, **flow control**
   and **terminator**. You will set the new PC's instrument the same way — and
   if this PC's Keithley is a different unit, you must configure it to match.

Also open `src/solarjv_analyzer/config.py` on the working machine and note
`MUX_PORT` and `GPIB_ADDRESS`. **These are almost certainly different on the new
PC** — COM port numbers are assigned per machine — but you need to know what
kind of value belongs there.

---

## Step 1 — Install Python

Download the **Windows installer (64-bit)** for the version you noted in Step 0
from <https://www.python.org/downloads/windows/>. If you have no working machine
to copy from, use the latest **3.13.x**.

> **Do not install 3.9 for this.** The repository contains a `.python-version`
> file saying `3.9.6` and the README mentions 3.9 — both refer only to
> **Briefcase packaging**, which pins that version to build the installers. The
> application itself needs **3.11 or newer**: the libraries it uses (numpy 2.3,
> scipy 1.16) dropped 3.9 and 3.10, and `requirements-windows.txt` will refuse
> to install. Keeping 3.9 installed alongside is fine and useful — just never
> run the app with it.

> Do **not** install Python from the Microsoft Store. Its sandboxed file access
> causes confusing permission failures when the app writes to `S:` and to
> `%LOCALAPPDATA%`.

In the installer:

1. Tick **"Add python.exe to PATH"** at the bottom of the first screen.
2. Click **"Customize installation"** → Next → on the Advanced Options page tick
   **"Install Python for all users"**. This matters: several whitelisted people
   log into this machine, and a per-user install is invisible to everyone but
   the person who ran it. The path should change to something like
   `C:\Program Files\Python313`.
3. Also tick **"Precompile standard library"**. Install.

### Command-line alternative

`pip` cannot install Python — it installs packages *into* a Python. Use
Windows' own package manager instead:

```powershell
winget install --id Python.Python.3.13 --source winget --scope machine
```

Drop `--scope machine` to install for your account only (no admin needed).

Or run the official installer unattended — check
<https://www.python.org/downloads/windows/> for the current 3.13 patch version
and substitute it below:

```powershell
curl.exe -L -o "$env:TEMP\py313.exe" https://www.python.org/ftp/python/3.13.15/python-3.13.15-amd64.exe
Start-Process -Wait "$env:TEMP\py313.exe" -ArgumentList "/passive InstallAllUsers=1 PrependPath=1 Include_launcher=1 CompileAll=1"
```

`InstallAllUsers=1` needs admin (use `0` for a per-user install);
`Include_launcher=1` is what provides the `py -3.13` version selector this guide
relies on.

### Confirm it worked

Close PowerShell, open a **new** one, and check:

```powershell
py -0p          # every Python on the machine, with its path
py -3.13 --version
py -3.13 -m pip --version
```

Both must print a version. If `py` is not recognised, the PATH tick was missed —
re-run the installer and choose *Modify*.

---

## Step 2 — Get the project onto the machine

Put it somewhere every user can reach. **Not** your Desktop or Documents folder,
which are per-user.

**With Git** (also lets you pull updates later — recommended):

Install Git for Windows from <https://git-scm.com/download/win> (all defaults are
fine), then:

```powershell
mkdir C:\SolarJV
cd C:\SolarJV
git clone https://github.com/uni-stuttgart-ipv/jv_setup_keithley_controller.git
cd jv_setup_keithley_controller
```

**Without Git:** download the ZIP from the GitHub page (green *Code* button →
*Download ZIP*), right-click it → **Properties** → tick **Unblock** → OK, then
extract it to `C:\SolarJV\`. Windows marks downloaded ZIPs as untrusted and
without unblocking you get odd permission errors later.

You should end up with `C:\SolarJV\jv_setup_keithley_controller\` containing
`src`, `tests` and `pyproject.toml`.

---

## Step 3 — Install the Python packages

**Check which Python you are about to use.** A machine can have several, and
`py` picks one for you — not always the one you want:

```powershell
py -0p          # lists every installed Python and its path
py -3.13 --version
```

If `py -3.13` is not found, go back to Step 1. Then, from an **administrator**
PowerShell (writing under `C:\SolarJV` needs it), naming the version on every
command so there is no ambiguity:

```powershell
cd C:\SolarJV\jv_setup_keithley_controller
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install -r requirements-windows.txt
py -3.13 -m pip install -e . --no-deps
```

This installs the known-good pinned versions, then registers the project itself
without letting pip re-resolve everything.

> **Why not a virtual environment?** On a shared single-purpose lab PC a
> machine-wide install is simpler: every whitelisted user gets a working app
> with no activation step. If you'd rather isolate it, create the venv inside
> `C:\SolarJV\` (so other users can read it), activate with
> `C:\SolarJV\jv_setup_keithley_controller\.venv\Scripts\Activate.ps1`, and run
> the same three commands. If PowerShell refuses to run the activation script,
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once will fix it.

Check it imports:

```powershell
py -3.13 -c "import PyQt5, pymeasure, pyqtgraph, pyvisa, serial, numpy, scipy, pandas, argon2; print('all packages OK')"
```

If this raises `ModuleNotFoundError`, the packages went to a different
interpreter than the one you just ran — see the first troubleshooting entry.

---

## Step 4 — Hardware drivers and the VISA backend

### 4a. USB-to-serial adapters

Both instruments talk over serial. If they connect through USB-to-RS232
adapters, Windows needs that adapter's driver — usually **FTDI**
(<https://ftdichip.com/drivers/vcp-drivers/>) or **Prolific PL2303**. Plug
everything in, then open **Device Manager** and expand **Ports (COM & LPT)**.

You should see one COM entry per adapter. If you see a yellow warning triangle
or *Unknown device* instead, the driver is missing — install it and re-plug.

**Write down which COM number is which.** Unplug one cable, see which entry
disappears, plug it back. Guessing here costs more time than checking.

### 4b. The VISA backend

PyVISA is only a wrapper; it needs a backend to actually drive the port. Match
whatever Step 0 told you the working PC uses.

**pyvisa-py** (already installed by Step 3) — pure Python, drives RS-232 through
pyserial. No extra download, no reboot. Sufficient for this setup, since the
Keithley is on a serial port, not GPIB.

**NI-VISA** — National Instruments' implementation. Bigger (about 1 GB, needs a
reboot) but it is the lab standard and ships **NI MAX**, which lets you open a
terminal to the instrument and send `*IDN?` by hand. That makes diagnosing "is
it the cable or the code?" far quicker. Download *NI-VISA Runtime* from
<https://www.ni.com/en/support/downloads/drivers/download.ni-visa.html>. If you
install it, remove the `pyvisa-py` line from `requirements-windows.txt` and
uninstall it (`py -3.13 -m pip uninstall pyvisa-py`) so there is no ambiguity about
which backend is in play.

Then check what PyVISA can see:

```powershell
py -c "import pyvisa; rm = pyvisa.ResourceManager(); print(rm); print(rm.list_resources())"
```

The Keithley should appear as `ASRL<n>::INSTR`, where `<n>` is its COM number.

### 4c. Set the Keithley's serial settings

The app opens the instrument with PyVISA's defaults: **9600 baud, 8 data bits,
no parity, 1 stop bit**. The Keithley must be set to match, or it will connect
and then return garbage — which looks like a software bug and is not one.

On the instrument: `MENU → COMMUNICATION → RS-232`, then set **BAUD = 9600**,
**FLOW = NONE**, and the terminator to whatever the working PC uses. Also make
sure the interface is set to **RS-232** and not GPIB.

The MUX is opened at **115200 baud** (fixed in `mux_controller.py`); it needs no
front-panel setup, only the right COM port.

---

## Step 5 — Point the app at this machine's COM ports

Open `C:\SolarJV\jv_setup_keithley_controller\src\solarjv_analyzer\config.py`
and set the two values to the COM numbers you found in Step 4a:

```python
MUX_PORT: str = "COM5"          # the MUX adapter
GPIB_ADDRESS = "ASRL3::INSTR"   # the Keithley — ASRL<n> where n is its COM number
```

So a Keithley on COM3 is `ASRL3::INSTR`, on COM7 it is `ASRL7::INSTR`.

Confirm the ports Windows is offering:

```powershell
py -c "import serial.tools.list_ports; [print(p.device, '-', p.description) for p in serial.tools.list_ports.comports()]"
```

---

## Step 6 — Check the software works before involving hardware

```powershell
cd C:\SolarJV\jv_setup_keithley_controller
py -3.13 -m pip install pytest
py -3.13 -m pytest tests/ -q
```

All tests should pass (there are 202 at the time of writing) and none of them
touch the instruments. If they pass, Python and every library are correctly
installed and any remaining problem is hardware, ports or permissions — which
narrows the search enormously.

---

## Step 7 — First run

Power on the chiller, sun simulator, Keithley and MUX, then:

```powershell
cd C:\SolarJV\jv_setup_keithley_controller
py -3.13 -m solarjv_analyzer.main
```

You should get the login window, then the calibration window with the hardware
status pill showing **CONNECTED**. If it shows *DISCONNECTED — CLICK TO RETRY*,
click it once (the app connects before you've had a chance to power things on,
which is why the retry exists); if it stays red, go to Troubleshooting.

### Logging in the first time

**A fresh machine has no accounts.** The real user database is
`%LOCALAPPDATA%\SolarJV\auth.db` and it does not exist until something creates
it. (The `users.db` committed inside `src/solarjv_analyzer/` belongs to the
*legacy* plaintext module that nothing uses — ignore it.) Accounts are per
machine, so one on the other PC does not help you here.

**And registration needs working email.** Creating an account through the UI
sends an OTP, so with SMTP unconfigured the flow cannot complete and you have no
way in. Two options:

**Option 1 — configure email** (do this eventually; it is the only thing that
scales past you). The app reads its SMTP settings in this order:
`S:\solarjv_email_config.json`, then the environment variables `SMTP_HOST`,
`SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_ADDR`, `SMTP_FROM_NAME`,
then `%USERPROFILE%\.solarjv\email_config.json`. Copying the config to `S:`
covers every machine and every user at once. Check outbound port 587 is not
firewalled.

**Option 2 — create the account directly** (fine for getting yourself running
today). The OTP is a gate in the login *dialog*; `register_user()` in
`auth/database.py` does not require it, so this creates a perfectly ordinary
Argon2id-hashed account:

```powershell
py -3.13 -c "import getpass; from solarjv_analyzer.auth.database import init_db, register_user; init_db(); p = getpass.getpass('New password: '); register_user('you@stud.uni-stuttgart.de', 'First', 'Last', 'stXXXXXX', p, p); print('account created')"
```

Arguments are `email, first_name, last_name, username, password, confirm`;
minimum 6 characters. `getpass` keeps the password out of your PowerShell
history. Verify with:

```powershell
py -3.13 -c "import getpass; from solarjv_analyzer.auth.database import authenticate_user; print('login OK:', authenticate_user('stXXXXXX', getpass.getpass('Password: ')))"
```

The Register and Forgot Password options in the UI are hidden until you press
**Ctrl+Shift+A** on the login screen.

### Where the app writes

| What | Where |
|---|---|
| Measurement data | `%USERPROFILE%\SolarJV_Data\` (or the folder chosen with Browse) |
| User accounts | `%LOCALAPPDATA%\SolarJV\auth.db` |
| Session logs | `%LOCALAPPDATA%\SolarJV\logs\` |
| Saved output-folder preference | `%USERPROFILE%\.solarjv\config.json` |

These are per-user **and** per-machine, so each whitelisted person gets their
own — expected, and the reason a colleague's PC "has no data" and no account
after your session. `auth.db` lives under `%LOCALAPPDATA%`, which is per-user
too: an account created by one Windows user is not visible to another.

---

## Step 8 — Make it launchable without a terminal

Lab users should not have to type commands. Create
`C:\SolarJV\Start SolarJV.bat` containing:

```bat
@echo off
cd /d C:\SolarJV\jv_setup_keithley_controller
py -3.13 -m solarjv_analyzer.main
if errorlevel 1 pause
```

Right-click it → *Send to* → *Desktop (create shortcut)*, and set the shortcut's
icon to `app_icon.ico` in the project folder. The `if errorlevel 1 pause` line
keeps the window open when something goes wrong, so the error is readable
instead of flashing past.

To put the shortcut on **every** user's desktop, drop it in
`C:\Users\Public\Desktop\` (needs admin).

---

## Troubleshooting

**`No matching distribution found for pyvisa==1.15.0`, with a line above it
about versions that "require a different python version"** — you are on Python
3.9 or 3.10. pip is saying your interpreter is too old, not that the package is
missing. Check with `py -0p`, install 3.13 (Step 1), then redo Step 3 using
`py -3.13` on every command. If the second command already ran, the project got
registered under the old Python with none of its libraries; clean that up first:

```powershell
py -3.9 -m pip uninstall solarjv-analyzer -y
```

**`ModuleNotFoundError: No module named 'PyQt5'` even though pip said it
installed** — pip installed into one Python and you ran another. Always use the
same `py -3.13` prefix for both. `py -3.13 -m pip list` shows what that specific
interpreter actually has.

**`WARNING: The script solarjv.exe is installed in ... which is not on PATH`** —
harmless. It only affects typing `solarjv` as a shortcut command; running
`py -3.13 -m solarjv_analyzer.main` works regardless.

**Python installed but only for you** — if the install path contains
`AppData\Local\Programs\Python`, it is a per-user install and no other account
on the PC can see it. Fine if you are the only operator; re-run the installer
with *Install for all users* if colleagues need it too.

**`py` is not recognized** — the "Add python.exe to PATH" box was not ticked.
Re-run the installer, choose *Modify*, tick it. Open a new PowerShell afterwards;
an already-open one keeps the old PATH.

**`Could not locate a VISA implementation`** — no backend. Either
`py -3.13 -m pip install pyvisa-py` or install NI-VISA (Step 4b).

**`VI_ERROR_RSRC_NFOUND` / the resource list is empty** — the adapter driver is
missing or the COM number in `config.py` is wrong. Check Device Manager and
re-run the `list_resources()` command from Step 4b.

**Connects, but every reading is garbage or times out** — the Keithley's baud
rate does not match. Set it to 9600 on the front panel (Step 4c).

**`Access is denied` on the COM port** — something else has it open. Close any
other instance of the app, plus PuTTY, NI MAX, Arduino IDE or a serial monitor.
Only one program can hold a COM port.

**`Invalid session handle. The resource might be closed.`** — a software bug
that was fixed in September 2026. Make sure this machine has the current code;
`git pull` if you cloned it.

**`UnicodeEncodeError` when a report is saved** — a `µ`, `°` or umlaut in the
Notes field. Known bug (audit finding A4); avoid those characters in Notes until
it is fixed, or the merged report will not be written.

**The app freezes for 20–60 seconds** — usually a network drive that is mapped
but unreachable. Check the `S:` mapping.

**Windows Defender / SmartScreen blocks something** — expected for a downloaded
ZIP that was not unblocked (Step 2), and for the FTDI installer. Use *More info
→ Run anyway* for the driver, and Unblock for the ZIP.

---

## Updating this machine later

```powershell
cd C:\SolarJV\jv_setup_keithley_controller
git pull
py -3.13 -m pip install -r requirements-windows.txt
py -3.13 -m pytest tests/ -q
```

Re-check `config.py` after a pull: it holds this machine's COM ports, so if the
pull overwrites it you must set them again. (Worth moving those two settings out
of source and into a per-machine config file — see the audit's medium/low list.)
