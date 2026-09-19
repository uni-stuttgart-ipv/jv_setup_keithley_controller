#!/usr/bin/env python3
"""Show every serial port, and what startup port resolution would decide.

Run this on the lab PC to check that the resolver sees the hardware the way it
is meant to — before trusting it to pick ports on its own.

    py -3.13 tools\\port_probe.py              # list ports + the decision
    py -3.13 tools\\port_probe.py --probe      # also send *IDN? to find the Keithley
    py -3.13 tools\\port_probe.py --json       # machine-readable

`--probe` opens candidate ports and sends `*IDN?`, which is a read-only
identity query. It never sends anything else, and it never touches a port that
looks like the MUX: that board expects binary hex frames and could read `*IDN?`
as one. Without `--probe` nothing is opened at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from solarjv_analyzer.instruments import port_resolver as pr   # noqa: E402


def describe(port) -> dict:
    return {
        "device": port.device,
        "vid": getattr(port, "vid", None),
        "pid": getattr(port, "pid", None),
        "serial": getattr(port, "serial_number", None),
        "location": getattr(port, "location", None),
        "description": getattr(port, "description", "") or "",
        "manufacturer": getattr(port, "manufacturer", "") or "",
        "looks_like_mux": pr.matches_signature(port, pr.MUX_VIDS, pr.MUX_KEYWORDS),
        "looks_like_keithley": pr.matches_signature(
            port, pr.KEITHLEY_VIDS, pr.KEITHLEY_KEYWORDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", action="store_true",
                        help="send *IDN? to candidate ports to confirm the Keithley")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    ports = pr._enumerate()
    rows = [describe(p) for p in ports]
    result = pr.resolve(ports=ports, probe=args.probe)

    if args.json:
        print(json.dumps({
            "ports": rows,
            "resolution": {
                "mux_port": result.mux_port,
                "mux_source": result.mux_source,
                "keithley_resource": result.keithley_resource,
                "keithley_source": result.keithley_source,
                "problems": result.problems,
            },
        }, indent=2))
        return 0 if result.ok else 1

    print("=" * 74)
    print(f"Serial ports on this machine   (python {sys.version.split()[0]})")
    print("=" * 74)
    if not rows:
        print("  none found — no adapters are plugged in, or the drivers are missing")
    for row in rows:
        guess = ("MUX?" if row["looks_like_mux"] else
                 "Keithley?" if row["looks_like_keithley"] else "—")
        print(f"\n  {row['device']}   [{guess}]")
        print(f"      vid:pid   {(row['vid'] or 0):04X}:{(row['pid'] or 0):04X}")
        print(f"      serial    {row['serial'] or '(none — identity falls back to the USB socket)'}")
        print(f"      location  {row['location'] or '(unknown)'}")
        print(f"      describes {row['description']}")

    print()
    print("-" * 74)
    print("What startup resolution would decide"
          + ("" if args.probe else "   (without --probe, so no *IDN? confirmation)"))
    print("-" * 74)
    print(f"  MUX       {result.mux_port or '— UNRESOLVED —':<16} via {result.mux_source}")
    print(f"  Keithley  {result.keithley_resource or '— UNRESOLVED —':<16} via {result.keithley_source}")
    for problem in result.problems:
        print(f"  !  {problem}")

    saved = pr.load_hardware_settings()
    if saved:
        print(f"\n  remembered in {pr.settings_path()}:")
        print(f"      MUX      {saved.get('mux_port', '—')}"
              f"  pinned={bool(saved.get('mux_pinned'))}")
        print(f"      Keithley {saved.get('keithley_resource', '—')}"
              f"  pinned={bool(saved.get('keithley_pinned'))}")
    else:
        print(f"\n  nothing remembered yet ({pr.settings_path()})")

    print()
    if result.ok:
        print("RESULT: PASS — both instruments resolve.")
        return 0
    print("RESULT: FAIL — see the problems above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
