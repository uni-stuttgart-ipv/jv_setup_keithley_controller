#!/usr/bin/env python3
"""Reproduce (or rule out) the intermittent GUI segfault, and say where it died.

    python tools/repro_window_crash.py            # 10 runs of the GUI suite
    python tools/repro_window_crash.py 30         # 30 runs
    python tools/repro_window_crash.py 10 --show  # real on-screen windows

Run it from the repository root, with the project's virtual environment
active. It needs nothing but pytest and the app's own dependencies.

WHAT IT DOES

Runs the GUI test suite over and over in a fresh process each time and
reports, for each run, whether the process finished normally or was killed.
When it is killed it names the last test that started -- which is the one
that died, because a segmentation fault kills the process instantly and the
test never gets to report a result.

HOW TO READ THE OUTPUT

A segmentation fault is not an ordinary test failure. The process is killed
outright, so there is no traceback and no "FAILED" line. What you see instead
is an exit status: 139 on macOS and Linux, 3221225477 (0xC0000005, "access
violation") on Windows. Either one means the same thing -- the program
reached into memory that had already been freed.

WHAT WE KNOW SO FAR (Linux container, roughly 1 run in 3)

The site VARIES from run to run -- it is not pinned to one test. One known
landing spot is test_pyqtgraphs_view_registry_does_not_grow_per_window, in
tests/gui/test_control_affordances.py, which deliberately holds three
closed-and-deleted windows alive at once, because pyqtgraph keeps a registry
(ViewBox.AllViews) that is weak on the Python wrapper but NOT on the C++
object underneath. A window whose C++ side has been freed while its Python
wrapper is still held leaves a stale entry there, and the next plot built
walks that registry and touches freed memory.

Running that test, or even its whole file, on its own does NOT crash. It
needs the earlier GUI tests to have built windows first. So the suite is the
reproducer, not any single test.

WHAT WOULD BE WORTH KNOWING FROM YOUR MACHINE

1. Does it happen at all on macOS or Windows, or only in the Linux container?
   If it never happens on either real platform, this is very likely an
   artefact of the test and the lab application is not at risk.
2. If it does happen, is it always the same test?
3. Does --show (real windows, real window manager) change the rate?

Send back the summary block this prints; that is enough to tell.
"""

import argparse
import os
import re
import subprocess
import sys

CRASH_CODES = {
    139: "segmentation fault (SIGSEGV)",
    -11: "segmentation fault (SIGSEGV)",
    134: "abort (SIGABRT)",
    -6: "abort (SIGABRT)",
    3221225477: "access violation (0xC0000005)",
    3221225725: "stack overflow (0xC00000FD)",
}


def crash_site(output):
    """Where the process died, as precisely as the output allows.

    With -v, pytest writes the test's name, then appends its result to the
    same line. So a trailing name with NO result is the test that died. A
    trailing name WITH a result means the crash came in whatever ran next,
    and that name never made it out of the buffer -- all we can honestly say
    is "just after".
    """
    lines = re.findall(r"^(tests/\S+::\S+.*)$", output, re.MULTILINE)
    if not lines:
        return "unknown (no test name in the output)"
    last = lines[-1].strip()
    name = last.split()[0]
    if re.search(r"\b(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b", last):
        return f"just after {name}"
    return f"in {name}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="?", type=int, default=10)
    parser.add_argument("--show", action="store_true",
                        help="build real on-screen windows instead of offscreen")
    parser.add_argument("--path", default="tests/gui",
                        help="what to run (default: tests/gui)")
    args = parser.parse_args()

    env = dict(os.environ)
    if not args.show:
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

    print(f"python   {sys.version.split()[0]}")
    print(f"platform {sys.platform}  "
          f"QPA={env.get('QT_QPA_PLATFORM', 'default (on-screen)')}")
    print(f"running  {args.path}  x{args.runs}")
    print("-" * 64, flush=True)

    crashes = []
    for n in range(1, args.runs + 1):
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", args.path, "-v", "-p", "no:randomly"],
            env=env, capture_output=True, text=True,
        )
        output = proc.stdout + proc.stderr
        code = proc.returncode

        if code in CRASH_CODES:
            where = crash_site(output)
            crashes.append(where)
            print(f"  run {n:3d}  CRASHED  ({CRASH_CODES[code]}, exit {code})")
            print(f"           died {where}", flush=True)
        elif code == 0:
            print(f"  run {n:3d}  clean", flush=True)
        else:
            # An ordinary test failure, which is a different thing entirely.
            print(f"  run {n:3d}  tests failed (exit {code}) -- not a crash",
                  flush=True)

    print("-" * 64)
    print(f"SUMMARY: {len(crashes)} crash(es) in {args.runs} runs")
    if crashes:
        for site in sorted(set(crashes)):
            print(f"  {crashes.count(site):3d}x  {site}")
    else:
        print("  No crash on this machine in this many runs.")
        print("  That is a real result: it points at the Linux container,")
        print("  not at the application.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
