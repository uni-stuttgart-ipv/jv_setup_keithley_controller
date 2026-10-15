"""Every live recorder file must be invisible to the sweeper.

This exists because of a bug that reached review: the DEFAULT J-V mode
(multi-file) handed pymeasure `..._ch1_forward.csv` while single-file and
combined mode handed it `..._ch1_forward_temp.csv`. pymeasure writes the
header at construction and appends for the whole sweep, so with the quiet
window at 0.0 the sweeper could claim, publish and delete a queued channel's
file before that channel ever ran — recreating the exact
`KeyError: 'Voltage (V)'` crash this subsystem was changed to prevent.

A case-by-case test would not have caught it, because the broken path was the
one nobody had written a case for. This checks the rule instead: every path
handed to `Results(...)` must be one `find_finished()` skips.
"""
import ast
import os

import pytest

from solarjv_analyzer.store import paths

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src", "solarjv_analyzer")
PRODUCERS = [
    os.path.join(SRC, "gui", "app_controller.py"),
    os.path.join(SRC, "windows", "calibration_window.py"),
]


def _literal_parts(node):
    """Every constant chunk of an f-string or plain string, joined."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        )
    return None


def _recorder_paths(tree):
    """(variable name, literal skeleton) for every path given to Results()."""
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            name = getattr(target, "id", None)
            if name is None:
                continue
            # `x = f"..."` and `x = os.path.join(dir, f"...")` are both used.
            direct = _literal_parts(node.value)
            if direct:
                assignments.setdefault(name, []).append(direct)
            for call in ast.walk(node.value):
                if not isinstance(call, ast.Call):
                    continue
                if getattr(call.func, "attr", "") != "join":
                    continue
                for arg in call.args:
                    text = _literal_parts(arg)
                    if text:
                        assignments.setdefault(name, []).append(text)

    handed_over = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Results":
            if len(node.args) < 2:
                continue
            arg = node.args[1]
            if isinstance(arg, ast.Name):
                for text in assignments.get(arg.id, []):
                    handed_over.append((arg.id, text))
            else:
                text = _literal_parts(arg)
                if text:
                    handed_over.append(("<inline>", text))
    return handed_over


@pytest.mark.parametrize("source", PRODUCERS, ids=lambda p: os.path.basename(p))
def test_every_recorder_path_is_skipped_by_the_sweeper(source):
    tree = ast.parse(open(source, encoding="utf-8").read())
    handed_over = _recorder_paths(tree)
    assert handed_over, f"no Results(...) paths found in {source} — did it move?"

    offenders = []
    for name, skeleton in handed_over:
        # The skeleton still has `{...}` holes; fill them with something that
        # cannot accidentally create a skip suffix.
        probe = skeleton.replace("{", "<").replace("}", ">")
        if not (paths.is_temp_name(probe)
                or paths.is_part_name(probe)
                or paths.is_journal_name(probe)):
            offenders.append(f"{name} -> {skeleton}")

    assert not offenders, (
        "These paths are handed to a pymeasure Results, which appends to them "
        "for the whole sweep, but the sweeper would treat them as finished "
        "files and publish and delete them mid-run:\n  "
        + "\n  ".join(offenders)
    )
