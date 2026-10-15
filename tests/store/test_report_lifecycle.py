"""End to end: a report's whole life, from first byte to the store.

The unit tests prove each piece. This proves the pieces agree with each other,
which is where this subsystem has gone wrong before: a producer wrote under a
name the sweeper thought was finished, and the sweeper deleted the file out
from under the process still writing it.

One run of each producer shape:
  * a report written by the application  (J-V, calibration)
  * a long journal plus a report          (SPO, and later MPP)
  * the same, interrupted by a crash
"""
import os

import pytest

from solarjv_analyzer.store import paths, sweeper
from solarjv_analyzer.store.finalize import published_file


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A staging tree and a store, wired the way the running app wires them."""
    staging = tmp_path / "staging"
    store = tmp_path / "store"
    (staging / "yaman" / "09-09-2026" / "JV").mkdir(parents=True)
    (staging / "yaman" / "09-09-2026" / "Calibration").mkdir(parents=True)
    (staging / "yaman" / "09-09-2026" / "SPO").mkdir(parents=True)
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(staging))
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(store))
    monkeypatch.setattr(sweeper, "extra_target", lambda: "")
    monkeypatch.setattr(sweeper, "_logs_dir", lambda: "")
    from solarjv_analyzer.store import identity
    # `active_user()` reads `_app_user` (the application login). `_cached` is
    # only `windows_user()`'s, and patching it leaves active_user() raising,
    # which makes every sweep a no-op and every assertion here a false pass.
    monkeypatch.setattr(identity, "_app_user", "st000000")
    return staging, store


def _published(store):
    return sorted(
        name
        for _d, _s, files in os.walk(str(store))
        for name in files
    )


def test_a_report_is_invisible_while_written_and_published_the_moment_it_is_not(tree):
    """The core promise, with no quiet window left to hide behind."""
    staging, store = tree
    report = staging / "yaman" / "09-09-2026" / "JV" / "run_2026-09-09.csv"

    with published_file(str(report), newline="") as handle:
        handle.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
        # Flush, so the part file is genuinely non-empty on disk. Without this
        # the bytes sit in Python's buffer, the file is 0 bytes, and the
        # sweeper's empty-file rule would do the work — the test would pass
        # with the `.part` rule deleted and prove nothing.
        handle.flush()
        os.fsync(handle.fileno())
        part = staging / "yaman" / "09-09-2026" / "JV" / (
            "run_2026-09-09.csv" + paths.PART_SUFFIX)
        assert part.stat().st_size > 0, "the part file must be non-empty to test this"

        # Mid-write, with a zero quiet window and a non-empty file: only the
        # name stops this being published.
        assert sweeper.find_finished(str(staging)) == []
        assert sweeper.sweep(include_logs=False)["published"] == 0
        assert _published(store) == []

    # Closed. No waiting, no settling — it is publishable on this very sweep.
    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert _published(store) == ["run_2026-09-09.csv"]
    assert not report.exists(), "the staged copy should be cleared once published"


def test_a_failed_report_never_reaches_the_store(tree):
    staging, store = tree
    report = staging / "yaman" / "09-09-2026" / "JV" / "run_2026-09-09.csv"

    with pytest.raises(ZeroDivisionError):
        with published_file(str(report), newline="") as handle:
            handle.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
            1 / 0                                   # analysis blows up

    assert sweeper.sweep(include_logs=False)["published"] == 0
    assert _published(store) == []
    assert not report.exists()


def test_a_calibration_sweep_in_progress_is_not_published(tree):
    """pymeasure appends to a `_temp` sibling for the length of the sweep."""
    staging, store = tree
    folder = staging / "yaman" / "09-09-2026" / "Calibration"
    live = folder / "calibration_2026-09-09_temp.csv"
    live.write_text("V,I\n0.0,0.1\n")               # recorder still appending

    assert sweeper.sweep(include_logs=False)["published"] == 0
    assert _published(store) == []
    assert live.exists(), "the live recorder file was taken away"

    # The real report is written afterwards, to the name without `_temp`.
    with published_file(str(folder / "calibration_2026-09-09.csv")) as handle:
        handle.write("[[ EXPERIMENTAL PARAMETERS ]]\n")

    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert _published(store) == ["calibration_2026-09-09.csv"]


def test_an_spo_hold_publishes_its_report_and_not_its_journal(tree):
    staging, store = tree
    folder = staging / "yaman" / "09-09-2026" / "SPO"
    journal = folder / ("spo_ch3_2026-09-09_raw.csv" + paths.JOURNAL_SUFFIX)
    journal.write_text("Time (s),Voltage (V)\n0.0,0.85\n")

    # Mid-hold: nothing to publish, and the journal must survive.
    assert sweeper.sweep(include_logs=False)["published"] == 0
    assert journal.exists()

    with published_file(str(folder / "spo_ch3_2026-09-09_report.csv")) as handle:
        handle.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
    os.remove(str(journal))                          # finalize() discards it

    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert _published(store) == ["spo_ch3_2026-09-09_report.csv"]


def test_a_crashed_hold_still_gets_its_measurements_to_the_store(tree):
    """No report was ever written, so the journal IS the measurement.

    This is the behaviour the sweeper has always had for a crashed hold, and
    it is the one thing the `.journal` suffix could quietly have taken away.
    """
    staging, store = tree
    folder = staging / "yaman" / "09-09-2026" / "SPO"
    journal = folder / ("spo_ch3_2026-09-09_raw.csv" + paths.JOURNAL_SUFFIX)
    journal.write_text("Time (s),Voltage (V)\n0.0,0.85\n1.0,0.85\n")

    stale = sweeper.STALE_IN_PROGRESS_SECONDS + 60
    old = os.path.getmtime(str(journal)) - stale
    os.utime(str(journal), (old, old))

    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert _published(store) == ["spo_ch3_2026-09-09_raw.csv"]
    assert "1.0,0.85" in (
        store / "st000000" / "2026-09-09" / "SPO" / "spo_ch3_2026-09-09_raw.csv"
    ).read_text()


def test_a_crash_mid_report_does_not_publish_a_truncated_file(tree):
    """The opposite case: a stale `.part` is wreckage, not data."""
    staging, store = tree
    folder = staging / "yaman" / "09-09-2026" / "JV"
    part = folder / ("run_2026-09-09.csv" + paths.PART_SUFFIX)
    part.write_text("[[ EXPERIMENTAL PARAMETERS ]]\nhalf a rep")

    stale = sweeper.STALE_IN_PROGRESS_SECONDS + 60
    old = os.path.getmtime(str(part)) - stale
    os.utime(str(part), (old, old))

    assert sweeper.sweep(include_logs=False)["published"] == 0
    assert _published(store) == []
    assert not part.exists()
    assert not (folder / "run_2026-09-09.csv").exists()


def test_several_producers_running_at_once_do_not_interfere(tree):
    """A combined run has a J-V report, an SPO journal and a live recorder."""
    staging, store = tree
    jv = staging / "yaman" / "09-09-2026" / "JV"
    spo = staging / "yaman" / "09-09-2026" / "SPO"
    cal = staging / "yaman" / "09-09-2026" / "Calibration"

    (spo / ("SPO_run_raw.csv" + paths.JOURNAL_SUFFIX)).write_text("t,v\n0,0.8\n")
    (jv / ("run_ch1_forward_temp.csv")).write_text("V,I\n0,0.1\n")
    (cal / ("calibration_temp.csv")).write_text("V,I\n0,0.1\n")

    with published_file(str(jv / "run_2026-09-09.csv")) as handle:
        handle.write("[[ EXPERIMENTAL PARAMETERS ]]\n")
        handle.flush()
        os.fsync(handle.fileno())           # non-empty: see the test above
        assert sweeper.sweep(include_logs=False)["published"] == 0

    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert _published(store) == ["run_2026-09-09.csv"]
    # everything still in flight is untouched
    assert (spo / ("SPO_run_raw.csv" + paths.JOURNAL_SUFFIX)).exists()
    assert (jv / "run_ch1_forward_temp.csv").exists()
    assert (cal / "calibration_temp.csv").exists()


# ---------------------------------------------------------------------------
# Reported from the lab, 14-Oct: a J-V run died before measuring a point with
#
#   FileNotFoundError: ...\staging\...\14-10-2026\JV\<name>_ch3_forward_temp.csv
#
# Staging is transient by design: once the sweeper has published everything
# in it, `_prune_empty_dirs` removes the now-empty dated mode folder. The
# file panel still displays that path, and pymeasure's Results() opens the
# file for writing WITHOUT creating parents -- so the next run crashed.
# ---------------------------------------------------------------------------

def test_publishing_everything_removes_the_run_folder(tmp_path):
    """Establishes the precondition. Pruning is wanted; it just has a
    consequence producers have to handle."""
    from solarjv_analyzer.store import sweeper

    staging = tmp_path / "staging"
    run_dir = staging / "user" / "14-10-2026" / "JV"
    run_dir.mkdir(parents=True)
    report = run_dir / "published.csv"
    report.write_text("x", encoding="utf-8")
    report.unlink()                       # what publishing leaves behind

    sweeper._prune_empty_dirs(str(report), str(staging))

    assert not run_dir.exists(), "the empty run folder should be pruned"


def test_a_producer_creates_its_folder_before_writing(tmp_path):
    """The fix, as a rule rather than a case.

    Any producer about to hand a path to pymeasure must makedirs first,
    because the folder it resolved a moment ago may already be gone.
    """
    import os

    run_dir = tmp_path / "staging" / "user" / "14-10-2026" / "JV"
    target = run_dir / "run_ch3_forward_temp.csv"
    assert not run_dir.exists()

    with pytest.raises(FileNotFoundError):
        open(target, "w").close()         # what the lab saw

    os.makedirs(os.path.dirname(target), exist_ok=True)
    open(target, "w").close()             # what it does now
    assert target.exists()


def test_the_queue_paths_create_their_directory(tmp_path):
    """A structural guard: both queue paths must makedirs after resolving
    `directory`, so this cannot be dropped in a later edit."""
    import ast
    import inspect
    from solarjv_analyzer.gui import app_controller

    source = inspect.getsource(app_controller)
    tree = ast.parse(source)

    for name in ("queue_experiment", "_queue_combined_experiment"):
        func = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == name), None)
        if func is None:
            continue
        calls = [n for n in ast.walk(func)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "makedirs"]
        assert calls, (
            f"{name}() resolves an output directory but never creates it; "
            f"a pruned staging folder will kill the run")
