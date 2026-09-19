"""What the sweeper decides to publish, and what it leaves alone."""
import os
import time

import pytest

from solarjv_analyzer.store import paths, sweeper


@pytest.fixture
def staging(tmp_path, monkeypatch):
    root = tmp_path / "staging"
    (root / "yaman" / "09-09-2026" / "Main").mkdir(parents=True)
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(root))
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setattr(sweeper, "extra_target", lambda: "")
    monkeypatch.setattr(sweeper, "_logs_dir", lambda: "")
    from solarjv_analyzer.store import identity
    monkeypatch.setattr(identity, "_cached", "st000000")
    return root


def stage(staging, name, text="a,b\n1,2\n", age_seconds=60):
    path = staging / "yaman" / "09-09-2026" / "Main" / name
    path.write_text(text)
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


def test_temp_files_are_never_published(staging):
    stage(staging, "run_ch1_forward_temp.csv")
    assert sweeper.find_finished(str(staging)) == []


def test_a_file_still_being_written_is_left_alone(staging):
    """An SPO raw CSV grows every sampling interval, so a live hold is skipped."""
    stage(staging, "hold_raw.csv", age_seconds=0)
    assert sweeper.find_finished(str(staging)) == []


def test_an_empty_file_is_skipped(staging):
    stage(staging, "empty.csv", text="")
    assert sweeper.find_finished(str(staging)) == []


def test_a_settled_report_is_found(staging):
    path = stage(staging, "jv_report.csv")
    assert sweeper.find_finished(str(staging)) == [str(path)]


def test_sweep_publishes_then_clears_staging(staging, tmp_path):
    stage(staging, "jv_report.csv", text="voltage,current\n0.5,-0.01\n")

    summary = sweeper.sweep(include_logs=False)

    assert summary["published"] == 1 and summary["failed"] == 0
    expected = tmp_path / "store" / "st000000" / "2026-09-09" / "Main" / "jv_report.csv"
    assert expected.read_text() == "voltage,current\n0.5,-0.01\n"
    assert sweeper.find_finished(str(staging)) == [], "staging copy was not removed"


def test_a_second_target_gets_an_identical_copy(staging, tmp_path, monkeypatch):
    second = tmp_path / "MyWork"
    second.mkdir()
    monkeypatch.setattr(sweeper, "extra_target", lambda: str(second))
    stage(staging, "jv_report.csv", text="payload\n")

    assert sweeper.sweep(include_logs=False)["published"] == 1

    store_copy = tmp_path / "store" / "st000000" / "2026-09-09" / "Main" / "jv_report.csv"
    assert store_copy.read_text() == (second / "jv_report.csv").read_text() == "payload\n"


def test_an_unreachable_store_keeps_the_file_for_later(staging, tmp_path, monkeypatch):
    """Scenario C: the run completed, so the data must not be lost.

    The store is made unreachable by pointing it under a regular file, which
    fails for every user — a chmod would be bypassed by root and the test
    would pass for the wrong reason.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("i am a file")
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(blocker / "store"))
    path = stage(staging, "jv_report.csv")

    summary = sweeper.sweep(include_logs=False)
    assert summary["published"] == 0
    assert summary["pending"] == 1
    assert path.exists(), "the only copy was deleted before it was published"

    # ...and it publishes once the store comes back.
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    assert sweeper.sweep(include_logs=False)["published"] == 1
    assert not path.exists()
    published = tmp_path / "store" / "st000000" / "2026-09-09" / "Main" / "jv_report.csv"
    assert published.exists()


def test_extra_target_round_trips(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    sweeper.set_extra_target(str(tmp_path / "chosen"))
    assert sweeper.extra_target() == str(tmp_path / "chosen")
    sweeper.set_extra_target("")
    assert sweeper.extra_target() == ""


# ---------------------------------------------------------------------------
# Regression: 2026-09-09, first real run on the lab PC.
#
# app_controller builds temps as "{base}_{stamp}_ch{n}_{dir}_temp{ext}" where
# ext comes from splitext(the user's filename). A user who types "Test" gets
# NO extension, so the original `_temp.csv` suffix check missed them: live
# pymeasure temp files were published to the store, and one was deleted
# mid-sweep, which surfaced as KeyError: 'Voltage (V)' from pandas reading a
# file that had just vanished.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "Test_2026-09-09_15-48-46_ch1_forward_temp",       # the exact failing name
    "Test_2026-09-09_15-48-46_ch1_reverse_temp",
    "Test_2026-09-09_15-48-46_ch1_forward_temp.csv",
    "Test_2026-09-09_15-48-46_ch1_forward_temp.dat",
])
def test_temps_are_never_published_whatever_the_extension(staging, name):
    stage(staging, name)
    assert sweeper.find_finished(str(staging)) == [], (
        f"{name} would be published — it is a live pymeasure temp file"
    )


def test_a_real_report_without_an_extension_is_still_published(staging, tmp_path):
    """The flip side: "Test" also produces a FINAL file with no extension."""
    stage(staging, "Test_2026-09-09_15-48-46_ch1_2", text="voltage,current\n0.5,-0.01\n")

    assert sweeper.sweep(include_logs=False)["published"] == 1

    published = (tmp_path / "store" / "st000000" / "2026-09-09" / "Main"
                 / "Test_2026-09-09_15-48-46_ch1_2")
    assert published.read_text() == "voltage,current\n0.5,-0.01\n"


def test_a_file_the_app_still_holds_open_is_left_alone(staging, monkeypatch):
    """The structural guard: no claim, no publish, no delete.

    Simulates the Windows behaviour where renaming a file another process has
    open fails. POSIX allows the rename, so the refusal is injected — what is
    being tested is the sequencing: a file that cannot be claimed must be left
    completely untouched, not published and not removed.
    """
    path = stage(staging, "jv_report.csv", text="half-written\n")

    def refuse(src, dst):
        raise OSError(32, "The process cannot access the file because it is "
                          "being used by another process")

    monkeypatch.setattr(sweeper.os, "replace", refuse)
    summary = sweeper.sweep(include_logs=False)

    assert summary["published"] == 0
    assert summary["pending"] == 1
    assert path.exists() and path.read_text() == "half-written\n"


def test_a_claim_left_by_an_interrupted_sweep_is_retried(staging, tmp_path):
    """A crash between claim and publish must not orphan the file."""
    path = stage(staging, "jv_report.csv" + paths.CLAIM_SUFFIX, text="payload\n",
                 age_seconds=0)          # fresh: claimed files skip the quiet check

    assert str(path) in sweeper.find_finished(str(staging))
    assert sweeper.sweep(include_logs=False)["published"] == 1

    # ...and it lands under its real name, not the claim name.
    published = (tmp_path / "store" / "st000000" / "2026-09-09" / "Main"
                 / "jv_report.csv")
    assert published.read_text() == "payload\n"
    assert not path.exists()


def test_a_failed_publish_gives_the_file_its_name_back(staging, tmp_path, monkeypatch):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("i am a file")
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(blocker / "store"))
    path = stage(staging, "jv_report.csv", text="payload\n")

    assert sweeper.sweep(include_logs=False)["published"] == 0

    assert path.exists(), "the file was left under its .publishing claim name"
    assert path.read_text() == "payload\n"
