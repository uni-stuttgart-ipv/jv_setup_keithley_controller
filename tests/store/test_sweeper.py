"""What the sweeper decides to publish, and what it leaves alone."""
import os
import time

import pytest

from solarjv_analyzer.store import paths, sweeper


@pytest.fixture
def staging(tmp_path, monkeypatch):
    root = tmp_path / "staging"
    (root / "yaman" / "09-09-2026" / "JV").mkdir(parents=True)
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(root))
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setattr(sweeper, "extra_target", lambda: "")
    monkeypatch.setattr(sweeper, "_logs_dir", lambda: "")
    from solarjv_analyzer.store import identity
    # `active_user()` reads `_app_user` — the APPLICATION login. `_cached`
    # belongs to `windows_user()`, and patching it left active_user()
    # raising "nobody is signed in", so every sweep below returned early
    # and published nothing.
    monkeypatch.setattr(identity, "_app_user", "st000000")
    return root


def stage(staging, name, text="a,b\n1,2\n", age_seconds=60):
    path = staging / "yaman" / "09-09-2026" / "JV" / name
    path.write_text(text)
    old = time.time() - age_seconds
    os.utime(path, (old, old))
    return path


def test_temp_files_are_never_published(staging):
    stage(staging, "run_ch1_forward_temp.csv")
    assert sweeper.find_finished(str(staging)) == []


def test_a_live_journal_is_left_alone(staging):
    """A running hold appends to `*.journal`, which is never a candidate.

    This used to rely on the file having been touched within the quiet
    window. It no longer does: the name says the writer has not finished, so
    even a journal that has been idle for an hour between samples is skipped.
    """
    stage(staging, "hold_raw.csv" + paths.JOURNAL_SUFFIX, age_seconds=0)
    assert sweeper.find_finished(str(staging)) == []

    stage(staging, "slow_raw.csv" + paths.JOURNAL_SUFFIX, age_seconds=3600)
    assert sweeper.find_finished(str(staging)) == []


def test_a_report_being_written_is_left_alone(staging):
    """`published_file()` builds under `*.part`; a partial report is not ours."""
    stage(staging, "jv_report.csv" + paths.PART_SUFFIX, age_seconds=0)
    stage(staging, "old_report.csv" + paths.PART_SUFFIX, age_seconds=3600)
    assert sweeper.find_finished(str(staging)) == []


def test_the_quiet_window_still_applies_when_configured(staging):
    """The setting is retained as a fallback, so it must still work."""
    stage(staging, "legacy.csv", age_seconds=0)
    assert sweeper.find_finished(str(staging), quiet_seconds=30) == []
    assert sweeper.find_finished(str(staging), quiet_seconds=0) != []


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
    expected = tmp_path / "store" / "st000000" / "2026-09-09" / "JV" / "jv_report.csv"
    assert expected.read_text() == "voltage,current\n0.5,-0.01\n"
    assert sweeper.find_finished(str(staging)) == [], "staging copy was not removed"


def test_a_second_target_gets_an_identical_copy(staging, tmp_path, monkeypatch):
    second = tmp_path / "MyWork"
    second.mkdir()
    monkeypatch.setattr(sweeper, "extra_target", lambda: str(second))
    stage(staging, "jv_report.csv", text="payload\n")

    assert sweeper.sweep(include_logs=False)["published"] == 1

    store_copy = tmp_path / "store" / "st000000" / "2026-09-09" / "JV" / "jv_report.csv"
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
    published = tmp_path / "store" / "st000000" / "2026-09-09" / "JV" / "jv_report.csv"
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

    published = (tmp_path / "store" / "st000000" / "2026-09-09" / "JV"
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
    published = (tmp_path / "store" / "st000000" / "2026-09-09" / "JV"
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


# --- the janitor: what a crash leaves behind ---------------------------------

def test_a_stale_journal_is_rescued_and_published(staging, monkeypatch):
    """A crashed hold's data reaches the store, as it did before `.journal`."""
    stale = sweeper.STALE_IN_PROGRESS_SECONDS + 60
    stage(staging, "hold_raw.csv" + paths.JOURNAL_SUFFIX, age_seconds=stale)

    sweeper._tidy_in_progress(str(staging))

    folder = staging / "yaman" / "09-09-2026" / "JV"
    assert (folder / "hold_raw.csv").exists(), "the journal should be promoted"
    assert not (folder / ("hold_raw.csv" + paths.JOURNAL_SUFFIX)).exists()
    # and it is now a normal candidate
    assert [os.path.basename(p) for p in sweeper.find_finished(str(staging))] \
        == ["hold_raw.csv"]


def test_a_stale_part_file_is_deleted_not_published(staging):
    """A half-written report must never reach the store under a real name."""
    stale = sweeper.STALE_IN_PROGRESS_SECONDS + 60
    stage(staging, "jv_report.csv" + paths.PART_SUFFIX, age_seconds=stale)

    sweeper._tidy_in_progress(str(staging))

    folder = staging / "yaman" / "09-09-2026" / "JV"
    assert not (folder / "jv_report.csv").exists(), "a partial report was promoted"
    assert not (folder / ("jv_report.csv" + paths.PART_SUFFIX)).exists()
    assert sweeper.find_finished(str(staging)) == []


def test_the_janitor_leaves_a_live_hold_alone(staging):
    """Age is measured from the last write, so a long hold is not disturbed."""
    young = sweeper.STALE_IN_PROGRESS_SECONDS - 60
    stage(staging, "hold_raw.csv" + paths.JOURNAL_SUFFIX, age_seconds=young)

    sweeper._tidy_in_progress(str(staging))

    folder = staging / "yaman" / "09-09-2026" / "JV"
    assert (folder / ("hold_raw.csv" + paths.JOURNAL_SUFFIX)).exists()
    assert not (folder / "hold_raw.csv").exists()


def test_a_stale_journal_whose_report_exists_is_discarded(staging):
    """The run finished after all — the journal is a leftover, not data."""
    stale = sweeper.STALE_IN_PROGRESS_SECONDS + 60
    stage(staging, "hold_raw.csv" + paths.JOURNAL_SUFFIX, age_seconds=stale)
    stage(staging, "hold_raw.csv", age_seconds=stale)

    sweeper._tidy_in_progress(str(staging))

    folder = staging / "yaman" / "09-09-2026" / "JV"
    assert not (folder / ("hold_raw.csv" + paths.JOURNAL_SUFFIX)).exists()
    assert (folder / "hold_raw.csv").read_text() == "a,b\n1,2\n"


def test_an_idle_sweep_does_not_touch_the_staging_root(staging, monkeypatch):
    """Otherwise the file watcher and the sweeper feed each other for ever.

    `sweep()` ends by saving the publish journal, which lives in the staging
    root — a directory the sidecar watches. An unconditional save made every
    sweep schedule the next one about 300 ms later, spawning a QThread each
    time, for the life of the application.
    """
    from solarjv_analyzer.store import identity
    monkeypatch.setattr(identity, "_app_user", "st000000")

    sweeper.sweep(include_logs=False)           # may legitimately write once
    root = str(staging)
    before = {
        name: os.path.getmtime(os.path.join(root, name))
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name))
    }

    time.sleep(0.01)
    sweeper.sweep(include_logs=False)           # nothing changed since

    after = {
        name: os.path.getmtime(os.path.join(root, name))
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name))
    }
    assert after == before, (
        "an idle sweep rewrote a file in the watched staging root, which "
        "retriggers the watcher and loops")


# --- the folder rename ------------------------------------------------------

def test_a_legacy_main_folder_still_publishes_under_jv(tmp_path, monkeypatch):
    """Files staged by an older build, or left across an upgrade.

    The J-V mode folder used to be called "Main". Nothing writes that name any
    more, but a file sitting in staging when the app is upgraded still has to
    reach the store, and it belongs under the name the folder has now.
    """
    root = tmp_path / "staging"
    (root / "yaman" / "09-09-2026" / "Main").mkdir(parents=True)
    monkeypatch.setenv("SOLARJV_STAGING_ROOT", str(root))
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setattr(sweeper, "extra_target", lambda: "")
    monkeypatch.setattr(sweeper, "_logs_dir", lambda: "")
    from solarjv_analyzer.store import identity
    monkeypatch.setattr(identity, "_app_user", "st000000")

    legacy = root / "yaman" / "09-09-2026" / "Main" / "old_report.csv"
    legacy.write_text("a,b\n1,2\n")
    old = time.time() - 60
    os.utime(legacy, (old, old))

    assert sweeper.sweep(include_logs=False)["published"] == 1

    assert (tmp_path / "store" / "st000000" / "2026-09-09" / "JV"
            / "old_report.csv").exists(), "a legacy Main file should land under JV"
    assert not (tmp_path / "store" / "st000000" / "2026-09-09" / "Main").exists()
