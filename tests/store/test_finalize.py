"""The report lifecycle: a final name appears only when the file is complete.

These tests exist because the sweeper's correctness now rests entirely on that
promise. If `published_file()` ever lets a partial file reach its final name,
a truncated report is published to the store under a name nobody can tell
apart from a good one.
"""
import os

import pytest

from solarjv_analyzer.store import paths
from solarjv_analyzer.store.finalize import discard, promote, published_file


def test_the_final_name_appears_only_after_the_handle_closes(tmp_path):
    final = tmp_path / "report.csv"
    with published_file(str(final), newline="") as handle:
        handle.write("a,b\n1,2\n")
        assert not final.exists(), "the final name appeared before the close"
        assert (tmp_path / ("report.csv" + paths.PART_SUFFIX)).exists()
    assert final.exists()
    assert not (tmp_path / ("report.csv" + paths.PART_SUFFIX)).exists()


def test_the_content_matches_a_plain_open(tmp_path):
    body = "[[ HEADER ]]\nx,y\n1.5,2.5\n"
    via_helper = tmp_path / "helper.csv"
    with published_file(str(via_helper), newline="") as handle:
        handle.write(body)
    via_open = tmp_path / "plain.csv"
    with open(str(via_open), "w", newline="") as handle:
        handle.write(body)
    assert via_helper.read_bytes() == via_open.read_bytes()


def test_an_exception_leaves_nothing_behind(tmp_path):
    final = tmp_path / "report.csv"
    with pytest.raises(ValueError):
        with published_file(str(final)) as handle:
            handle.write("half a rep")
            raise ValueError("analysis blew up mid-write")
    assert not final.exists(), "a partial report reached its final name"
    assert not (tmp_path / ("report.csv" + paths.PART_SUFFIX)).exists()


def test_an_existing_report_is_never_overwritten(tmp_path):
    final = tmp_path / "report.csv"
    final.write_text("the original report")
    with pytest.raises(FileExistsError):
        with published_file(str(final)) as handle:
            handle.write("a different run")
    assert final.read_text() == "the original report"


def test_a_failed_rename_keeps_the_part_file(tmp_path):
    """The data must survive a promote() that cannot complete."""
    final = tmp_path / "report.csv"
    final.write_text("in the way")
    part = tmp_path / ("report.csv" + paths.PART_SUFFIX)
    with pytest.raises(FileExistsError):
        with published_file(str(final)) as handle:
            handle.write("new content")
    # published_file only discards when the body completed cleanly; here the
    # body succeeded and the rename failed, so the bytes are still on disk.
    assert part.exists()
    assert part.read_text() == "new content"


def test_missing_parent_directories_are_created(tmp_path):
    final = tmp_path / "yaman" / "09-09-2026" / "JV" / "report.csv"
    with published_file(str(final)) as handle:
        handle.write("x")
    assert final.exists()


def test_promote_refuses_an_occupied_name(tmp_path):
    part = tmp_path / ("r.csv" + paths.PART_SUFFIX)
    part.write_text("new")
    final = tmp_path / "r.csv"
    final.write_text("old")
    with pytest.raises(FileExistsError):
        promote(str(part), str(final))
    assert final.read_text() == "old"
    assert part.read_text() == "new"


def test_discard_is_quiet_about_a_missing_file(tmp_path):
    discard(str(tmp_path / "never-existed.part"))     # must not raise


def test_the_part_name_is_what_the_sweeper_excludes(tmp_path):
    """One definition, two users — the writer and the sweeper must agree."""
    final = tmp_path / "report.csv"
    with published_file(str(final)) as handle:
        handle.write("x")
        part_name = os.path.basename(str(final) + paths.PART_SUFFIX)
        assert paths.is_part_name(part_name)
    assert not paths.is_part_name(os.path.basename(str(final)))
