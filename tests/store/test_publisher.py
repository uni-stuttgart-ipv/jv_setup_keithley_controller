"""The publish contract, against the real measured behaviour of S:\\Data\\JV.

The share refuses delete and refuses rename, but ALLOWS an existing file to be
rewritten — so `shutil.copyfile` would silently destroy someone's report.
Exclusive create is the only thing preventing that, and these tests are what
keep it in place.
"""
import os

import pytest

from solarjv_analyzer.store.publisher import (
    PublishError, publish_file, sha256_of, _is_partial_of,
)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return path


def test_publishes_and_verifies(tmp_path):
    src = write(str(tmp_path / "stage" / "r.csv"), "a,b\n1,2\n")
    dest = str(tmp_path / "store")

    written = publish_file(src, dest)

    assert os.path.dirname(written) == dest
    assert sha256_of(written) == sha256_of(src)


def test_a_different_file_is_never_overwritten(tmp_path):
    """The property that protects existing data on a share that allows writes."""
    dest = str(tmp_path / "store")
    original = write(os.path.join(dest, "r.csv"), "SOMEONE ELSE'S DATA\n")
    original_bytes = open(original, "rb").read()

    src = write(str(tmp_path / "stage" / "r.csv"), "my completely different data\n")
    written = publish_file(src, dest)

    assert os.path.basename(written) == "r_2.csv"
    assert open(original, "rb").read() == original_bytes, "existing file was clobbered"


def test_collision_counter_keeps_going(tmp_path):
    dest = str(tmp_path / "store")
    write(os.path.join(dest, "r.csv"), "one\n")
    write(os.path.join(dest, "r_2.csv"), "two\n")
    src = write(str(tmp_path / "stage" / "r.csv"), "three\n")

    assert os.path.basename(publish_file(src, dest)) == "r_3.csv"


def test_republishing_the_same_bytes_is_a_no_op(tmp_path):
    """Idempotent: a retry after a crash must not create `_2` duplicates."""
    src = write(str(tmp_path / "stage" / "r.csv"), "identical\n")
    dest = str(tmp_path / "store")

    first = publish_file(src, dest)
    second = publish_file(src, dest)

    assert first == second
    assert sorted(os.listdir(dest)) == ["r.csv"]


def test_an_interrupted_copy_is_repaired_in_place(tmp_path):
    """The store cannot delete, so a partial file must be rewritten, not skipped.

    Otherwise one dropped connection leaves a truncated report beside the good
    one, permanently, with nothing to tell them apart.
    """
    payload = "".join(f"{i},{i * 2}\n" for i in range(500))
    src = write(str(tmp_path / "stage" / "r.csv"), payload)
    dest = str(tmp_path / "store")
    partial = os.path.join(dest, "r.csv")
    write(partial, payload[:200])          # what an aborted copy leaves behind

    written = publish_file(src, dest)

    assert written == partial, "should have repaired the wreck, not sidestepped it"
    assert sha256_of(partial) == sha256_of(src)
    assert sorted(os.listdir(dest)) == ["r.csv"], "no duplicate was created"


def test_a_zero_byte_destination_counts_as_partial(tmp_path):
    src = write(str(tmp_path / "stage" / "r.csv"), "real content\n")
    dest = str(tmp_path / "store")
    empty = os.path.join(dest, "r.csv")
    write(empty, "")

    assert publish_file(src, dest) == empty
    assert open(empty).read() == "real content\n"


def test_longer_destination_is_not_treated_as_partial(tmp_path):
    src = write(str(tmp_path / "s" / "r.csv"), "short\n")
    longer = write(str(tmp_path / "d" / "r.csv"), "short\nplus more that we did not write\n")
    assert _is_partial_of(longer, src) is False


def test_publisher_never_deletes_or_replaces_in_the_store(tmp_path, monkeypatch):
    """A single stray os.remove here would be unrecoverable on the real share."""
    dest = str(tmp_path / "store")
    write(os.path.join(dest, "r.csv"), "existing\n")
    src = write(str(tmp_path / "stage" / "r.csv"), "new content\n")

    def forbidden(name):
        def guard(*args, **kwargs):
            raise AssertionError(f"publisher called os.{name} with {args!r}")
        return guard

    monkeypatch.setattr(os, "remove", forbidden("remove"))
    monkeypatch.setattr(os, "unlink", forbidden("unlink"))
    monkeypatch.setattr(os, "replace", forbidden("replace"))
    monkeypatch.setattr(os, "rename", forbidden("rename"))
    monkeypatch.setattr(os, "rmdir", forbidden("rmdir"))

    publish_file(src, dest)


def test_unusable_destination_raises_rather_than_lying(tmp_path):
    """A publish that cannot happen must raise, so staging is kept.

    The destination is made impossible by putting it *under a regular file*
    rather than by chmod: permission bits are bypassed by root, and CI here
    runs as root, so a chmod-based test silently passes for the wrong reason.
    """
    src = write(str(tmp_path / "stage" / "r.csv"), "data\n")
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("i am a file")
    dest = str(blocker / "store")

    with pytest.raises(PublishError):
        publish_file(src, dest)
