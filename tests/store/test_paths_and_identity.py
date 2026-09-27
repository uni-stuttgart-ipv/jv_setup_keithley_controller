"""Path arithmetic and identity resolution."""
import os
from datetime import datetime

import pytest

from solarjv_analyzer.store import identity, paths


# --- identity -------------------------------------------------------------

def test_identity_is_not_taken_from_the_environment(monkeypatch):
    """%USERNAME% is user-writable; the folder name is the attribution."""
    monkeypatch.setattr(identity, "_cached", "")
    monkeypatch.setattr(identity, "_from_token", lambda: "IPV\\st000000")
    monkeypatch.setenv("USERNAME", "i-said-so")
    assert identity.windows_user(refresh=True) == "st000000"


@pytest.mark.parametrize("raw, expected", [
    ("IPV\\st000000", "st000000"),
    ("st000000@stud.uni-stuttgart.de", "st000000"),
    ("  st000000  ", "st000000"),
    ("weird/name", "name"),
    ("bad:name*here", "bad_name_here"),
])
def test_account_names_reduce_to_a_safe_folder_name(raw, expected):
    assert identity._sanitise(raw) == expected


def test_refuses_to_invent_an_identity(monkeypatch):
    monkeypatch.setattr(identity, "_cached", "")
    monkeypatch.setattr(identity, "_from_token", lambda: "")
    monkeypatch.setattr(identity.getpass, "getuser", lambda: "")
    with pytest.raises(RuntimeError):
        identity.windows_user(refresh=True)


# --- paths ----------------------------------------------------------------

def test_store_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLARJV_STORE_ROOT", str(tmp_path))
    assert paths.store_root() == str(tmp_path)


def test_destination_layout():
    got = paths.destination_dir("st000000", "2026-09-09", "Main", root="R")
    assert got == os.path.join("R", "st000000", "2026-09-09", "Main")


def test_store_date_is_iso():
    assert paths.store_date(datetime(2026, 9, 9)) == "2026-09-09"


@pytest.mark.parametrize("name, is_temp", [
    ("jv_2026-09-09_ch1_forward_temp.csv", True),
    ("jv_2026-09-09_ch1_reverse_temp.csv", True),
    ("jv_2026-09-09_ch1.csv", False),
    ("spo_2026-09-09_raw.csv", False),
    ("spo_2026-09-09_report.csv", False),
])
def test_temp_files_are_recognised(name, is_temp):
    """Excluding temps by name is also what makes an aborted JV run publish
    nothing: an abort never reaches the merge step."""
    assert paths.is_temp_name(name) is is_temp


def test_staged_path_carries_the_run_date_and_mode(tmp_path):
    """DirectoryManager writes dd-mm-yyyy; the store needs yyyy-mm-dd.

    Taking the date from the folder the app created — rather than from today —
    is what stops a run that crosses midnight from being split.
    """
    staged = tmp_path / "yaman" / "09-09-2026" / "SPO" / "hold_report.csv"
    staged.parent.mkdir(parents=True)
    staged.write_text("x")

    date_str, mode = paths.classify_staged(str(staged), str(tmp_path))

    assert (date_str, mode) == ("2026-09-09", "SPO")


def test_unexpected_layout_falls_back_to_the_file_date(tmp_path):
    staged = tmp_path / "loose.csv"
    staged.write_text("x")
    date_str, mode = paths.classify_staged(str(staged), str(tmp_path))
    expected = datetime.fromtimestamp(staged.stat().st_mtime).strftime("%Y-%m-%d")
    assert (date_str, mode) == (expected, "Main")


def test_log_date_comes_from_the_log_filename():
    assert paths.log_date("session_2026-09-09_10-58-42.log") == "2026-09-09"


# ---------------------------------------------------------------------------
# Application login identity (replaces the Windows account for store paths)
# ---------------------------------------------------------------------------

def test_app_user_is_sanitised_like_a_windows_name():
    """A username is free text typed at registration — it must not be able to
    carry path separators into the store tree."""
    try:
        assert identity.set_app_user(r"LAB\st195620") == "st195620"
        assert identity.set_app_user("st195620@uni.de") == "st195620"
        # Separators are not escaped, they are STRIPPED — only the last
        # segment survives, so a username can never walk out of the tree.
        assert identity.set_app_user("a/b\\c") == "c"
        assert identity.set_app_user("../../etc") == "etc"
    finally:
        identity.set_app_user("")


def test_active_user_refuses_when_nobody_is_signed_in():
    """No fallback to the Windows account: publishing to a different folder
    than the operator expects is worse than not publishing."""
    identity.set_app_user("")
    with pytest.raises(RuntimeError):
        identity.active_user()


def test_active_user_follows_the_current_login():
    """The re-login loop can serve several operators in one process, so the
    identity must NOT be cached the way the process identity is."""
    try:
        identity.set_app_user("st000001")
        assert identity.active_user() == "st000001"
        identity.set_app_user("st000002")
        assert identity.active_user() == "st000002"     # not the first one
    finally:
        identity.set_app_user("")


def test_logout_clears_the_identity():
    try:
        identity.set_app_user("st000001")
        assert identity.app_user() == "st000001"
        identity.set_app_user("")
        assert identity.app_user() == ""
    finally:
        identity.set_app_user("")
