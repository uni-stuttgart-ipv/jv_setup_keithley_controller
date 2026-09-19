"""
Tests for tools/vision_inspector.py — the vision QA bridge.

Only the *pure* model-selection helpers are tested here (no network, no API
key, no google-genai import). The SDK import is lazy inside the module
specifically so this logic can be exercised without the package installed.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import vision_inspector as vi  # noqa: E402


# --------------------------------------------------------------------------
# _flash_version_tuple — stable Flash model name -> semantic version
# --------------------------------------------------------------------------

def test_flash_version_tuple_parses_stable():
    assert vi._flash_version_tuple("models/gemini-3.6-flash") == (3, 6)
    assert vi._flash_version_tuple("models/gemini-2.5-flash") == (2, 5)
    assert vi._flash_version_tuple("models/gemini-2.0-flash") == (2, 0)


def test_flash_version_tuple_rejects_aliases_and_variants():
    # "-latest" must NOT match — that's the experimental alias we refuse.
    assert vi._flash_version_tuple("models/gemini-2.5-flash-latest") is None
    assert vi._flash_version_tuple("models/gemini-2.5-flash-thinking") is None
    assert vi._flash_version_tuple("models/gemini-2.5-flash-preview-09-2025") is None
    assert vi._flash_version_tuple("models/gemini-2.5-pro") is None
    assert vi._flash_version_tuple("models/gemini-2.5-flash-lite") is None


def test_flash_version_tuple_missing_models_prefix():
    assert vi._flash_version_tuple("gemini-3.6-flash") is None


# --------------------------------------------------------------------------
# _select_newest_stable_flash — semantic (not float) comparison
# --------------------------------------------------------------------------

def test_select_newest_prefers_higher_semver():
    # 3.10 > 3.9 as a semantic version; a float compare would tie/collapse
    # both to 3.1 and pick arbitrarily.
    names = [
        "models/gemini-3.9-flash",
        "models/gemini-3.10-flash",
        "models/gemini-2.5-flash",
    ]
    assert vi._select_newest_stable_flash(names) == "gemini-3.10-flash"


def test_select_newest_skips_excluded_variants():
    names = [
        "models/gemini-2.5-flash-latest",   # alias — skip
        "models/gemini-2.5-flash-lite",     # lite — skip
        "models/gemini-2.5-flash-thinking", # thinking — skip
        "models/gemini-2.5-pro",            # pro — skip
        "models/gemini-2.0-flash",          # only valid candidate
    ]
    assert vi._select_newest_stable_flash(names) == "gemini-2.0-flash"


def test_select_newest_returns_none_when_nothing_matches():
    assert vi._select_newest_stable_flash([]) is None
    assert vi._select_newest_stable_flash(["models/gemini-2.5-pro"]) is None


def test_select_newest_strips_models_prefix():
    assert vi._select_newest_stable_flash(["models/gemini-3.0-flash"]) == "gemini-3.0-flash"


# --------------------------------------------------------------------------
# resolve_model — cache + fallback behaviour (fake client)
# --------------------------------------------------------------------------

class _FakeModels:
    def __init__(self, names):
        self._names = names

    def list(self):
        for n in self._names:
            yield type("M", (), {"name": n, "supported_actions": ["generateContent"]})()


class _FakeClient:
    def __init__(self, names):
        self.models = _FakeModels(names)


def test_resolve_model_logs_model_and_source(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(vi, "CACHE_TTL_SECONDS", 7 * 24 * 3600)

    model = vi.resolve_model(_FakeClient(["models/gemini-3.6-flash"]))
    assert model == "gemini-3.6-flash"
    err = capsys.readouterr().err
    assert "[vision_inspector] using model: gemini-3.6-flash (discovered)" in err


def test_resolve_model_falls_back_when_discovery_fails(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "CACHE_PATH", tmp_path / "cache.json")

    class _ExplodingModels:
        def list(self):
            raise RuntimeError("network down")

    class _BadClient:
        def __init__(self):
            self.models = _ExplodingModels()

    model = vi.resolve_model(_BadClient())
    assert model == vi.PINNED_FALLBACK_MODEL
    err = capsys.readouterr().err
    assert "(pinned fallback)" in err


def test_resolve_model_uses_cache_without_hitting_network(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache.json"
    cache.write_text('{"model": "gemini-3.0-flash", "resolved_at": %f}' % __import__("time").time())
    monkeypatch.setattr(vi, "CACHE_PATH", cache)
    monkeypatch.setattr(vi, "CACHE_TTL_SECONDS", 7 * 24 * 3600)

    class _NoNetworkClient:
        def __init__(self):
            self.models = _ExplodingModelsForCache()

    class _ExplodingModelsForCache:
        def list(self):
            raise AssertionError("network should not be called when cache is fresh")

    model = vi.resolve_model(_NoNetworkClient())
    assert model == "gemini-3.0-flash"
    assert "(cache)" in capsys.readouterr().err


def test_resolve_model_force_refresh_bypasses_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "cache.json"
    cache.write_text('{"model": "gemini-2.0-flash", "resolved_at": %f}' % __import__("time").time())
    monkeypatch.setattr(vi, "CACHE_PATH", cache)
    monkeypatch.setattr(vi, "CACHE_TTL_SECONDS", 7 * 24 * 3600)

    model = vi.resolve_model(
        _FakeClient(["models/gemini-3.6-flash"]), force_refresh=True
    )
    assert model == "gemini-3.6-flash"
    assert "(discovered)" in capsys.readouterr().err
