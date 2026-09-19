"""
Tests for tools/prompt_builder.py — the scenario-driven prompt constructor.

These are pure (no google-genai, no theme import — a fake tokens object is
passed in) and assert the *structural* properties that buy accuracy: grounding
first, real hex values injected, JSON shape demanded, and the adversarial pass
defaulting to retraction.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prompt_builder as pb  # noqa: E402


class FakeTokens:
    """Mimics the attributes prompt_builder reads from gui/theme/tokens.py."""

    PRIMARY = "#053a46"
    OK_GREEN = "#10b981"
    ERROR = "#ba1a1a"
    ERROR_BRIGHT = "#ef4444"
    SURFACE = "#faf9fa"
    SURFACE_LOWEST = "#ffffff"
    SURFACE_LOW = "#f4f3f4"
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 12
    SPACING_LG = 16
    SPACING_XL = 24
    RADIUS_SM = 8
    RADIUS = 16
    RADIUS_PILL = 24


@pytest.fixture
def tok():
    return FakeTokens()


# --------------------------------------------------------------------------
# render_rules — real hex values from the tokens object
# --------------------------------------------------------------------------

def test_render_rules_injects_real_hex_values(tok):
    rules = pb.render_rules(tok)
    assert "#053a46" in rules
    assert "#10b981" in rules
    assert "#ba1a1a" in rules


def test_render_rules_encodes_green_status_only_rule(tok):
    rules = pb.render_rules(tok)
    assert "STATUS ONLY" in rules
    assert "control filled green or blue" in rules.lower() or "green or blue" in rules.lower()


# --------------------------------------------------------------------------
# build_prompt — grounding-first ordering (the anti-leading-the-witness fix)
# --------------------------------------------------------------------------

def test_style_check_rules_appear_after_neutral_inventory(tok):
    prompt = pb.build_prompt("style-check", tok=tok)
    ground_idx = prompt.index("STEP 1")
    rules_idx = prompt.index("#053a46")
    # The hex rule must appear in the JUDGE step, after the neutral inventory —
    # the model must commit to what it sees before being told what's correct.
    assert prompt.index("STEP 2") < rules_idx
    assert ground_idx < rules_idx
    assert "#053a46" not in prompt[: prompt.index("STEP 2")]


def test_layout_check_mentions_plot_symmetry(tok):
    prompt = pb.build_prompt("layout-check", tok=tok)
    assert "pixel-matched" in prompt
    assert "STEP 1" in prompt and "STEP 2" in prompt and "STEP 3" in prompt


def test_style_check_demands_json_schema(tok):
    prompt = pb.build_prompt("style-check", tok=tok)
    assert '"verdicts"' in prompt
    assert '"pass": true|false' in prompt


# --------------------------------------------------------------------------
# match-reference — requires a reference image
# --------------------------------------------------------------------------

def test_match_reference_requires_reference(tok):
    with pytest.raises(ValueError, match="--reference"):
        pb.build_prompt("match-reference", tok=tok)


def test_match_reference_has_diff_schema(tok):
    prompt = pb.build_prompt("match-reference", reference="ref.png", tok=tok)
    assert '"diffs"' in prompt
    assert "IMAGE 1" in prompt and "IMAGE 2" in prompt


# --------------------------------------------------------------------------
# unknown scenario
# --------------------------------------------------------------------------

def test_unknown_scenario_raises(tok):
    with pytest.raises(ValueError, match="unknown scenario"):
        pb.build_prompt("nope", tok=tok)


def test_unknown_theme_raises(tok):
    with pytest.raises(ValueError, match="unknown theme"):
        pb.build_prompt("style-check", tok=tok, theme="bogus")


# --------------------------------------------------------------------------
# theme="auth" — the auth flow palette, so green is not misfired as teal
# --------------------------------------------------------------------------

def test_auth_theme_injects_auth_palette_not_teal_rule(tok):
    prompt = pb.build_prompt("style-check", tok=tok, theme="auth")
    assert "#10b981" in prompt
    assert "#f1f5f9" in prompt
    assert "#94a3b8" in prompt
    # The main-app teal-mandate must not be the rule being applied.
    assert "MUST be teal" not in prompt


def test_auth_theme_declares_green_correct(tok):
    prompt = pb.build_prompt("style-check", tok=tok, theme="auth")
    assert "do not flag" in prompt.lower() or "CORRECT" in prompt


# --------------------------------------------------------------------------
# consistency-check — cross-screen drift scenario
# --------------------------------------------------------------------------

def test_consistency_check_prompt_structure(tok):
    prompt = pb.build_prompt("consistency-check", tok=tok)
    assert '"scenario": "consistency-check"' in prompt
    assert "STEP 1" in prompt and "STEP 2" in prompt and "STEP 3" in prompt
    # The diff must be grounded in a per-screen inventory before rules appear.
    assert prompt.index("STEP 1") < prompt.index("STEP 2")


def test_consistency_check_inventory_before_rules(tok):
    prompt = pb.build_prompt("consistency-check", tok=tok, theme="auth")
    # The canonical palette hex must appear only after the inventory/diff steps.
    assert prompt.index("STEP 1") < prompt.index("#10b981")


def test_consistency_check_needs_no_reference(tok):
    # Unlike match-reference, consistency-check must NOT require --reference.
    prompt = pb.build_prompt("consistency-check", tok=tok)
    assert "diffs" in prompt


# --------------------------------------------------------------------------
# adversarial prompt — defaults to retracting unverifiable claims
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# adversarial prompt — defaults to retracting unverifiable claims
# --------------------------------------------------------------------------

def test_adversarial_prompt_defaults_to_retraction():
    prior = '{"scenario": "style-check", "verdicts": [{"pass": false}]}'
    adv = pb.build_adversarial_prompt(prior, "style-check")
    assert prior in adv
    assert "RETRACTING" in adv
    assert '"retracted"' in adv
