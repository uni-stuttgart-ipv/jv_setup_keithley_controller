#!/usr/bin/env python3
"""
prompt_builder.py — scenario-driven prompt construction for vision QA.

Pure Python (no `google-genai` import) so the prompt logic can be unit-tested
without an API key or network. `tools/vision_inspector.py` calls this to turn
a *scenario* + the live design tokens into a grounding-first, evidence-forcing
prompt, instead of shipping a hand-written prompt that leads the model to
confirm whatever it was told to expect.

Why grounding-first matters
---------------------------
A naive prompt ("controls must be teal #053a46 — are they?") primes the vision
model with the answer and produces confident-but-wrong "yes, teal" reports even
when a button is blue. Every prompt built here therefore runs in three ordered
phases inside a single call:

    1. ground      — neutral inventory of what is on screen (locations + observed
                     colors), BEFORE any rule is revealed;
    2. judge       — compare that inventory against the injected design tokens;
    3. synthesize  — a structured JSON verdict list (not prose), so a claim is a
                     machine-checkable field rather than a sentence.

Design rules are injected from `gui/theme/tokens.py` (the single source of
truth) at build time — never hand-typed here — so the rules cannot drift out of
sync with the design system.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Scenario names are data (metadata + a builder), so adding a new scenario is a
# new entry here — no branching deep inside vision_inspector.py.
SCENARIOS = {
    "style-check": "Design-system color/status compliance (teal controls, green status-only, red errors).",
    "layout-check": "Alignment, clipping/overlap, spacing, and combined-view plot symmetry.",
    "match-reference": "Compare a candidate render against a reference image and list differences.",
    "consistency-check": "Inventory N screens of one flow, then flag styling drift between screens.",
}


def _load_tokens():
    """Import the design tokens from the single source of truth. Done lazily so
    `--help`, raw mode, and unit tests never need the theme package on the path."""
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from solarjv_analyzer.gui.theme import tokens

    return tokens


def render_auth_rules() -> str:
    """Rules for the auth/login flow, which is deliberately EXCLUDED from the
    main design system (it keeps its own dark navy + green palette). The
    canonical values below mirror src/solarjv_analyzer/auth/login_dialog.py.

    Without this, style-check evaluates auth screens against the main-app
    teal rule and emits a single false "green should be teal" complaint on
    every screen — while missing the real drift (muted text, button radius,
    danger color) because the main-app spec never mentions those tokens."""
    return "\n".join(
        [
            "AUTH FLOW DESIGN RULES (source: auth/login_dialog.py — the auth flow is",
            "deliberately EXCLUDED from the main app theme, so teal #053a46 does NOT",
            "apply here):",
            "",
            "Color semantics:",
            "- Primary action buttons (Sign In, Register, Send Verification Code, Verify,",
            "  Set New Password) MUST be green #10b981 with white text; hover #34d399,",
            "  pressed #059669. Green-on-navy is CORRECT here — do not flag it as a",
            "  teal violation.",
            "- Surfaces: window #0f172a, card #1e293b (dark navy).",
            "- Text: primary #f1f5f9, muted #94a3b8.",
            "- Inputs: background rgba(30,41,59,0.8), border rgba(148,163,184,0.15),",
            "  radius 10px, focus border 1px #10b981.",
            "- Error / danger text: #f87171.",
            "- Primary button: radius 10px, font-size 15px, padding 14px 0.",
            "- Secondary/link buttons (Cancel, Resend Code, Forgot password?):",
            "  transparent background, muted #94a3b8 text, hover #f1f5f9.",
        ]
    )


def render_rules(tok) -> str:
    """Render the design rules as a text block from a tokens module (or any
    object exposing the same attributes — unit tests pass a fake)."""
    return "\n".join(
        [
            "DESIGN RULES (source of truth: gui/theme/tokens.py):",
            "",
            "Color semantics:",
            f"- Interactive controls (buttons, toggles, checkboxes, selected tabs) "
            f"MUST be teal {tok.PRIMARY}.",
            f"- Green {tok.OK_GREEN} is STATUS ONLY (instrument-connected dots, PASS "
            f"states). It must NEVER fill a button/toggle/checkbox.",
            f"- Red {tok.ERROR} (bright variant {tok.ERROR_BRIGHT}) is for errors, "
            f"abort, and logout only.",
            f"- Backgrounds use the neutral surface ladder ({tok.SURFACE}, "
            f"{tok.SURFACE_LOWEST}, {tok.SURFACE_LOW}, ...).",
            "",
            "Spacing & shape:",
            f"- 4px baseline grid; spacing tokens XS={tok.SPACING_XS} SM={tok.SPACING_SM} "
            f"MD={tok.SPACING_MD} LG={tok.SPACING_LG} XL={tok.SPACING_XL}.",
            f"- Corner radii: SM={tok.RADIUS_SM}, card={tok.RADIUS}, pill={tok.RADIUS_PILL}.",
            "",
            "A control filled green or blue (not teal) is a DEFECT. A status dot "
            "filled teal or red (not green) is a DEFECT.",
        ]
    )


# The JSON shape every scenario's synthesize step must emit. Kept as a single
# constant so the prompt and the parser in vision_inspector.py stay in sync.
JSON_VERDICT_SCHEMA = (
    'Respond with a single JSON object (no markdown fences, no prose outside it) of the form:\n'
    '{"scenario": "<name>", '
    '"verdicts": [{"location": "<where>", "widget": "<what>", '
    '"observed": "<color/state you actually see>", "expected": "<rule it must satisfy>", '
    '"pass": true|false, "confidence": "high|medium|low", '
    '"evidence": "<what in the image supports this>"}], '
    '"summary": "<one or two sentences>"}'
)

JSON_DIFF_SCHEMA = (
    'Respond with a single JSON object (no markdown fences, no prose outside it) of the form:\n'
    '{"scenario": "match-reference", '
    '"diffs": [{"element": "<what>", "candidate": "<candidate value>", '
    '"reference": "<reference value>", "severity": "high|medium|low", '
    '"action": "<concrete change that closes the gap>"}], '
    '"summary": "<one or two sentences>"}'
)

JSON_CONSISTENCY_SCHEMA = (
    'Respond with a single JSON object (no markdown fences, no prose outside it) of the form:\n'
    '{"scenario": "consistency-check", '
    '"diffs": [{"element": "<shared role, e.g. primary action button or muted text>", '
    '"candidate": "<observed on the deviating screen(s)>", '
    '"reference": "<canonical value from the rules>", "severity": "high|medium|low", '
    '"action": "<concrete change that closes the gap>"}], '
    '"summary": "<one or two sentences>"}'
)


def _style_check_prompt(rules: str) -> str:
    return f"""You are a visual QA reviewer for a rendered screenshot of a PyQt5 lab instrument.

Be scrupulously accurate. Report only what is actually visible in the image; say
"not visible" for anything you cannot see. Never fill in a plausible answer.

STEP 1 — NEUTRAL INVENTORY (do this before reading any rules below):
List every distinct UI element you can see: its location (e.g. "top-left",
"left sidebar", "right panel", "bottom bar"), its role (button, toggle,
checkbox, status dot, text label, plot canvas, tab, input field, table), and the
color you observe described in your own words (e.g. "dark teal", "green",
"blue", "red", "white", "light gray") — do NOT assign hex codes yet. Include
every colored element, even small status dots and checkboxes.

STEP 2 — JUDGE (now apply the rules below to the inventory from Step 1):

{rules}

For each item from Step 1, record whether it complies or violates. Cover both:
report violations AND the items that pass, so it is clear each element was
checked. A control (button/toggle/checkbox) filled green or blue is a violation;
a status dot filled teal or red is a violation.

STEP 3 — SYNTHESIZE:
{JSON_VERDICT_SCHEMA}"""


def _layout_check_prompt(rules: str) -> str:
    return f"""You are a visual QA reviewer for a rendered screenshot of a PyQt5 lab instrument.

Be scrupulously accurate. Report only what is actually visible in the image; say
"not visible" for anything you cannot see. Never fill in a plausible answer.

STEP 1 — NEUTRAL INVENTORY (before reading any rules below):
Describe the layout: list every panel, heading, plot canvas, button row, and
table, with its approximate region (top / bottom / left / right / center). Note
any element that looks clipped (text cut off), overlapping another element, or
misaligned with its neighbor. If two plot canvases sit side by side, note whether
they appear equal in size and vertically aligned.

STEP 2 — JUDGE (now apply the rules below to the inventory from Step 1):

{rules}
Layout rules:
- Side-by-side plot canvases must be pixel-matched: equal width, aligned top and
  bottom edges.
- No text or widget may be clipped or overlapping.

For each region/panel from Step 1, record whether it complies or violates.

STEP 3 — SYNTHESIZE:
{JSON_VERDICT_SCHEMA}"""


def _match_reference_prompt(rules: str) -> str:
    return f"""You are comparing two rendered screenshots of the same PyQt5 application.

IMAGE 1 (attached first) is the CANDIDATE — a newly built UI.
IMAGE 2 (attached second) is the REFERENCE — the target it should resemble.

Be scrupulously accurate. Report only what is actually visible.

STEP 1 — NEUTRAL INVENTORY:
Describe the candidate (IMAGE 1) in detail: layout structure, panels, controls,
and the colors you observe on them.

STEP 2 — COMPARE:
Compare IMAGE 1 to IMAGE 2 and list every difference you can see: structural
(missing/extra panels, moved elements), color (a control that should be teal but
is a different color), spacing, and text. For each difference, state what the
candidate has versus what the reference has.

{rules}

STEP 3 — SYNTHESIZE:
{JSON_DIFF_SCHEMA}"""


def _consistency_check_prompt(rules: str) -> str:
    return f"""You are a visual QA reviewer comparing MULTIPLE rendered screenshots of the
SAME PyQt5 application's authentication flow (login, register, forgot-password,
OTP verification). Each attached image shows a DIFFERENT step of one flow, and
they must share a single visual language. The images are labeled IMAGE 1..N in
the order attached.

Be scrupulously accurate. Report only what is actually visible; say "not
visible" for anything you cannot see. Never fill in a plausible answer.

STEP 1 — NEUTRAL INVENTORY (per image, BEFORE reading any rules below):
For EACH image, list its distinct UI elements and the color/size you observe in
your own words (no hex codes yet). For every screen, cover at minimum:
- the primary action button (its label, fill color, text color, corner radius),
- text input fields (background, border, focus border, corner radius),
- secondary/link buttons ("Cancel", "Resend Code", "Forgot password?"),
- error text (if visible) and its color,
- heading and subtitle text (size, weight, color),
- the surface / card background color.

STEP 2 — CROSS-SCREEN CONSISTENCY DIFF:
Treat elements that play the SAME role across screens as one shared element
(e.g. "the primary action button" appears on every screen; "the muted subtitle
text" appears on several). For each shared role, compare its observed color,
border-radius, font-size, and padding across screens. Report every role whose
styling DIFFERS between screens — a difference is a consistency defect
regardless of which screen is "correct".

STEP 3 — JUDGE AGAINST CANONICAL PALETTE:
Now apply the rules below: for each drifting role, identify which screen(s)
match the canonical palette and which deviate.

{rules}

STEP 4 — SYNTHESIZE:
{JSON_CONSISTENCY_SCHEMA}"""


def build_prompt(
    scenario: str,
    reference: str | None = None,
    tok=None,
    theme: str = "main",
) -> str:
    """Compose a grounding-first prompt for `scenario`.

    `reference` is required only for the match-reference scenario (the path is
    used for a human-readable label; the image bytes are attached separately).
    `tok` is the tokens module (or fake) — defaults to importing the real one.
    `theme` selects the spec to evaluate against: "main" (the design-system
    tokens in gui/theme/tokens.py) or "auth" (the login-flow palette).
    """
    if scenario not in SCENARIOS:
        raise ValueError(
            f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIOS)}"
        )
    if theme not in ("main", "auth"):
        raise ValueError(f"unknown theme {theme!r}; expected 'main' or 'auth'")

    if theme == "auth":
        rules = render_auth_rules()
    else:
        if tok is None:
            tok = _load_tokens()
        rules = render_rules(tok)

    if scenario == "style-check":
        return _style_check_prompt(rules)
    if scenario == "layout-check":
        return _layout_check_prompt(rules)
    if scenario == "match-reference":
        if not reference:
            raise ValueError("--reference is required for the match-reference scenario")
        return _match_reference_prompt(rules)
    if scenario == "consistency-check":
        return _consistency_check_prompt(rules)
    raise AssertionError("unreachable")  # guarded above


def build_adversarial_prompt(prior_json: str, scenario: str) -> str:
    """Build the second-pass prompt: try to refute the first pass's findings.

    Told to *default to retracting* anything it cannot re-confirm with its own
    inspection — this catches the model confirming its own plausible-but-wrong
    first impression."""
    return f"""You previously reviewed an image (scenario: {scenario}) and reported these findings:

{prior_json}

Now act as an adversarial verifier on the SAME image. Your job is to catch false
positives, not to be agreeable. For every finding above that claims a violation
(pass=false) or a difference, independently look at the image again and decide
whether it is ACTUALLY visible or was assumed/imagined.

Default to RETRACTING any claim you cannot see with your own inspection.

Re-emit JSON with the same schema as before, but:
- keep only findings you can confirm with visible evidence,
- append a "retracted" list: {{"retracted": [{{"element": "<what>",
  "reason": "<why you could not confirm it>"}}]}}."""
