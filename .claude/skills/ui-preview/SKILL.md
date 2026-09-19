---
name: ui-preview
description: Verify GUI changes without hardware or a display. Use after changing gui/, spo/spo_widget.py, windows/, or the theme - renders every window offscreen, saves PNG previews, and runs numeric PASS/FAIL geometry checks.
---

# Offscreen UI Verification

A GUI change is not done when it compiles. Verify by rendering.

## 1. Render (always)

    QT_QPA_PLATFORM=offscreen python3 tools/render_preview.py

Renders the main window (combined + Advanced) and the calibration window
offscreen, writes PNGs to `tools/previews/`, and prints numeric PASS/FAIL
checks. **Any FAIL is a broken change.** Numeric checks performed:

- Exactly ONE `BrowserWidget` / `AnalysisPanel` instance exists (the
  shared-widget rule — a duplicate leaves one view permanently blank).
- Combined-view JV and SPO canvases match within 12 px height / 4 px width (a
  regression here means `_set_plot_widget_chrome_visible` or
  `theme.style_plot` broke).
- Every controller-bound widget attribute still exists on both windows.

For a bug fix, green geometry checks plus `pytest tests/gui -q` are enough.
Stop here.

## 2. Look at it (only when the change is visual)

If you can see images, open the PNGs in `tools/previews/` directly — that is
the cheapest and most reliable visual check available, and it needs no API key.

Reach for `tools/vision_inspector.py` only when (a) the backend model is
text-only and cannot see the PNGs, or (b) a deliberate color-token or layout
overhaul needs a second, adversarial opinion. Never to answer a question that
is readable from the code.

    python3 tools/vision_inspector.py tools/previews/main_window.png --scenario style-check
    python3 tools/vision_inspector.py tools/previews/main_window.png --scenario layout-check
    python3 tools/vision_inspector.py tools/previews/main_window.png --scenario match-reference --reference ref.png

- `style-check` — color/status compliance, rules imported live from
  `gui/theme/tokens.py`.
- `layout-check` — alignment, clipping/overlap, spacing, plot symmetry.
- `match-reference` — diff a render against a reference image.
- `--verify` adds a second adversarial pass that retracts what it cannot
  re-confirm: ~2× cost and latency. Worth it for a redesign, wasted on
  iteration.

Scenario mode builds a grounding-first prompt from the live design tokens;
prefer it over a hand-written prompt, which leads the model to confirm whatever
it was told to expect. Output is a human summary plus structured JSON
(`--json-only`). Needs `GEMINI_API_KEY` or `GOOGLE_API_KEY`; the script
auto-discovers and caches a current Gemini Flash vision model.

A teal/green/red violation it flags is a real defect even when every numeric
check is green.

## Rules it is checking against

Full detail in [`docs/ui.md`](../../../docs/ui.md). The short version:

- All colors come from `gui/theme/tokens.py`. Grep your diff for raw hex before
  finishing; only theme files may define them.
- QSS has no letter-spacing — use `theme.fonts.label_font()`.
- Teal = interactive controls. Green = STATUS ONLY (connected dots, PASS).
  Red = errors/abort/logout. Never a green button.
- A per-widget stylesheet drops inherited box-model properties — restate
  padding and border.
