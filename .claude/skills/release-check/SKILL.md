---
name: release-check
description: Final verification before committing, handing over work, or declaring a task complete in this repo. Scales the checks to what actually changed.
---

# Release Check

Run the checks the change earns — no more, no less. Skipping a check that
*applies* is how sign bugs shipped here; running all of them for a typo is how
a five-minute fix becomes an hour.

## Always

1. **Compile what you touched**
   `python3 -m py_compile $(git diff --name-only --diff-filter=ACM | grep '\.py$')`
2. **The area's tests** — the matching row from the table in CLAUDE.md
   § "Verify what you changed" (all pytest runs need `QT_QPA_PLATFORM=offscreen`).
3. **The whole suite** — `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/ -q`
   Cheap (~6-16 s). Run it before any hand-over, commit, or "done", even when
   the change was narrow.
4. **State the remaining risk in one line.** Hardware timing, abort latency and
   instrument behaviour cannot be verified from a sandbox — say so plainly
   rather than implying a hardware path is proven.

## When `gui/`, `windows/`, `spo/spo_widget.py` or `gui/theme/` changed

5. `QT_QPA_PLATFORM=offscreen python3 tools/render_preview.py` — every geometry
   check must print PASS.
6. **Look at the PNGs** in `tools/previews/` if you can see images — that is the
   cheapest visual check there is. Only fall back to
   `tools/vision_inspector.py --scenario style-check` (add `--verify` for a
   deliberate redesign) when you cannot, or when a color/layout judgement needs
   a second opinion. See the `ui-preview` skill. This is **not** required for a
   bug fix whose geometry checks are green.
7. `grep -nE '#[0-9a-fA-F]{6}'` your diff — raw hex outside `gui/theme/` belongs
   in `tokens.py`.

## When `analysis/` or `spo/` math changed

8. Confirm the `physics-change` skill was actually followed: the new test came
   first, failed against the old code, and its docstring derives the expected
   value analytically — not from the implementation's output.

## Docs

9. Update only what the change invalidated: README § "Solar Cell Metrics" for a
   formula, `docs/metrics.md` / `docs/ui.md` / `docs/architecture.md` for a
   contract, `CLAUDE.md` if a hard rule moved, and `AUDIT.md` if a finding was
   fixed or discovered (one row — the narrative lives in
   `docs/audit-2026-08.md`).
