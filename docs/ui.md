# UI & theme — read before touching `gui/`, `windows/` or `spo/spo_widget.py`

## The design system lives in `gui/theme/`

`gui/theme/` is the single source of truth ("Precision Instrument Interface",
see DESIGN.md):

- `tokens.py` — colors, spacing, radii. Primary is slate teal `#053a46`, **not**
  blue. New colors go here, never inline.
- `fonts.py` — Geist/Inter/JetBrains Mono loader plus `label_font()` /
  `data_font()` factories. **QSS has no letter-spacing property** — use the
  factories when you need tracking.
- `stylesheet.py` — the base/dialog QSS builders.
- `plots.py` — pyqtgraph styling. `style_plot()` pins the left-axis width so
  side-by-side plots align.

`gui/style.py` and `gui/calibration_style.py` are **legacy shims** that
re-export theme values under the old names (`BASE_STYLESHEET`,
`COLOR_ACCENT_BLUE`, …). Never edit colors there — edit `theme/tokens.py`.

`DIALOG_STYLESHEET` is applied at the `QApplication` level so every popup
matches. `main.py` calls `load_design_fonts()` + `apply_global_plot_config()`
before any window is built.

Layout spacing uses the `SPACING_XS/SM/MD/LG` tokens, not hardcoded pixels.

## Color semantics

- **All interactive controls are teal** — buttons, toggles, checkboxes (plain
  fill, no glyph, checked = `#053a46`).
- **Green (`#10b981`) is reserved strictly for status**: instrument-connected
  dots, PASS states. Do not reintroduce green controls.
- **Red** is for errors, abort and logout.
- Named object selectors (`#QueueButton`, `#AbortButton`, `#ModeButton`,
  `#SavePlotButton`, `#BottomTabBar`) hold per-button overrides in
  `jv_analyzer_window.py`.
- The **login dialog (`auth/`) is deliberately excluded** from the theme — it
  keeps its own dark styling.

## Shared-widget rule

The bottom section (Experiment Queue browser + Channel Analysis panel) and
`plot_widget` are **single instances** shared between the combined and Advanced
views, reparented in `_on_mode_button_click`. Never instantiate a second copy
per view: the Manager and controller bind to
`self.browser_widget`/`self.analysis_panel`, and a duplicate leaves one view
permanently blank. Regression test:
`tests/gui/test_shared_bottom_section.py`.

## The Log page — one design in every view

Every graph area carries the same centred **Plot | Log** pill bar, built by
`_make_plot_log_switch()`: the JV+SPO graphs, the Advanced JV plot and the
Advanced SPO plot. Switching view never moves the log or changes how it looks.
The log replaces only the graph it sits above — the Experiment Queue below
stays visible — and the view's Save-plot button hides on the Log page.

pymeasure's `LogWidget` is **gone**, along with its handler and the
`_create_fallback_handler` path. It could not be used in more than one place:
`_blinking_start()` does `self.parent().parent()` and calls `tabBar()` on the
result, so it only survives inside a `QTabWidget`, and its blink state
(`_blink_qtimer`, `_blink_color`, `tab_index`) lives in **class** attributes,
so two instances fight over one timer. That parentless class-level QTimer is one
known dangling-pointer source (hence the `_revive_pymeasure_blink_timer`
workaround in `tests/conftest.py`), but removing the widget did **not** stop
the intermittent segfault — see G1 in AUDIT.md.

`gui/widgets/log_panel.py` replaces it:

- **Three `LogPanel` widgets, one handler.** Each view owns a panel, but
  `install_root_handler()` puts a single `QtLogHandler` on the root logger and
  every panel connects to its signal. A handler per panel would mean three
  formatting passes per record and three handlers left on the process-wide root
  logger each time `main.py`'s relogin loop builds a new window.
- **Shared history.** `_history` keeps the last `MAX_LINES` records, so a panel
  shown for the first time mid-session is not blank.
- **Records arrive on the measurement worker threads.** The handler is a
  `QObject` that emits a signal; the queued connection is what makes touching
  the widget legal. `_on_record` also catches `RuntimeError` and unsubscribes,
  because PyQt only auto-disconnects when the *Python* wrapper is collected,
  which can be long after the C++ object is gone.
- The emitter is a parentless QObject held by a module global, so
  `install_root_handler()` touches it and rebuilds if SIP collected it — the
  same failure mode the conftest works around for pymeasure's timer.

Severity is carried by colour **and** a fixed-width tag (`WARN`, `ERR `),
because colour alone does not survive a screenshot pasted into a bug report.
INFO gets a blank tag so the message column still lines up. A warning or error
arriving while the plots are showing marks that view's Log tab (`Log  •`, red)
until the operator opens it. Regression tests: `tests/gui/test_log_panel.py`.

## Two controls that lied about their state

- **Device architecture** (`ParameterTab`): n-i-p and p-i-n were two shades of
  the same teal, so the selected architecture was unreadable — and it flips the
  sign convention of every metric. p-i-n is now `tokens.ARCH_PIN_RED`, a
  *desaturated* red, deliberately not `ERROR_BRIGHT`: functional red means
  something is wrong, and an architecture choice never is. The active side's
  label carries the colour and the weight.
- **Hold-voltage source** (Advanced→SPO): Manual/Quick JV shared the generic
  `ModeButton` object name, whose inner-tab style paints the unselected button
  pale grey on transparent — indistinguishable from disabled, so operators did
  not realise Quick JV was a choice. They are now `#HoldModeButton`: outlined
  teal when unselected, solid teal when selected.

Both are pinned by pixel-reading tests in
`tests/gui/test_control_affordances.py`, not by asserting on QSS text.

## pyqtgraph's ViewBox registry must be pruned

`ViewBox.__init__` ends with `updateAllViewLists()`, which walks
`ViewBox.AllViews` and calls a method on **every** entry. That dictionary is
weak on the Python *wrapper*, not on the C++ object, so every ViewBox from a
closed window stays in it while anything still holds the wrapper. The JV+SPO
view registers four, and `main.py` builds a new window on each trip round the
relogin loop: measured, three open/close cycles left 12 of 12 entries pointing
at destroyed objects and the list only grew.

`theme.forget_dead_viewboxes()` drops them. `JVAnalyzerWindow` calls it at the
top of `_layout()` (before any plot exists) and in `closeEvent`, which holds
the registry at four. Regression test in
`tests/gui/test_control_affordances.py`.

## Combined-view plot symmetry

The JV and SPO canvases must stay pixel-matched.
`_set_plot_widget_chrome_visible(False)` hides the pymeasure `PlotWidget`'s
internal chrome (axis selectors, coordinates label, margins) in the combined
view and restores it for Advanced; both plots get `theme.style_plot()` (fixed
axis width). Do not add per-panel chrome that breaks the symmetry. A pymeasure
`PlotWidget` composite = selector row + PlotFrame (+ coordinates label) +
margins; hide/restore only via `_set_plot_widget_chrome_visible`.

## Qt/QSS traps (each cost real debugging time)

- QSS has **no letter-spacing**. Use the font factories.
- A **per-widget stylesheet drops inherited box-model properties** — always
  restate padding and border.
- `QPushButton.sizeHint()` **ignores child layouts** — use
  `setSizeConstraint(SetMinimumSize)`.
- A `QLabel` without word wrap forces its full text width as its minimum and can
  clip a whole panel.
- A `QLabel` holding a pixmap reports `heightForWidth`: a squeezed width also
  shrinks its layout height while the pixmap stays full size and clips on all
  four sides. Set `label.setMinimumSize(scaled.size())` and crop wide white
  margins before scaling.
- **Unhandled exceptions in Qt slots abort the whole process** (qFatal) —
  validate inputs *before* mutating state like `is_busy`.

## Verifying a UI change

```bash
QT_QPA_PLATFORM=offscreen python3 tools/render_preview.py
```

Renders every window offscreen, writes PNGs to `tools/previews/`, and prints
numeric PASS/FAIL geometry checks (plot alignment, widget existence) — no vision
needed. This is sufficient for layout and clipping bugs.

The Gemini vision pass is for **deliberate visual redesigns only** — a
color-token or layout overhaul a bounding-box check cannot judge. Never to
answer a question that's readable from the code. It needs `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`):

```bash
python3 tools/vision_inspector.py tools/previews/<window>.png --scenario style-check
```

`--verify` adds an adversarial second pass; `--scenario layout-check` and
`match-reference` also exist. See the `ui-preview` skill.
