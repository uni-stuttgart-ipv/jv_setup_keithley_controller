"""
LEGACY COMPATIBILITY SHIM — the design system now lives in gui/theme/.

Every name that windows/widgets historically imported from this module is
preserved, but all values are generated from gui.theme tokens ("Precision
Instrument Interface", DESIGN.md). Editing colors HERE is wrong — edit
gui/theme/tokens.py so the whole application stays in sync.

The login dialog (auth/) deliberately keeps its own dark styling and does
not consume these tokens.
"""

from solarjv_analyzer.gui.theme import tokens as _t
from solarjv_analyzer.gui.theme.stylesheet import (
    base_stylesheet as _base_stylesheet,
    dialog_stylesheet as _dialog_stylesheet,
    checkbox_stylesheet as _checkbox_stylesheet,
)

# -------------------------------------------------------------------------
# Legacy names → theme tokens
# -------------------------------------------------------------------------
FONT_FAMILY = "'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

COLOR_TEXT = _t.ON_SURFACE
COLOR_TEXT_MUTED = _t.ON_SURFACE_VARIANT
COLOR_TEXT_HEADER = _t.ON_SURFACE
COLOR_BORDER = _t.SURFACE_VARIANT
COLOR_BORDER_INPUT = _t.OUTLINE_VARIANT
COLOR_BG = _t.SURFACE_LOWEST
COLOR_BG_SUBTLE = _t.SURFACE_LOW
COLOR_BG_HOVER = _t.SURFACE_CONTAINER

# "Blue" accents are now the design system's slate teal primary.
COLOR_ACCENT_BLUE = _t.PRIMARY
COLOR_ACCENT_BLUE_HOVER = _t.PRIMARY_CONTAINER
COLOR_ACCENT_GREEN = _t.OK_GREEN
COLOR_ACCENT_GREEN_HOVER = "#059669"
COLOR_ACCENT_RED = _t.ERROR_BRIGHT
COLOR_ACCENT_RED_HOVER = "#dc2626"
COLOR_TOGGLE_GREEN = _t.OK_GREEN_DARK

SPACING_XS = _t.SPACING_XS
SPACING_SM = _t.SPACING_SM
SPACING_MD = _t.SPACING_MD
SPACING_LG = _t.SPACING_LG

# -------------------------------------------------------------------------
# Stylesheets (generated from theme)
# -------------------------------------------------------------------------
CHECKBOX_STYLESHEET = _checkbox_stylesheet()
BASE_STYLESHEET = _base_stylesheet()
DIALOG_STYLESHEET = _dialog_stylesheet()
