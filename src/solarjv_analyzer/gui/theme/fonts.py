"""
Design fonts: Geist (headings), Inter (body), JetBrains Mono (data/labels).

All SIL OFL — bundle the .ttf/.otf files in resources/fonts/ for full
fidelity; graceful fallback to Segoe UI / Consolas otherwise. Letter
spacing is NOT expressible in QSS, so uppercase micro-labels must use the
QFont factories here.
"""

import logging
import os

from PyQt5 import QtGui

logger = logging.getLogger(__name__)

_FONTS_LOADED = False
_available_families = set()


def _fonts_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "resources", "fonts",
    )


def load_design_fonts() -> None:
    """Register bundled design fonts (idempotent). Call after QApplication
    exists — typically once in main.py."""
    global _FONTS_LOADED, _available_families
    if _FONTS_LOADED:
        return
    _FONTS_LOADED = True

    folder = _fonts_dir()
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".ttf", ".otf")):
                fid = QtGui.QFontDatabase.addApplicationFont(
                    os.path.join(folder, name)
                )
                if fid >= 0:
                    for fam in QtGui.QFontDatabase.applicationFontFamilies(fid):
                        _available_families.add(fam)
                        logger.info(f"Loaded design font: {fam}")

    _available_families |= set(QtGui.QFontDatabase().families())


def _resolve(preferred, fallbacks):
    for fam in (preferred, *fallbacks):
        if fam in _available_families:
            return fam
    return preferred  # let Qt substitute


def heading_family() -> str:
    return _resolve("Geist", ("Inter", "Segoe UI", "Helvetica Neue", "Arial"))


def body_family() -> str:
    return _resolve("Inter", ("Segoe UI", "Helvetica Neue", "Arial"))


def mono_family() -> str:
    return _resolve("JetBrains Mono", ("Consolas", "Courier New", "monospace"))


def label_font(size_px: int = 12, weight: int = QtGui.QFont.DemiBold) -> QtGui.QFont:
    """'Label SM': mono, for uppercase micro-labels, +8% letter spacing."""
    f = QtGui.QFont(mono_family())
    f.setPixelSize(size_px)
    f.setWeight(weight)
    f.setLetterSpacing(QtGui.QFont.PercentageSpacing, 108.0)
    return f


def data_font(size_px: int = 14, weight: int = QtGui.QFont.Normal) -> QtGui.QFont:
    """'Data Tabular': mono so fluctuating values do not jitter."""
    f = QtGui.QFont(mono_family())
    f.setPixelSize(size_px)
    f.setWeight(weight)
    return f
