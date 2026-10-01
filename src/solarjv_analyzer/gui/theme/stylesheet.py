"""
QSS builders — all generated from theme.tokens so every window stays in sync.

- base_stylesheet():        app-wide foundation (main window, SPO tab, widgets)
- dialog_stylesheet():      QApplication-level popup styling
- calibration_stylesheet(): calibration window + checklist dialog

Qt QSS limitations honoured: no letter-spacing (use theme.fonts factories),
no transitions, no box-shadow (depth = 1px borders + tonal layering, which
is the design system's stated preference anyway).
"""

from .tokens import (
    SURFACE, SURFACE_LOWEST, SURFACE_LOW, SURFACE_CONTAINER, SURFACE_HIGH,
    SURFACE_VARIANT, ON_SURFACE, ON_SURFACE_VARIANT, OUTLINE, OUTLINE_VARIANT,
    PRIMARY, PRIMARY_CONTAINER, PRIMARY_FIXED_DIM, SECONDARY,
    SECONDARY_CONTAINER, ON_PRIMARY, ERROR, ERROR_CONTAINER,
    OK_GREEN_DARK, OK_GREEN_DARKER, RADIUS_SM, RADIUS,
)
from .fonts import heading_family, body_family, mono_family


def checkbox_stylesheet() -> str:
    """Plain teal-filled checkbox (no glyph) — controls use the primary
    accent; green is reserved for STATUS indicators (dots, PASS states)."""
    return f"""
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: 5px;
        background-color: {SURFACE_LOWEST};
    }}
    QCheckBox::indicator:hover {{
        border-color: {OUTLINE};
    }}
    QCheckBox::indicator:checked {{
        background-color: {PRIMARY};
        border: 1px solid {PRIMARY};
        image: none;
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: {PRIMARY_CONTAINER};
        border-color: {PRIMARY_CONTAINER};
    }}
    """


def scrollbar_stylesheet() -> str:
    return f"""
    QScrollBar:vertical {{
        background: transparent; width: 8px; margin: 0;
    }}
    QScrollBar:horizontal {{
        background: transparent; height: 8px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {OUTLINE_VARIANT}; border-radius: 4px; min-height: 30px;
    }}
    QScrollBar::handle:horizontal {{
        background: {OUTLINE_VARIANT}; border-radius: 4px; min-width: 30px;
    }}
    QScrollBar::handle:hover {{ background: {OUTLINE}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        width: 0; height: 0; background: none;
    }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    """


def base_stylesheet() -> str:
    """App-wide foundation: Precision Instrument look on the existing
    widget structure (a re-skin — layouts are unchanged)."""
    body = body_family()
    mono = mono_family()

    return f"""
    QMainWindow {{ background-color: {SURFACE}; }}
    QWidget {{
        font-family: '{body}', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 13px;
        color: {ON_SURFACE};
        background-color: {SURFACE};
    }}
    """ + checkbox_stylesheet() + f"""
    /* Group Boxes (cards): white surface, 16px radius, 1px tonal border */
    QGroupBox {{
        font-weight: 600;
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS}px;
        margin-top: 20px;
        padding: 16px;
        background-color: {SURFACE_LOWEST};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {SECONDARY};
        left: 10px;
    }}

    /* Buttons */
    QPushButton {{
        font-weight: 600;
        border-radius: {RADIUS_SM}px;
        padding: 8px 16px;
        background-color: {SURFACE_LOWEST};
        border: 1px solid {OUTLINE_VARIANT};
        color: {ON_SURFACE};
    }}
    QPushButton:hover {{
        background-color: {SURFACE_LOW};
        border-color: {OUTLINE};
    }}
    QPushButton:pressed {{ background-color: {SURFACE_CONTAINER}; }}
    QPushButton:disabled {{
        background-color: {SURFACE_VARIANT};
        color: {OUTLINE};
        border: 1px solid {SURFACE_VARIANT};
    }}

    /* Tabs: teal-tinted selection (was mint green) */
    QTabWidget::pane {{
        border: 1px solid {SURFACE_VARIANT};
        background: {SURFACE_LOWEST};
        border-radius: {RADIUS_SM}px;
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        min-width: 92px;
        max-width: 92px;
        padding: 10px 6px;
        margin: 0px;
        border: none;
        border-bottom: 2px solid transparent;
        color: {SECONDARY};
        font-weight: 500;
        text-align: center;
    }}
    QTabBar::tab:hover {{
        color: {ON_SURFACE};
        background: {SURFACE_LOW};
    }}
    QTabBar::tab:selected {{
        color: {PRIMARY};
        background: {SECONDARY_CONTAINER};
        border-bottom: 2px solid {PRIMARY};
        border-top-left-radius: {RADIUS_SM}px;
        border-top-right-radius: {RADIUS_SM}px;
        font-weight: 600;
    }}

    /* Inputs: JetBrains Mono for data-entry precision (per DESIGN.md) */
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
        font-family: '{mono}', Consolas, 'Courier New', monospace;
        padding: 8px 12px;
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: {RADIUS_SM}px;
        background: {SURFACE_LOWEST};
        color: {ON_SURFACE};
        selection-background-color: {PRIMARY_FIXED_DIM};
        selection-color: {ON_SURFACE};
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
        border: 1px solid {PRIMARY};
        outline: none;
    }}
    /* Read-only inputs keep full-contrast text on a barely-tinted surface so
       an auto-calculated value (e.g. the NPLC readout) reads crisply instead
       of looking washed out — the non-editable state is hinted by the tint,
       never by dimming the value. */
    QLineEdit[readOnly="true"] {{
        background: {SURFACE_LOW};
        color: {ON_SURFACE};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    /* Scrollbars & splitters */
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    """ + scrollbar_stylesheet() + f"""
    QSplitter::handle {{
        background: {SURFACE_LOW};
        width: 4px;
        height: 4px;
    }}
    QSplitter::handle:hover {{ background: {OUTLINE_VARIANT}; }}

    /* Tables & lists */
    QTableWidget {{
        background: {SURFACE_LOWEST};
        gridline-color: {SURFACE_LOW};
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS_SM}px;
    }}
    QHeaderView::section {{
        background: {SURFACE_LOW};
        padding: 10px;
        border: none;
        border-bottom: 1px solid {SURFACE_VARIANT};
        font-weight: 600;
        color: {ON_SURFACE_VARIANT};
    }}
    QTreeWidget, QListWidget {{
        background: {SURFACE_LOWEST};
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS_SM}px;
    }}
    """


def dialog_stylesheet() -> str:
    """QApplication-level popup styling: every QMessageBox/QDialog in the
    app (login dialogs excluded — they set their own styles on top)."""
    body = body_family()

    return f"""
    QMessageBox, QInputDialog, QDialog {{
        background-color: {SURFACE_LOWEST};
        font-family: '{body}', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }}
    QMessageBox QLabel, QInputDialog QLabel {{
        color: {ON_SURFACE};
        font-size: 13px;
    }}
    QMessageBox QPushButton, QInputDialog QPushButton, QDialog QPushButton {{
        font-weight: 600;
        font-size: 12px;
        border-radius: {RADIUS_SM}px;
        padding: 8px 18px;
        min-width: 72px;
        background-color: {SURFACE_LOWEST};
        border: 1px solid {OUTLINE_VARIANT};
        color: {ON_SURFACE};
    }}
    QMessageBox QPushButton:hover, QInputDialog QPushButton:hover, QDialog QPushButton:hover {{
        background-color: {SURFACE_LOW};
        border-color: {OUTLINE};
    }}
    QMessageBox QPushButton:default, QInputDialog QPushButton:default {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border: none;
    }}
    QMessageBox QPushButton:default:hover, QInputDialog QPushButton:default:hover {{
        background-color: {PRIMARY_CONTAINER};
    }}
    """ + checkbox_stylesheet()
