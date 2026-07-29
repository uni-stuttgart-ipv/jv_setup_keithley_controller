"""
Shared UI style constants for the SolarJV Analyzer application.

Centralises the modern flat color palette / font used across the main
window, the calibration window, and all popup dialogs (QMessageBox,
QDialog, QInputDialog, etc.) so the whole application looks consistent.

This module intentionally contains ONLY presentation (CSS-like Qt style
sheet strings) — no widget logic or behavior lives here.
"""

FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

# Core palette (kept in sync with jv_analyzer_window.py / calibration_window.py)
COLOR_TEXT = "#1e293b"
COLOR_TEXT_MUTED = "#64748b"
COLOR_TEXT_HEADER = "#0f172a"
COLOR_BORDER = "#e2e8f0"
COLOR_BORDER_INPUT = "#cbd5e1"
COLOR_BG = "#ffffff"
COLOR_BG_SUBTLE = "#f8fafc"
COLOR_BG_HOVER = "#f1f5f9"
COLOR_ACCENT_BLUE = "#2563eb"
COLOR_ACCENT_BLUE_HOVER = "#1d4ed8"
COLOR_ACCENT_GREEN = "#10b981"
COLOR_ACCENT_GREEN_HOVER = "#059669"
COLOR_ACCENT_RED = "#ef4444"
COLOR_ACCENT_RED_HOVER = "#dc2626"
COLOR_TOGGLE_GREEN = "#16a34a"


# -------------------------------------------------------------------------
# Shared checkbox styling: a plain green-filled square when checked
# (no checkmark glyph) instead of the OS-native check-tick icon, matching
# the app's green accent used throughout (toggle switches, selected tabs).
# -------------------------------------------------------------------------
CHECKBOX_STYLESHEET = f"""
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {COLOR_BORDER_INPUT};
        border-radius: 4px;
        background-color: {COLOR_BG};
    }}
    QCheckBox::indicator:hover {{
        border-color: #94a3b8;
    }}
    QCheckBox::indicator:checked {{
        background-color: {COLOR_TOGGLE_GREEN};
        border: 1px solid {COLOR_TOGGLE_GREEN};
        image: none;
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: #15803d;
        border-color: #15803d;
    }}
"""


# -------------------------------------------------------------------------
# Shared base rules reused by both the main window and calibration window.
# Each window still defines its own #ObjectName button variants on top of
# this, but the shared foundation (fonts, cards, inputs, tabs, scrollbars)
# guarantees the two windows look identical.
# -------------------------------------------------------------------------
BASE_STYLESHEET = f"""
    QMainWindow {{ background-color: {COLOR_BG}; }}
    QWidget {{
        font-family: {FONT_FAMILY};
        font-size: 13px;
        color: {COLOR_TEXT};
        background-color: {COLOR_BG};
    }}
""" + CHECKBOX_STYLESHEET + f"""
    /* Group Boxes (Cards) */
    QGroupBox {{
        font-weight: 600;
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        margin-top: 20px;
        padding: 16px;
        background-color: {COLOR_BG};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        color: {COLOR_TEXT_MUTED};
        left: 10px;
    }}

    /* Buttons */
    QPushButton {{
        font-weight: 500;
        border-radius: 6px;
        padding: 8px 16px;
        background-color: {COLOR_BG};
        border: 1px solid {COLOR_BORDER_INPUT};
        color: #334155;
    }}
    QPushButton:hover {{
        background-color: {COLOR_BG_SUBTLE};
        border-color: #94a3b8;
    }}
    QPushButton:pressed {{
        background-color: {COLOR_BG_HOVER};
    }}

    /* Modern Tabs */
    QTabWidget::pane {{
        border: 1px solid {COLOR_BORDER};
        background: {COLOR_BG};
        border-radius: 8px;
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
        color: {COLOR_TEXT_MUTED};
        font-weight: 500;
        text-align: center;
    }}
    QTabBar::tab:hover {{
        color: {COLOR_TEXT_HEADER};
        background: {COLOR_BG_HOVER};
    }}
    QTabBar::tab:selected {{
        color: #15803d;
        background: #dcfce7;
        border-bottom: 2px solid #16a34a;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: 600;
    }}

    /* Inputs */
    QLineEdit, QComboBox, QDoubleSpinBox {{
        padding: 8px 12px;
        border: 1px solid {COLOR_BORDER_INPUT};
        border-radius: 6px;
        background: {COLOR_BG};
        color: {COLOR_TEXT_HEADER};
        selection-background-color: #bfdbfe;
    }}
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
        border: 1px solid #3b82f6;
        outline: none;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    /* Scrollbars & Splitters */
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QSplitter::handle {{
        background: {COLOR_BG_HOVER};
        width: 4px;
        height: 4px;
    }}
    QSplitter::handle:hover {{
        background: {COLOR_BORDER_INPUT};
    }}

    /* Tables & Lists */
    QTableWidget {{
        background: {COLOR_BG};
        gridline-color: {COLOR_BG_HOVER};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
    }}
    QHeaderView::section {{
        background: {COLOR_BG_SUBTLE};
        padding: 10px;
        border: none;
        border-bottom: 1px solid {COLOR_BORDER};
        font-weight: 600;
        color: #475569;
    }}
"""


# -------------------------------------------------------------------------
# Global popup / dialog styling (QMessageBox, QDialog, QInputDialog, ...).
# Applied once at the QApplication level in main.py so every popup in the
# app — login, calibration checklist, warnings, confirmations — shares the
# same modern look, regardless of which window spawned it.
# -------------------------------------------------------------------------
DIALOG_STYLESHEET = f"""
    QMessageBox, QInputDialog, QDialog {{
        background-color: {COLOR_BG};
        font-family: {FONT_FAMILY};
    }}
    QMessageBox QLabel, QInputDialog QLabel {{
        color: {COLOR_TEXT};
        font-size: 13px;
    }}
    QMessageBox QPushButton, QInputDialog QPushButton, QDialog QPushButton {{
        font-family: {FONT_FAMILY};
        font-weight: 600;
        font-size: 13px;
        border-radius: 6px;
        padding: 8px 18px;
        min-width: 72px;
        background-color: {COLOR_BG};
        border: 1px solid {COLOR_BORDER_INPUT};
        color: #334155;
    }}
    QMessageBox QPushButton:hover, QInputDialog QPushButton:hover, QDialog QPushButton:hover {{
        background-color: {COLOR_BG_SUBTLE};
        border-color: #94a3b8;
    }}
    QMessageBox QPushButton:default, QInputDialog QPushButton:default {{
        background-color: {COLOR_ACCENT_BLUE};
        color: white;
        border: none;
    }}
    QMessageBox QPushButton:default:hover, QInputDialog QPushButton:default:hover {{
        background-color: {COLOR_ACCENT_BLUE_HOVER};
    }}
""" + CHECKBOX_STYLESHEET
