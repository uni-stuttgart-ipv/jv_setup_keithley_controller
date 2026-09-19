"""
Calibration UI stylesheet - now a thin layer over gui.theme (the single
source of truth for the design system). Kept as its own module so the
calibration window's imports stay stable; all tokens and font machinery
live in gui/theme/.
"""

from solarjv_analyzer.gui.theme.tokens import (  # noqa: F401 - re-exported
    SURFACE, SURFACE_LOWEST, SURFACE_LOW, SURFACE_CONTAINER, SURFACE_HIGH,
    SURFACE_VARIANT, ON_SURFACE, ON_SURFACE_VARIANT, OUTLINE, OUTLINE_VARIANT,
    PRIMARY, PRIMARY_CONTAINER, ON_PRIMARY, SECONDARY, ERROR, ERROR_CONTAINER,
    OK_GREEN, WARN_AMBER, WARN_AMBER_BG, RADIUS_SM, RADIUS,
)
from solarjv_analyzer.gui.theme.fonts import (  # noqa: F401 - re-exported
    load_design_fonts, heading_family, body_family, mono_family,
    label_font, data_font,
)


def calibration_stylesheet() -> str:
    """Full QSS for the calibration window + checklist dialog."""
    heading = heading_family()
    body = body_family()
    mono = mono_family()

    return f"""
    /* ---------- Base surfaces ---------- */
    QMainWindow, QDialog {{
        background-color: {SURFACE};
    }}
    QWidget {{
        font-family: '{body}';
        font-size: 13px;
        color: {ON_SURFACE};
    }}
    QScrollArea {{ border: none; background: transparent; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{
        background: transparent; width: 8px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {OUTLINE_VARIANT}; border-radius: 4px; min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {OUTLINE}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0; background: none;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    /* ---------- Header bar ---------- */
    QFrame#HeaderBar {{
        background-color: {SURFACE};
        border-bottom: 1px solid {SURFACE_VARIANT};
    }}
    QLabel#Brand {{
        font-family: '{heading}';
        font-size: 22px;
        font-weight: 700;
        color: {ON_SURFACE};
    }}
    QLabel#HeaderTitle {{
        font-family: '{mono}';
        font-size: 13px;
        font-weight: 600;
        color: {ON_SURFACE_VARIANT};
    }}
    QFrame#HeaderDivider {{
        background-color: {OUTLINE_VARIANT};
        max-width: 1px;
    }}
    QPushButton#StatusPill {{
        background-color: {SURFACE_LOWEST};
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS}px;
        padding: 6px 14px;
        font-family: '{mono}';
        font-size: 11px;
        font-weight: 600;
        color: {ON_SURFACE_VARIANT};
        text-align: left;
    }}
    QPushButton#StatusPill:hover {{ background-color: {SURFACE_LOW}; }}

    /* ---------- Cards ---------- */
    QFrame#Card {{
        background-color: {SURFACE_LOWEST};
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS}px;
    }}
    QLabel#CardTitle {{
        font-family: '{mono}';
        font-size: 12px;
        font-weight: 700;
        color: {PRIMARY};
    }}
    QFrame#CardTitleRule {{
        background-color: {SURFACE_VARIANT};
        max-height: 1px;
        border: none;
    }}
    QLabel#StepBadge {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border-radius: 11px;
        font-family: '{mono}';
        font-size: 11px;
        font-weight: 700;
    }}
    QLabel#StepText {{
        font-family: '{mono}';
        font-size: 12.5px;
        color: {ON_SURFACE};
    }}
    QLabel#FieldLabel {{
        font-family: '{heading}';
        font-size: 12px;
        font-weight: 600;
        color: {ON_SURFACE};
    }}
    QLabel#MiniLabel {{
        font-family: '{mono}';
        font-size: 10px;
        font-weight: 600;
        color: {SECONDARY};
    }}

    /* ---------- Inputs (JetBrains Mono per design) ---------- */
    QDoubleSpinBox, QLineEdit, QComboBox {{
        font-family: '{mono}';
        font-size: 13px;
        color: {ON_SURFACE};
        background-color: {SURFACE_LOWEST};
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: {RADIUS_SM}px;
        padding: 7px 10px;
        selection-background-color: {PRIMARY_CONTAINER};
        selection-color: {ON_PRIMARY};
    }}
    QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {PRIMARY};
    }}
    QDoubleSpinBox:read-only, QLineEdit:read-only {{
        background-color: {SURFACE_LOW};
        color: {ON_SURFACE_VARIANT};
    }}
    QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: 0px; border: none;
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}

    /* ---------- Buttons ---------- */
    QPushButton {{
        font-family: '{mono}';
        font-size: 12px;
        font-weight: 700;
        border-radius: {RADIUS}px;
        padding: 10px 16px;
        background-color: {SURFACE_CONTAINER};
        color: {ON_SURFACE};
        border: 1px solid {OUTLINE_VARIANT};
    }}
    QPushButton:hover {{ background-color: {SURFACE_HIGH}; }}

    QPushButton#RunButton {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border: none;
        border-radius: 24px;
        padding: 15px;
        font-size: 13px;
    }}
    QPushButton#RunButton:hover {{ background-color: {PRIMARY_CONTAINER}; }}
    QPushButton#RunButton:disabled {{
        background-color: {SURFACE_VARIANT}; color: {OUTLINE};
    }}

    QPushButton#ProceedButton {{
        background-color: transparent;
        color: {ON_SURFACE};
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: 24px;
        padding: 13px;
    }}
    QPushButton#ProceedButton:hover {{ background-color: {SURFACE_LOW}; }}
    QPushButton#ProceedButton:disabled {{
        color: {OUTLINE_VARIANT}; border-color: {SURFACE_VARIANT};
        background: transparent;
    }}

    QPushButton#SkipButton {{
        background-color: transparent;
        color: {ERROR};
        border: 1px solid {ERROR};
        border-radius: 24px;
        padding: 9px;
    }}
    QPushButton#SkipButton:hover {{ background-color: {ERROR_CONTAINER}; }}

    QPushButton#UnlockButton {{
        background-color: {SURFACE_LOW};
        color: {PRIMARY};
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: {RADIUS}px;
        padding: 5px 12px;
        font-size: 10px;
    }}
    QPushButton#UnlockButton:checked {{
        background-color: {PRIMARY}; color: {ON_PRIMARY}; border-color: {PRIMARY};
    }}

    QPushButton#GhostSmall {{
        background: {SURFACE_CONTAINER};
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: {RADIUS_SM}px;
        padding: 6px 10px;
        font-size: 11px;
    }}

    /* ---------- Directory group (DirectoryManager widget) ---------- */
    QGroupBox {{
        border: none;
        margin-top: 4px;
        font-family: '{mono}';
        font-size: 12px;
        font-weight: 700;
        color: {PRIMARY};
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 0px; }}

    /* ---------- Measurements card ---------- */
    QLabel#MetricName {{
        font-family: '{heading}';
        font-size: 20px;
        font-weight: 700;
        color: {PRIMARY};
    }}
    QLabel#BigValue {{
        font-family: '{mono}';
        font-size: 34px;
        font-weight: 700;
        color: {ON_SURFACE};
    }}
    QLabel#BigUnit {{
        font-family: '{mono}';
        font-size: 13px;
        font-weight: 700;
        color: {SECONDARY};
    }}
    QProgressBar {{
        border: none; background: {SURFACE_VARIANT};
        border-radius: 2px; max-height: 4px;
    }}
    QProgressBar::chunk {{ background-color: {PRIMARY}; border-radius: 2px; }}
    QProgressBar#SweepProgress::chunk {{ background-color: {PRIMARY}; }}

    /* ---------- Status chip (WAITING / MEASURING / PASS / FAIL) ------- */
    QLabel#StatusChip {{
        font-family: '{mono}';
        font-size: 12px;
        font-weight: 700;
        border-radius: {RADIUS_SM}px;
        padding: 8px;
        background: {SURFACE_LOW};
        color: {ON_SURFACE_VARIANT};
    }}

    /* ---------- Checklist dialog ---------- */
    QFrame#ChecklistRow {{
        background-color: {SURFACE_LOWEST};
        border: 1px solid {SURFACE_VARIANT};
        border-radius: {RADIUS_SM}px;
    }}
    QFrame#ChecklistRow:hover {{ border-color: {OUTLINE_VARIANT}; }}
    QCheckBox {{
        font-family: '{mono}';
        font-size: 12.5px;
        color: {ON_SURFACE};
        spacing: 10px;
        background: transparent;
        border: none;
    }}
    QCheckBox::indicator {{
        width: 18px; height: 18px;
        border: 1px solid {OUTLINE_VARIANT};
        border-radius: 5px;
        background: {SURFACE_LOWEST};
    }}
    QCheckBox::indicator:checked {{
        background-color: {PRIMARY};
        border-color: {PRIMARY};
    }}
    QPushButton#ConfirmBtn {{
        background-color: {PRIMARY};
        color: {ON_PRIMARY};
        border: none;
        border-radius: 22px;
        padding: 13px 24px;
        font-size: 12px;
    }}
    QPushButton#ConfirmBtn:hover {{ background-color: {PRIMARY_CONTAINER}; }}
    QPushButton#ConfirmBtn:disabled {{
        background-color: {SURFACE_VARIANT}; color: {OUTLINE};
    }}
    QLabel#DialogHeader {{
        font-family: '{heading}';
        font-size: 19px;
        font-weight: 600;
        color: {ON_SURFACE};
    }}
    QLabel#DialogSub {{
        font-family: '{body}';
        font-size: 12px;
        color: {ON_SURFACE_VARIANT};
    }}
    """
