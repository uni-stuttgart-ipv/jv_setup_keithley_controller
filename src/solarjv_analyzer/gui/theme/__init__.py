"""
gui.theme — the design system ("Precision Instrument Interface").

Public API:
    from solarjv_analyzer.gui.theme import (
        tokens,                      # color/spacing/radius constants
        load_design_fonts,           # call once after QApplication exists
        label_font, data_font,       # QFont factories (letter-spacing etc.)
        base_stylesheet,             # app-wide QSS
        dialog_stylesheet,           # QApplication-level popup QSS
        style_plot, dashed_marker_line, apply_global_plot_config,  # pyqtgraph
    )
"""

from . import tokens  # noqa: F401
from .fonts import (  # noqa: F401
    load_design_fonts, heading_family, body_family, mono_family,
    label_font, data_font,
)
from .stylesheet import (  # noqa: F401
    base_stylesheet, dialog_stylesheet, checkbox_stylesheet,
    scrollbar_stylesheet,
)
from .plots import (  # noqa: F401
    apply_global_plot_config, style_plot, dashed_marker_line,
    forget_dead_viewboxes,
)
