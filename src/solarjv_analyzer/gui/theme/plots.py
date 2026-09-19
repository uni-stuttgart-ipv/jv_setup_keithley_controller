"""
pyqtgraph theming — one place for every plot's look.

Key trick for side-by-side plots (JV+SPO combined view): pyqtgraph sizes
the left axis to its tick-label width, so "Current (A)" and "Power (mW)"
plots get DIFFERENT canvas rectangles. style_plot() pins the axis width so
any two styled plots align to the pixel.
"""

import pyqtgraph as pg
from PyQt5 import QtCore

from .fonts import mono_family
from .tokens import (
    OUTLINE_VARIANT, SECONDARY, ON_SURFACE_VARIANT,
    PLOT_AXIS_WIDTH, PLOT_GRID_ALPHA, ERROR,
)


def apply_global_plot_config():
    """Call once before any plot is created."""
    pg.setConfigOption('background', '#ffffff')
    pg.setConfigOption('foreground', ON_SURFACE_VARIANT)


def style_plot(plot_item, lock_axis_width: bool = True):
    """Apply the design-system look to a pyqtgraph PlotItem."""
    plot_item.showGrid(x=True, y=True, alpha=PLOT_GRID_ALPHA)
    for side in ('left', 'bottom'):
        axis = plot_item.getAxis(side)
        axis.setPen(pg.mkPen(color=OUTLINE_VARIANT, width=1))
        axis.setTextPen(pg.mkPen(color=SECONDARY))
        # Pin the axis-label CSS so two side-by-side plots render their
        # bottom axes at identical height. A pymeasure PlotWidget labels its
        # axis via CSS ("font-size: 10pt; font-family: Arial") while a bare
        # pyqtgraph PlotWidget leaves labelStyle empty (default font) — that
        # mismatch makes the two graph areas uneven. Re-applying the label
        # with one consistent style collapses the difference.
        axis.setLabel(
            axis.labelText,
            units=axis.labelUnits,
            **{"font-size": "10pt", "font-family": mono_family(), "color": ON_SURFACE_VARIANT},
        )
    if lock_axis_width:
        plot_item.getAxis('left').setWidth(PLOT_AXIS_WIDTH)
    return plot_item


def dashed_marker_line(angle: float = 0):
    """A dashed error-red marker line (e.g. target-Isc) — design language."""
    line = pg.InfiniteLine(
        angle=angle, movable=False,
        pen=pg.mkPen(ERROR, width=1, style=QtCore.Qt.DashLine),
    )
    line.setOpacity(0.5)
    return line


def forget_dead_viewboxes() -> int:
    """Drop pyqtgraph ViewBoxes whose C++ object has been destroyed.

    `ViewBox.__init__` ends with `updateAllViewLists()`, which walks
    `ViewBox.AllViews` and calls a method on **every** entry. That dictionary
    is weak on the Python wrapper, not on the C++ object, so every ViewBox
    from a closed window stays in it for as long as anything still holds the
    wrapper — a traceback frame, a test fixture, a stale reference. The walk
    then dereferences freed memory, which usually happens to look valid and
    occasionally does not.

    The JV+SPO view alone registers four ViewBoxes, and `main.py` builds a new
    window on every trip round the relogin loop. Measured before this existed:
    three open/close cycles left 12 of 12 entries pointing at destroyed
    objects, and the list only grew. With it, the registry stays at four.

    Call before building plots and after tearing them down. Returns how many
    entries were dropped.
    """
    try:
        from pyqtgraph.graphicsItems.ViewBox import ViewBox
    except Exception:                                    # pragma: no cover
        return 0
    try:
        from PyQt5 import sip
    except ImportError:                                  # pragma: no cover
        try:
            import sip
        except Exception:
            return 0

    def _is_dead(view):
        try:
            return sip.isdeleted(view)
        except (TypeError, RuntimeError):
            return True                                  # unusable either way

    removed = 0
    for view in list(getattr(ViewBox, "AllViews", {}).keys()):
        if _is_dead(view):
            ViewBox.AllViews.pop(view, None)
            removed += 1
    for name, view in list(getattr(ViewBox, "NamedViews", {}).items()):
        if _is_dead(view):
            ViewBox.NamedViews.pop(name, None)
    return removed
