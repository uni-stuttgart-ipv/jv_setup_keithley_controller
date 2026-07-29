"""
Toggle Switch Widget

A modern, animated iOS-style toggle switch used for the Channel Selection
UI (and "Select All"). Behaves like a QCheckBox/QRadioButton (isChecked(),
setChecked(), toggled signal) so it's a drop-in replacement wherever the
app previously used checkboxes, but renders as a sliding pill switch with
a green "on" accent matching the app's existing success color.
"""

from PyQt5 import QtCore, QtGui, QtWidgets


class ToggleSwitch(QtWidgets.QAbstractButton):
    """An animated on/off slider switch."""

    def __init__(
        self,
        parent=None,
        width: int = 44,
        height: int = 24,
        on_color: str = "#2ecc71",
        off_color: str = "#cbd5e1",
        handle_color: str = "#ffffff",
    ):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)

        self._track_width = width
        self._track_height = height
        self._on_color = QtGui.QColor(on_color)
        self._off_color = QtGui.QColor(off_color)
        self._handle_color = QtGui.QColor(handle_color)
        self._handle_position = 1.0 if self.isChecked() else 0.0

        self._animation = QtCore.QPropertyAnimation(self, b"handle_position", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)

        self.toggled.connect(self._animate_to_state)
        self.setFixedSize(width, height)

    # -------------------------------------------------------------------
    # Animated "handle_position" property (0.0 = off, 1.0 = on)
    # -------------------------------------------------------------------

    def _get_handle_position(self) -> float:
        return self._handle_position

    def _set_handle_position(self, value: float):
        self._handle_position = value
        self.update()

    handle_position = QtCore.pyqtProperty(float, _get_handle_position, _set_handle_position)

    def _animate_to_state(self, checked: bool):
        self._apply_visual_state(checked)

    def _apply_visual_state(self, checked: bool):
        target = 1.0 if checked else 0.0
        if not self.isVisible():
            # Not yet shown (e.g. initial state set during __init__ before
            # the widget is displayed): jump directly so it doesn't render
            # a stale "off" position on first paint.
            self._handle_position = target
            self.update()
            return
        self._animation.stop()
        self._animation.setStartValue(self._handle_position)
        self._animation.setEndValue(target)
        self._animation.start()

    def sync_visual_state(self):
        """Force the handle to visually match isChecked(). Call this after
        changing the checked state while signals were blocked (e.g. via
        blockSignals()), since in that case the toggled-driven animation
        never fires and the switch would otherwise look stuck."""
        self._apply_visual_state(self.isChecked())

    # -------------------------------------------------------------------
    # Painting
    # -------------------------------------------------------------------

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._track_width, self._track_height)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)

        track_rect = QtCore.QRectF(0, 0, self.width(), self.height())
        radius = self.height() / 2.0

        track_color = self._blend(self._off_color, self._on_color, self._handle_position)
        if not self.isEnabled():
            track_color = track_color.lighter(130)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track_rect, radius, radius)

        margin = 2.0
        handle_diameter = self.height() - margin * 2
        max_travel = self.width() - handle_diameter - margin * 2
        handle_x = margin + self._handle_position * max_travel
        handle_rect = QtCore.QRectF(handle_x, margin, handle_diameter, handle_diameter)
        painter.setBrush(self._handle_color)
        painter.drawEllipse(handle_rect)

    @staticmethod
    def _blend(color_a: QtGui.QColor, color_b: QtGui.QColor, t: float) -> QtGui.QColor:
        t = max(0.0, min(1.0, t))
        r = color_a.red() + (color_b.red() - color_a.red()) * t
        g = color_a.green() + (color_b.green() - color_a.green()) * t
        b = color_a.blue() + (color_b.blue() - color_a.blue()) * t
        return QtGui.QColor(int(r), int(g), int(b))
