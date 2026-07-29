"""
Channel Pinout Image Helper

Resolves the path to the bundled channel_pinout.png reference image
(shipped inside the package under solarjv_analyzer/resources/ so it
travels correctly with packaged builds) and provides a convenience
function to build a ready-to-use QLabel showing it, scaled to fit the
Channel Selection card.
"""

import logging
import os

from PyQt5 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")
PINOUT_IMAGE_PATH = os.path.join(_RESOURCES_DIR, "channel_pinout.png")


def build_pinout_label(max_height: int = 130) -> QtWidgets.QLabel:
    """Return a QLabel with the reference pinout image scaled to fit,
    or a plain text placeholder if the image asset is missing."""
    label = QtWidgets.QLabel()
    label.setAlignment(QtCore.Qt.AlignCenter)

    pixmap = QtGui.QPixmap(PINOUT_IMAGE_PATH)
    if pixmap.isNull():
        logger.warning(f"Channel pinout image not found at {PINOUT_IMAGE_PATH}")
        label.setText("Reference Pinout\n(image not found)")
        label.setStyleSheet("color: gray; font-size: 8pt;")
        return label

    scaled = pixmap.scaledToHeight(max_height, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scaled)
    return label
