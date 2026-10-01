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

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

_RESOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "resources")
PINOUT_IMAGE_PATH = os.path.join(_RESOURCES_DIR, "channel_pinout.png")

# Cache the margin-cropped source image so the (one-time) numpy scan
# doesn't rerun on every tab construction.
_cropped_cache = {}


def _crop_white_margins(image: QtGui.QImage) -> QtGui.QImage:
    """Crop the uniform white/transparent border from the pinout image.

    The bundled ``channel_pinout.png`` has ~660px of blank space on the
    left (and ~190px on the right) with the diagram touching the top and
    bottom edges. Left uncropped, the scaled pixmap is far wider than the
    channel-selector column, so the layout squeezes the label and the
    centered pixmap is clipped at the top. Cropping the margins first makes
    the scaled image fit the column without clipping.
    """
    image = image.convertToFormat(QtGui.QImage.Format_ARGB32)
    h, w = image.height(), image.width()

    ptr = image.constBits()
    ptr.setsize(image.byteCount())
    bpl = image.bytesPerLine()
    arr = np.frombuffer(ptr, np.uint8).reshape(h, bpl)[:, : w * 4]
    arr = arr.reshape(h, w, 4).astype(np.int16)

    b, g, r, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    background = (a == 0) | ((r >= 250) & (g >= 250) & (b >= 250))
    content = ~background

    rows = np.any(content, axis=1)
    cols = np.any(content, axis=0)
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    if ys.size == 0 or xs.size == 0:
        return image

    pad = 4
    top = max(0, int(ys[0]) - pad)
    left = max(0, int(xs[0]) - pad)
    bottom = min(h - 1, int(ys[-1]) + pad)
    right = min(w - 1, int(xs[-1]) + pad)

    return image.copy(left, top, right - left + 1, bottom - top + 1)


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

    if PINOUT_IMAGE_PATH not in _cropped_cache:
        _cropped_cache[PINOUT_IMAGE_PATH] = QtGui.QPixmap.fromImage(
            _crop_white_margins(pixmap.toImage())
        )
    pixmap = _cropped_cache[PINOUT_IMAGE_PATH]

    scaled = pixmap.scaledToHeight(max_height, QtCore.Qt.SmoothTransformation)
    label.setPixmap(scaled)
    # Keep the label from being squeezed below the pixmap size: a QLabel
    # with a pixmap reports heightForWidth, so a compressed width shrinks
    # its layout height while the pixmap stays full-size and gets clipped.
    label.setMinimumSize(scaled.size())
    return label
