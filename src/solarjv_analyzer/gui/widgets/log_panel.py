"""Live log view for the main window's graph area.

Why not reuse pymeasure's `LogWidget`
-------------------------------------
The Advanced view already shows one, and it works there — but only there.
`LogWidget._blinking_start()` reaches for `self.parent().parent()` and calls
`tabBar()` on whatever it finds, so the widget only survives inside a
`QTabWidget`; the first WARNING would raise inside a signal handler anywhere
else. Its blink state is also held in *class* attributes, so two instances
would fight over one timer. This panel is a plain `QPlainTextEdit` with its own
handler, which can live anywhere and can be shown beside the pymeasure one.

Thread safety: log records arrive from the measurement worker threads, not the
GUI thread. The handler is a `QObject` that emits a signal; the connection to
the text view is queued automatically across threads, which is the same
mechanism pymeasure uses.
"""
import html
import logging

from PyQt5 import QtCore, QtGui, QtWidgets

from solarjv_analyzer.gui.theme import tokens

logger = logging.getLogger(__name__)

# One colour per level. INFO is left at the default text colour: it is most of
# the traffic, and colouring it makes the lines that matter harder to find.
LEVEL_COLORS = {
    "DEBUG": tokens.LOG_DEBUG_GREY,
    "WARNING": tokens.WARN_AMBER,
    "ERROR": tokens.LOG_ERROR,
    "CRITICAL": tokens.LOG_CRITICAL,
}

# Lines kept. Long SPO runs log steadily; an unbounded document grows without
# limit and eventually makes the GUI thread crawl on repaint. Several panels
# exist (one per view), so this is per panel as well as for the shared history.
MAX_LINES = 2000


# A fixed-width tag per level, so the message column lines up whatever the
# severity. INFO is blank rather than labelled: it is most of the traffic, and
# a column of "INFO" makes the two lines that matter harder to find.
LEVEL_TAGS = {
    "DEBUG": "dbg ", "INFO": "    ", "WARNING": "WARN",
    "ERROR": "ERR ", "CRITICAL": "CRIT",
}


class _TaggedFormatter(logging.Formatter):
    def format(self, record):
        record.tag = LEVEL_TAGS.get(record.levelname, record.levelname[:4])
        return super().format(record)


class _SignalEmitter(QtCore.QObject):
    record = QtCore.pyqtSignal(str, str)          # (levelname, formatted text)


class QtLogHandler(logging.Handler):
    """A logging handler that hands records to a widget on the GUI thread."""

    def __init__(self):
        super().__init__()
        self.emitter = _SignalEmitter()
        self.setFormatter(
            _TaggedFormatter("%(asctime)s  %(tag)s  %(message)s",
                             datefmt="%H:%M:%S")
        )

    def emit(self, record):
        try:
            self.emitter.record.emit(record.levelname, self.format(record))
        except RuntimeError:
            # The receiving widget's C++ object is gone (window closed while a
            # worker was still logging). Nothing to do; never raise out of a
            # log call.
            pass
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# One handler, one history, any number of views.
#
# Each view (JV+SPO, Advanced/JV, Advanced/SPO) shows its own LogPanel, but a
# handler PER PANEL would mean N formatting passes per record and N handlers
# left on the process-wide root logger every time a window is rebuilt — the
# relogin loop in main.py builds a new window on every login. So the handler is
# a module singleton: panels connect to its signal and disconnect when they go.
_root_handler = None
_history = []


def install_root_handler(level=logging.DEBUG):
    """Attach the shared handler to the root logger. Idempotent.

    The emitter is a PARENTLESS QObject kept alive only by this module global,
    which SIP will garbage-collect if the QApplication it was created under
    goes away — the same failure pymeasure's class-level `LogWidget
    ._blink_qtimer` has, and which `tests/conftest.py` works around by
    recreating it. Touch the C++ object first and rebuild if it is gone, so a
    dead emitter can never take a window's construction down with it.
    """
    global _root_handler
    if _root_handler is not None:
        try:
            _root_handler.emitter.objectName()
        except RuntimeError:
            logging.root.removeHandler(_root_handler)
            _root_handler = None
    if _root_handler is None:
        _root_handler = QtLogHandler()
        _root_handler.emitter.record.connect(_remember)
    _root_handler.setLevel(level)
    if _root_handler not in logging.root.handlers:
        logging.root.addHandler(_root_handler)
    return _root_handler


def _remember(levelname, text):
    _history.append((levelname, text))
    if len(_history) > MAX_LINES:
        del _history[:-MAX_LINES]


def history():
    """Everything logged so far, so a panel built later is not blank."""
    return list(_history)


def reset_for_tests():
    global _root_handler
    if _root_handler is not None:
        logging.root.removeHandler(_root_handler)
        _root_handler = None
    _history.clear()


class LogPanel(QtWidgets.QWidget):
    """Read-only live log with a severity filter.

    `attention` goes True when a WARNING or worse arrives while the panel is
    not the visible page, so the tab showing it can say so. `mark_seen()`
    clears it.
    """

    attention_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min_level = logging.DEBUG
        self._attention = False
        self._records = []                        # (levelno, levelname, text)
        self.handler = None

        self._build_ui()

    # -- construction ------------------------------------------------------
    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 8)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        self.level_filter = QtWidgets.QComboBox()
        self.level_filter.setObjectName("LogLevelFilter")
        # "All messages" starts at DEBUG, not INFO. The Analysis tab's
        # "Enable detailed validation (debug only)" switch raises the ROOT
        # logger to DEBUG — but this filter used to sit at INFO, so the extra
        # trace the operator had just asked for was recorded and then hidden.
        # Nothing arrives below INFO unless that switch is on, so this costs
        # nothing the rest of the time.
        for label, level in (("All messages", logging.DEBUG),
                             ("Warnings & errors", logging.WARNING),
                             ("Errors only", logging.ERROR)):
            self.level_filter.addItem(label, level)
        self.level_filter.currentIndexChanged.connect(self._on_filter_changed)
        self.level_filter.setCursor(QtCore.Qt.PointingHandCursor)

        self.autoscroll = QtWidgets.QCheckBox("Follow")
        self.autoscroll.setObjectName("LogFollow")
        self.autoscroll.setChecked(True)
        self.autoscroll.setToolTip("Keep the newest line in view")

        self.copy_button = QtWidgets.QPushButton("Copy")
        self.clear_button = QtWidgets.QPushButton("Clear")
        for button in (self.copy_button, self.clear_button):
            button.setObjectName("LogAction")
            button.setCursor(QtCore.Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.clear_button.clicked.connect(self.clear)

        controls.addWidget(QtWidgets.QLabel("Show:"))
        controls.addWidget(self.level_filter)
        controls.addWidget(self.autoscroll)
        controls.addStretch(1)
        controls.addWidget(self.copy_button)
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        self.view = QtWidgets.QPlainTextEdit()
        self.view.setObjectName("LogView")
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(MAX_LINES)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.view.setFont(QtGui.QFont("JetBrains Mono, Menlo, Consolas", 11))
        layout.addWidget(self.view, stretch=1)

    # -- logging -----------------------------------------------------------
    def attach_to_root(self, level=logging.DEBUG):
        """Start receiving records, and show what was logged before this panel
        existed. The root logger's own level still gates what reaches the
        handler at all — `toggle_debug_logging()` moves that."""
        self.handler = install_root_handler(level)
        self.handler.emitter.record.connect(self._on_record)
        for levelname, text in history():
            self._on_record(levelname, text, remember=False)

    def detach(self):
        """Stop this panel receiving records. The shared handler stays on the
        root logger — it belongs to the process, not to any one widget — but
        nothing is delivered to a widget that is being destroyed."""
        if self.handler is None:
            return
        try:
            self.handler.emitter.record.disconnect(self._on_record)
        except TypeError:
            pass                                  # already disconnected
        self.handler = None

    def _on_record(self, levelname, text, remember=True):
        levelno = logging.getLevelName(levelname)
        if not isinstance(levelno, int):
            levelno = logging.INFO
        self._records.append((levelno, levelname, text))
        if len(self._records) > MAX_LINES:
            del self._records[:-MAX_LINES]

        try:
            if levelno >= self._min_level:
                self._append(levelname, text)
            if remember and levelno >= logging.WARNING and not self.isVisible():
                self._set_attention(True)
        except RuntimeError:
            # Our C++ side has been destroyed while the shared handler was
            # still delivering to us. PyQt only auto-disconnects when the
            # PYTHON wrapper is collected, which can be much later — and a
            # worker thread logging during window teardown gets here first.
            # Unsubscribe rather than raise inside a Qt slot.
            self.detach()

    def _append(self, levelname, text):
        color = LEVEL_COLORS.get(levelname)
        escaped = html.escape(text).replace(" ", "&nbsp;")
        if color:
            self.view.appendHtml(
                f'<span style="color:{color};">{escaped}</span>')
        else:
            self.view.appendHtml(escaped)

        if self.autoscroll.isChecked():
            bar = self.view.verticalScrollBar()
            bar.setValue(bar.maximum())

    # -- filter / actions --------------------------------------------------
    def _on_filter_changed(self, index):
        self._min_level = self.level_filter.itemData(index) or logging.INFO
        self._rebuild()

    def _rebuild(self):
        """Re-render the kept records at the current filter level."""
        self.view.clear()
        for levelno, levelname, text in self._records:
            if levelno >= self._min_level:
                self._append(levelname, text)

    def clear(self):
        self._records.clear()
        self.view.clear()

    def copy_to_clipboard(self):
        QtWidgets.QApplication.clipboard().setText(self.view.toPlainText())

    # -- attention ---------------------------------------------------------
    def _set_attention(self, value):
        if self._attention != value:
            self._attention = value
            self.attention_changed.emit(value)

    def has_attention(self):
        return self._attention

    def mark_seen(self):
        self._set_attention(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.mark_seen()
