"""
Design tokens — "Precision Instrument Interface" (DESIGN.md).

THE single source of truth for colors, spacing, and radii across the whole
application. No widget module should hardcode a hex value: import it from
here (or via the legacy names in gui/style.py, which re-export these).
"""

# ---------------------------------------------------------------------------
# Surfaces & text (Material-3 style tonal ladder from DESIGN.md)
# ---------------------------------------------------------------------------
SURFACE = "#faf9fa"              # window background
SURFACE_LOWEST = "#ffffff"       # cards / inputs
SURFACE_LOW = "#f4f3f4"
SURFACE_CONTAINER = "#eeeeee"
SURFACE_HIGH = "#e8e8e9"
SURFACE_VARIANT = "#e2e2e3"      # card borders
ON_SURFACE = "#1a1c1d"           # primary text
ON_SURFACE_VARIANT = "#41484b"   # secondary text
OUTLINE = "#71787b"
OUTLINE_VARIANT = "#c0c8cb"      # input borders

# ---------------------------------------------------------------------------
# Accents
# ---------------------------------------------------------------------------
PRIMARY = "#053a46"              # deep slate teal — primary actions
PRIMARY_CONTAINER = "#24515e"    # hover
PRIMARY_FIXED_DIM = "#a1cddd"    # selection tint
SECONDARY = "#506167"
SECONDARY_CONTAINER = "#d1e3e9"  # selected-state tint
ON_PRIMARY = "#ffffff"

ERROR = "#ba1a1a"
ERROR_BRIGHT = "#ef4444"         # functional red (abort buttons, alarms)
ERROR_CONTAINER = "#ffdad6"
ON_ERROR_CONTAINER = "#93000a"

# Device architecture. n-i-p is the default and stays primary teal; p-i-n gets
# a DESATURATED red so the two are distinguishable at a glance without the
# toggle reading as an alarm — functional red (ERROR_BRIGHT) means something is
# wrong, and an architecture choice never is.
ARCH_NIP_TEAL = PRIMARY
ARCH_PIN_RED = "#9e5a5a"
ARCH_PIN_RED_TEXT = "#8a4a4a"    # same hue, darkened for label text on white

OK_GREEN = "#10b981"             # functional status green
OK_GREEN_DARK = "#16a34a"        # checkbox / toggle fill (per CLAUDE.md)
OK_GREEN_DARKER = "#15803d"
WARN_AMBER = "#b45309"
# Log severity. WARNING reuses WARN_AMBER; ERROR is its own shade rather than
# ERROR_BRIGHT, which is tuned for a filled abort button, not for text on white.
LOG_DEBUG_GREY = "#8a9296"
LOG_ERROR = "#b91c1c"
LOG_CRITICAL = "#7f1d1d"
WARN_AMBER_BRIGHT = "#f59e0b"
WARN_AMBER_BG = "#fef3c7"

# ---------------------------------------------------------------------------
# Shape & spacing (4px baseline grid)
# ---------------------------------------------------------------------------
RADIUS_SM = 8                    # inputs, chips, small elements
RADIUS = 16                      # cards
RADIUS_PILL = 24                 # large action pills

SPACING_XS = 4
SPACING_SM = 8
SPACING_MD = 12
SPACING_LG = 16
SPACING_XL = 24

# Plot styling
PLOT_AXIS_WIDTH = 60             # fixed left-axis width so side-by-side
                                 # canvases align regardless of tick labels
PLOT_GRID_ALPHA = 0.12

# ---------------------------------------------------------------------------
# Measurement channel colours — ONE definition, used everywhere
# ---------------------------------------------------------------------------
# The plot curve, the experiment browser swatch, the Channel Analysis row and
# the channel badges in the action bar must all show the SAME colour for a
# given channel, or the operator cannot tell which trace belongs to which row.
#
# These used to be declared twice — once in AppController (for the pens) and
# once in AnalysisPanel (for the table) — with a comment claiming they were
# "kept in sync". They were not: every channel differed, and channel 1 was a
# bright blue on the plot against a near-black teal in the table. A comment
# cannot keep two lists in step, so there is now only one list.
#
# Values are the PLOT palette, because that is what the operator reads first.
CHANNEL_COLORS = {
    1: "#0984e3",   # Blue
    2: "#00b894",   # Green
    3: "#e17055",   # Orange
    4: "#a29bfe",   # Purple
    5: "#fdcb6e",   # Yellow
    6: "#e84393",   # Pink
}

# Shown for a channel outside 1-6 (should not happen; grey rather than crash).
CHANNEL_COLOR_FALLBACK = "#64748b"
