"""
Shared chart styling for the logistics workstream.

One place for the palette and the matplotlib defaults, so every figure in the
report reads as one system. Colours are the validated default palette:
the 3-slot categorical set clears CVD separation on all pairs
(worst dE 9.2, target >= 8) and the normal-vision floor (worst dE 24.0, floor 15).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- palette -------------------------------------------------------------
SURFACE = "#fcfcfb"      # chart surface
INK = "#0b0b0b"          # primary text
INK_2 = "#52514e"        # secondary text
MUTED = "#898781"        # axis labels / annotations
GRID = "#e1e0d9"         # hairline gridlines
AXIS = "#c3c2b7"         # baseline / axis rule

# Categorical slots, used in fixed order and never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA = SERIES[0], SERIES[1], SERIES[2]

# Sequential ramp (one hue, light -> dark) for continuous magnitude.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

# Diverging pair (warm/cool poles + neutral gray midpoint) for polarity.
DIV_COOL = "#2a78d6"
DIV_WARM = "#d03b3b"
DIV_MID = "#f0efec"

# Status colours - reserved for good/bad meaning, never reused as a series.
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"


def apply_style():
    """Set the global matplotlib defaults: thin marks, recessive chrome."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # never dashed - dashes read as "threshold"
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def clean_axes(ax, grid_axis="y"):
    """
    Strip the box down to one recessive baseline plus hairline gridlines.
    grid_axis: "x", "y", or "both" - the axes that KEEP a gridline.
    """
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(False)
    if grid_axis == "both":
        ax.grid(axis="both", color=GRID, linewidth=0.8)
    else:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.tick_params(length=0)
    return ax


def title_block(ax, title, subtitle=None):
    """
    Headline states the finding; the subtitle carries the caveat/method.
    A chart whose title is just a variable name makes the reader do the work.
    """
    n_lines = title.count("\n") + 1
    ax.set_title(title, pad=20 + 14 * n_lines if subtitle else 14)
    if subtitle:
        ax.text(0, 1.028, subtitle, transform=ax.transAxes,
                fontsize=9.5, color=MUTED, va="bottom", ha="left")


def source_note(fig, text):
    fig.text(0.005, -0.02, text, fontsize=8.5, color=MUTED, ha="left", va="top")
