"""Shared visual style — sober academic-journal register.

White background, neutral sans typography, thin black axes, no chartjunk, a
restrained palette that stays legible in grayscale (each series also carries a
distinct line style), and a standard framed legend. Built to read as a figure in
Econometrica / Journal of Econometrics / AEJ, not as an editorial infographic.
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# --- core ink ------------------------------------------------------------
INK   = "#1a1a1a"   # text, axes, primary series
GRAY  = "#666666"   # secondary / zero line / ticks
MUTED = GRAY        # alias
LINE  = "#cccccc"   # legend edge / hairlines
PAPER = "white"

# --- restrained palette (distinct in grayscale; pair with a line style) ---
BRICK = "#9e2b25"
NAVY  = "#27496d"
GREEN = "#3f6f52"
OCHRE = "#9a7d29"
PALETTE = [INK, BRICK, NAVY, GREEN, OCHRE]
STYLES  = ["-", (0, (5, 2)), (0, (1, 1.6)), (0, (6, 2, 1, 2)), (0, (3, 1.5))]

# back-compat aliases (older scripts referenced these names)
RUST, SLATE, SAGE, SIENNA, AMBER = BRICK, NAVY, GREEN, OCHRE, "#cfcfcf"

# sequential map for the size figures — restrained, light grey -> brick
SIZE_CMAP = LinearSegmentedColormap.from_list(
    "graybrick", ["#efefef", "#cfcfcf", "#9e9e9e", "#9e2b25", "#5a1714"])

MONO = {"family": "Menlo", "fontsize": 9}


def apply():
    mpl.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "savefig.bbox":      "tight",
        "font.family":       ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "mathtext.fontset":  "cm",
        "text.color":        INK,
        "axes.edgecolor":    "#333333",
        "axes.linewidth":    0.8,
        "axes.labelcolor":   INK,
        "axes.labelsize":    11,
        "axes.titlesize":    11.5,
        "axes.titleweight":  "normal",
        "axes.titlecolor":   INK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "xtick.color":       INK,
        "ytick.color":       INK,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.frameon":    True,
        "legend.framealpha": 1.0,
        "legend.edgecolor":  LINE,
        "legend.fancybox":   False,
        "legend.borderpad":  0.6,
        "legend.fontsize":   10,
        "figure.dpi":        150,
    })
