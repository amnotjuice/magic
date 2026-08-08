"""House style for the paper's plotted figures (Fig. 2, 3, 4 and 7).

Follows the reference style: serif type, a black frame on all four sides, a
dashed grey grid behind the marks, no tick marks, frameless legends, and the
NPG accent palette.

The reference draws on a 12in canvas at 38pt labels and lets LaTeX scale the
result down. We keep this paper's rule of drawing at the final print size
instead -- ACL \\columnwidth is 3.03in and \\textwidth 6.3in -- so the sizes
below are the reference's proportions evaluated there: 38/34/32pt over 12in is
9.6/8.6/8.1pt over 3.03in, and its 3pt axis rule is 0.76pt. Drawing at final
size keeps every figure's text at one weight regardless of how wide the figure
is, which scaling down does not.

Usage:
    import figstyle as fs
    fs.use()
    ...
    fs.frame(ax)            # 4-sided frame, dashed y grid, no tick marks
    fs.frame(ax, grid="both")
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# NPG (ggsci) palette. The reference passes these with a CC alpha suffix; we
# drop the alpha and set it per artist instead, so vector output stays opaque
# where it should be.
RED = "#E64B35"
SKY = "#4DBBD5"
TEAL = "#00A087"
NAVY = "#3C5488"
SALMON = "#F39B7F"
SLATE = "#8491B4"
MINT = "#91D1C2"
BROWN = "#7E6148"

# Semantic roles, so a palette change stays in this file.
OURS = NAVY          # every curve or point that is MAGIC
CONTRAST = RED       # the population or backbone it is read against
THIRD = TEAL
FOURTH = SKY
DKGRAY, MDGRAY, LTGRAY = "#3A3A3A", "#6E6E6E", "#9A9A9A"   # baselines
RULE = "#7A7A7A"     # reference lines (zero, threshold, fitted floor)

CYCLE = [NAVY, RED, TEAL, SKY, SALMON, SLATE, MINT, BROWN]

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    # black frame on all four sides, as in the reference
    "axes.linewidth": 0.9,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    # dashed grey grid behind the marks
    "grid.linestyle": "--",
    "grid.color": "gray",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.6,
    # the reference hides the tick marks and keeps only the labels
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
}


def use():
    """Install the house style. Call once, before creating any figure."""
    plt.rcParams.update(RC)


def frame(ax, grid="y", nbins=None):
    """Frame one axes the way the reference does.

    grid: "y", "x", "both" or None -- which dashed gridlines to draw.
    nbins: (x, y) maximum tick counts, when the default locator is too busy.
    """
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(RC["axes.linewidth"])
        sp.set_edgecolor("black")
    if grid:
        ax.grid(axis="both" if grid == "both" else grid)
    else:
        ax.grid(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    if nbins:
        from matplotlib.ticker import MaxNLocator
        bx, by = nbins
        if bx:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=bx, integer=True))
        if by:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=by))
