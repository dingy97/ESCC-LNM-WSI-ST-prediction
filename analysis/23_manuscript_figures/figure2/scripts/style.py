"""Shared journal typography, physical layout and vector export for Stage 23."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np

OUT = Path(__file__).resolve().parents[2]
MM = 1 / 25.4
INK, MUTED, GRID = "#182632", "#4C5E6B", "#D9E3E8"
BLUE, ORANGE, TEAL, LIGHT = "#0072B2", "#C44F23", "#237E83", "#E8F2F4"
COLORS = {
    "Logistic Regression": BLUE, "SVM": "#C4511C", "Random Forest": "#008A68", "MLP": "#A65083",
    "LightGBM": "#866321", "KNN": "#547984", "GBDT": "#7550A0", "Gaussian Naive Bayes": "#59656C",
    "Extra Trees": "#192630", "CatBoost": "#B84264", "AdaBoost": "#A56D00", "XGBoost": "#41438E",
}
ABBR = dict(zip(COLORS, ["LR", "SVM", "RF", "MLP", "LGBM", "KNN", "GBDT", "GNB", "ET", "CB", "Ada", "XGB"]))
ORDER = list(COLORS)
MARKERS = dict(zip(COLORS, ["o", "s", "^", "D", "v", "P", ">", "X", "p", "<", "h", "*"]))
LNM = {0: TEAL, 1: ORANGE}
LNM_LABELS = {0: "LNM−", 1: "LNM+"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": MUTED, "axes.linewidth": .55,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "axes.labelpad": 2.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "xtick.major.size": 2.1, "ytick.major.size": 2.1, "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "legend.frameon": False, "legend.fontsize": 7, "legend.handlelength": 1.6,
    "legend.handletextpad": .45, "legend.labelspacing": .3, "legend.borderaxespad": .3,
    "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": .45,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.transparent": False,
})


def data(code):
    return pd.read_csv(OUT / f"figure{code[0]}/source_data/Figure{code}.csv", float_precision="round_trip")


class Frame:
    """Place artists using millimetres within a panel, from the bottom left."""
    def __init__(self, fig, bounds):
        self.fig = fig
        self.x, self.y, self.w, self.h = bounds
        self.fw, self.fh = np.asarray(fig.get_size_inches()) / MM

    def axes(self, left=10, bottom=10, right=2, top=12):
        return self.fig.add_axes([(self.x+left)/self.fw, (self.y+bottom)/self.fh,
            (self.w-left-right)/self.fw, (self.h-bottom-top)/self.fh])

    def text(self, x, y, string, **kwargs):
        return self.fig.text((self.x+x)/self.fw, (self.y+y)/self.fh, string, **kwargs)

    def header(self, letter, title, note=None, note_color=MUTED):
        self.text(.4, self.h-1.2, letter, fontsize=11, fontweight="bold", va="top")
        self.text(6.1, self.h-1.8, title, fontsize=8.5, fontweight="bold", va="top")
        if note:
            self.text(6.1, self.h-6.1, note, fontsize=7, va="top", color=note_color)


def unit_axis(ax, axis="both", *, percentages=False, step=.25):
    values = np.arange(0, 1 + step/2, step)
    form = mtick.PercentFormatter(1, decimals=0) if percentages else mtick.FormatStrFormatter("%.2g")
    if axis in ("x", "both"):
        ax.set(xlim=(0, 1), xticks=values)
        ax.xaxis.set_major_formatter(form)
    if axis in ("y", "both"):
        ax.set(ylim=(0, 1), yticks=values)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0) if percentages else mtick.FormatStrFormatter("%.2g"))


def roc_axes(ax):
    unit_axis(ax)
    ax.plot([0, 1], [0, 1], ls="--", color="#7C8A94", lw=.7, zorder=0)
    ax.set(xlabel="False-positive rate", ylabel="Sensitivity")
    ax.grid(True)


def ranked_bars(ax, d):
    cutoff = d.threshold.iloc[0]
    ax.bar(d["rank"], d.probability, width=.91, color=d.LNM.map(LNM), linewidth=0)
    ax.axhline(cutoff, color=INK, lw=.85, ls="--")
    ax.set(xlim=(0, len(d)+1), xticks=[1, round(len(d)/2), len(d)],
           xlabel="Case rank by score", ylabel="Predicted probability")
    unit_axis(ax, "y")
    ax.grid(axis="y")


def export(fig, folder, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=600)
    plt.close(fig)


def build(number, functions, layout, standalone, height, heading=None, *, only=None):
    base = OUT / f"figure{number}"
    for letter, draw in functions.items():
        if only and letter not in only:
            continue
        w, h = standalone[letter]
        fig = plt.figure(figsize=(w*MM, h*MM))
        draw(Frame(fig, (2, 1, w-4, h-2)), True)
        export(fig, base / "panels", f"Figure{number}{letter}")
    fig = plt.figure(figsize=(180*MM, height*MM))
    if heading:
        heading(fig, height)
    for letter, draw in functions.items():
        draw(Frame(fig, layout[letter]), False)
    export(fig, base / "combined", f"Figure{number}_combined")
