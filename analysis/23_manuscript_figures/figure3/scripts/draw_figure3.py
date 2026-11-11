"""Render the matched historical WSI comparison as a supplementary figure."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figure2/scripts"))
from style import BLUE, INK, MUTED, ORANGE, TEAL, build, data, np, roc_axes, unit_axis

SERIES = {"A": ("Visual", TEAL, "o"), "B": ("Inferred RNA", BLUE, "s"),
          "D": ("Visual + RNA + programs", ORANGE, "^")}
LEGACY = "Historical · legacy LNM labels (102/58)"
SHORT = {"A": "Visual", "B": "Inferred RNA", "D": "Fusion"}


def footer_key(frame, ax, *, title=None):
    ax.legend(loc="lower center", bbox_to_anchor=((frame.x+frame.w/2)/frame.fw, (frame.y+.7)/frame.fh),
              bbox_transform=frame.fig.transFigure, title=title, title_fontsize=6.7,
              fontsize=6.6, ncol=2, borderaxespad=0, columnspacing=.8)


def rocs(frame, standalone):
    d = data("3A")
    frame.header("A", "ROC by representation", LEGACY)
    ax = frame.axes(left=11, bottom=19, right=2, top=12)
    roc_axes(ax)
    for (series, (label, color, _)), ls in zip(SERIES.items(), [":", "-", "--"]):
        g = d[d.series.eq(series)]
        ax.step(g.FPR, g.TPR, where="post", color=color, lw=1.2, ls=ls,
                label=f"{label if standalone else SHORT[series]}  {g.AUROC.iloc[0]:.3f}")
    footer_key(frame, ax)


def auc(frame, standalone):
    d = data("3B").set_index("series")
    frame.header("B", "AUROC and uncertainty", LEGACY)
    ax = frame.axes(left=20, bottom=19, right=3, top=12)
    for i, (series, (label, color, marker)) in enumerate(SERIES.items()):
        row = d.loc[series]
        ax.errorbar(row.AUROC, i, xerr=[[row.AUROC-row.CI_low], [row.CI_high-row.AUROC]],
                    fmt=marker, color=color, lw=1.1, capsize=2, ms=4.5)
        ax.text(row.AUROC, i-.24, f"{row.AUROC:.3f}", color=color, ha="center", va="bottom", fontsize=7)
    ax.set(yticks=range(3), yticklabels=["Visual", "Inferred RNA", "Visual + RNA\n+ programs"],
           ylim=(2.55,-.7), xlabel="AUROC (95% CI)")
    unit_axis(ax, "x", step=.5)
    ax.axvline(.5, color=MUTED, ls="--", lw=.65)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)


def calibration(frame, standalone):
    d = data("3C")
    frame.header("C", "Calibration", LEGACY)
    ax = frame.axes(left=11, bottom=22, right=2, top=12)
    unit_axis(ax)
    ax.plot([0,1], [0,1], color=MUTED, ls="--", lw=.65, zorder=0)
    for series, (label, color, marker) in SERIES.items():
        g = d[d.series.eq(series)]
        ax.plot(g.mean_score, g.event_fraction, color=color, marker=marker, ms=3.4, lw=1.,
                label=f"{label if standalone else SHORT[series]}  {g.Brier.iloc[0]:.3f}")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed LNM fraction")
    ax.grid(True)
    footer_key(frame, ax, title="Brier score")


def performance(frame, standalone):
    d = data("3D")
    frame.header("D", "Pooled OOF performance", LEGACY)
    ax = frame.axes(left=10, bottom=15, right=2, top=13)
    metrics = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "Brier"]
    x = np.arange(len(metrics))
    for i, (series, (label, color, _)) in enumerate(SERIES.items()):
        table = d[d.series.eq(series)].set_index("metric")
        ax.bar(x+(i-1)*.23, table.loc[metrics].value, width=.215, color=color, label=label, lw=0)
    unit_axis(ax, "y")
    ax.set(xticks=x, ylabel="Metric value")
    ax.set_xticklabels([*metrics[:-1], "Brier ↓"], rotation=40, ha="right", rotation_mode="anchor")
    ax.grid(axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=((frame.x+frame.w/2)/frame.fw, (frame.y+1.0)/frame.fh),
              bbox_transform=frame.fig.transFigure, ncol=3, fontsize=6.8, borderaxespad=0)


def decision(frame, standalone):
    d = data("3E")
    frame.header("E", "Decision curves", LEGACY)
    ax = frame.axes(left=12, bottom=11, right=2, top=12)
    colors = {k:(v[0],v[1]) for k,v in SERIES.items()}
    colors.update({"Treat all":("Treat all", "#7A8791"), "Treat none":("Treat none", INK)})
    for (series, (label, color)), ls in zip(colors.items(), [":", "-", "--", "-", "--"]):
        g = d[d.series.eq(series)]
        ax.plot(g.threshold, g.net_benefit, color=color, ls=ls, lw=1.05, label=label)
    ax.set(xlim=(.04,.81), xticks=[.05,.2,.4,.6,.8], xlabel="Threshold probability", ylabel="Net benefit")
    ax.grid(True)
    ax.legend(loc="lower left", ncol=2, fontsize=6.5)


def heading(fig, height):
    fig.text(.012, 1-2/height, "HISTORICAL MATCHED COMPARISON", fontsize=9, fontweight="bold", va="top", color=ORANGE)
    fig.text(.012, 1-6.5/height, "160 surgical case groups · legacy labels · ESD three-way comparison unavailable",
             fontsize=7.2, va="top", color=MUTED)


if __name__ == "__main__":
    functions = dict(zip("ABCDE", [rocs, auc, calibration, performance, decision]))
    layout = dict(A=(1,63,58,61), B=(61,63,58,61), C=(121,63,58,61),
                  D=(1,2,102,59), E=(105,2,74,59))
    sizes = dict(A=(100,87), B=(105,76), C=(102,87), D=(155,94), E=(118,89))
    build(3, functions, layout, sizes, 135, heading)
