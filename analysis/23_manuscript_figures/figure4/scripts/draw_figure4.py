"""Render the frozen-model ESD risk-stratification diagnostic, Figure 4A–F."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figure2/scripts"))
from matplotlib.patches import Patch
from style import BLUE, INK, LNM, LNM_LABELS, MUTED, ORANGE, TEAL, build, data, np, ranked_bars, roc_axes, unit_axis

DIAGNOSTIC = "ESD diagnostic · unresolved transfer"


def distribution(frame, standalone):
    d = data("4A")
    frame.header("A", "Scores by LNM status", DIAGNOSTIC, ORANGE)
    ax = frame.axes(left=11, bottom=10, right=2, top=12)
    vals = [d.loc[d.LNM.eq(i)].probability.to_numpy() for i in (0,1)]
    boxes = ax.boxplot(vals, positions=[0,1], widths=.45, showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1.0),
                       whiskerprops=dict(color=MUTED, lw=.65), capprops=dict(color=MUTED, lw=.65))
    for i, (box, arr) in enumerate(zip(boxes["boxes"], vals)):
        box.set(facecolor=LNM[i], alpha=.22, edgecolor=LNM[i])
        jitter = ((np.arange(len(arr))*.61803398875)%1-.5)*.46
        ax.scatter(i+jitter, arr, color=LNM[i], marker="o" if i==0 else "^", s=15 if i==0 else 23,
                   linewidth=.34, edgecolors="white", alpha=.82, zorder=3, clip_on=False)
    ax.axhline(d.threshold.iloc[0], color=INK, lw=.7, ls="--")
    ax.set(xlim=(-.6,1.6), xticks=[0,1], xticklabels=[f"{LNM_LABELS[i]} (n={len(vals[i])})" for i in (0,1)],
           ylabel="Predicted probability")
    unit_axis(ax, "y")
    ax.grid(axis="y")


def external_roc(frame, standalone):
    d = data("4B")
    r = d.iloc[0]
    frame.header("B", "Frozen-model ROC", DIAGNOSTIC, ORANGE)
    ax = frame.axes(left=10.5, bottom=10.5, right=2, top=12)
    roc_axes(ax)
    ax.step(d.FPR, d.TPR, where="post", color=BLUE, lw=1.5)
    ax.text(.06,.93,f"AUROC {r.AUROC:.3f}\n95% CI {r.CI_low:.3f}–{r.CI_high:.3f}",
            transform=ax.transAxes, va="top", color=BLUE, fontsize=7.4)


def matrix(frame, standalone):
    d = data("4C")
    threshold = d.threshold.iloc[0]
    frame.header("C", "Confusion matrix", f"Frozen cutoff {threshold:.3f} · diagnostic", ORANGE)
    ax = frame.axes(left=12, bottom=11, right=2, top=13)
    for row in d.itertuples():
        ax.add_patch(PatchCell(row.predicted, row.actual, row.actual == row.predicted))
        ax.text(row.predicted+.5, row.actual+.44, str(row.n), fontsize=13, ha="center", va="center", fontweight="bold")
        ax.text(row.predicted+.5, row.actual+.72, row.cell, fontsize=7, color=MUTED, ha="center", va="center")
    ax.set(xlim=(0,2), ylim=(2,0), xticks=[.5,1.5], xticklabels=["Low", "High"],
           yticks=[.5,1.5], yticklabels=["LNM−", "LNM+"], xlabel="Predicted risk group")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def PatchCell(x, y, correct):
    from matplotlib.patches import Rectangle
    return Rectangle((x,y), 1, 1, facecolor="#DAECEE" if correct else "#F8DECD",
                     edgecolor="white", linewidth=1.5)


def observed(frame, standalone):
    d = data("4D")
    frame.header("D", "Observed LNM by group", DIAGNOSTIC, ORANGE)
    ax = frame.axes(left=12, bottom=12, right=2, top=12)
    x = np.arange(len(d))
    ax.bar(x, d.fraction, width=.5, color=[TEAL,ORANGE], alpha=.9, lw=0)
    ax.errorbar(x, d.fraction, yerr=[d.fraction-d.CI_low, d.CI_high-d.fraction],
                fmt="none", color=INK, lw=.8, capsize=2.4)
    ax.set(xlim=(-.57,1.57), xticks=x, xticklabels=[f"{r.group}\n{r.LNM_positive}/{r.cases} LNM+" for r in d.itertuples()],
           ylabel="Observed LNM fraction")
    unit_axis(ax, "y", percentages=True)
    ax.grid(axis="y")


def clinical_metrics(frame, standalone):
    d = data("4E")
    frame.header("E", "Classification performance", "Diagnostic · exact binomial 95% CI", ORANGE)
    ax = frame.axes(left=19, bottom=10.5, right=2, top=12)
    y = np.arange(len(d))
    ax.errorbar(d.value, y, xerr=[d.value-d.CI_low, d.CI_high-d.value], color=BLUE, fmt="o",
                ms=4, lw=1.1, capsize=2)
    ax.set(yticks=y, yticklabels=d.metric, ylim=(3.65,-.65), xlabel="Proportion (95% CI)")
    unit_axis(ax, "x", percentages=True, step=.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")


def ranking(frame, standalone):
    d = data("4F")
    frame.header("F", "Ranked ESD scores", f"Frozen cutoff {d.threshold.iloc[0]:.3f} · diagnostic", ORANGE)
    ax = frame.axes(left=11, bottom=10, right=2, top=17)
    ranked_bars(ax,d)
    ax.legend(handles=[Patch(facecolor=c, label=LNM_LABELS[i]) for i,c in LNM.items()],
              loc="upper left", ncol=2, bbox_transform=frame.fig.transFigure,
              bbox_to_anchor=((frame.x+11)/frame.fw, (frame.y+frame.h-10.2)/frame.fh), borderaxespad=0)


def heading(fig, height):
    fig.text(.012,1-2/height,"ESD TRANSFER DIAGNOSTIC",fontsize=9,fontweight="bold",va="top",color=ORANGE)
    fig.text(.012,1-6.5/height,"59 accession groups · 5 LNM+ · 749 H&E slides · clinical high-risk status unverified",
             fontsize=7.2,va="top",color=MUTED)


if __name__ == "__main__":
    functions = dict(zip("ABCDEF", [distribution, external_roc, matrix, observed, clinical_metrics, ranking]))
    layout = dict(A=(1,58,58,51), B=(61,58,58,51), C=(121,58,58,51),
                  D=(1,1,58,54), E=(61,1,58,54), F=(121,1,58,54))
    sizes = dict(A=(89,78), B=(95,85), C=(86,73), D=(88,76), E=(99,76), F=(116,79))
    build(4, functions, layout, sizes, 121, heading)
