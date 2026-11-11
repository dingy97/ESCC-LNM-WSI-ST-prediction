"""Render Figure 2A–I from the audited figure tables."""
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from scipy.special import expit
from style import (ABBR, BLUE, COLORS, GRID, INK, LNM, LNM_LABELS, MARKERS, MUTED, ORANGE,
                   ORDER, build, data, np, plt, ranked_bars, roc_axes, unit_axis)


def heatmap(frame, standalone):
    d = data("2A").set_index("classifier").loc[ORDER]
    metrics = ["AUROC", "AUPRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Brier"]
    raw = d[metrics].to_numpy()
    visual = raw.copy()
    visual[:, -1] = 1 - raw[:, -1]
    cmap, norm = plt.colormaps["Blues"], Normalize(0, 1)
    frame.header("A", "Outer OOF metrics", "160 patients · threshold 0.50")
    ax = frame.axes(left=12, bottom=14, right=2, top=21)
    ax.pcolormesh(visual, cmap=cmap, norm=norm, edgecolors="white", linewidth=.55, rasterized=False)
    ax.set(xlim=(0, 8), ylim=(12, 0), xticks=np.arange(8)+.5, yticks=np.arange(12)+.5)
    ax.set_xticklabels(["AUROC", "AUPRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Brier ↓"],
                       rotation=55, ha="left", rotation_mode="anchor", fontsize=6.7)
    ax.set_yticklabels([ABBR[n] for n in ORDER])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=2)
    for label, name in zip(ax.get_yticklabels(), ORDER):
        label.set_color(COLORS[name])
        label.set_fontweight("bold" if name == ORDER[0] else "normal")
    ax.get_yticklabels()[0].set_fontweight("bold")
    for i in range(12):
        for j in range(8):
            r, g, b, _ = cmap(norm(visual[i, j]))
            light = .2126*r + .7152*g + .0722*b
            ax.text(j+.5, i+.5, f"{raw[i,j]:.3f}", ha="center", va="center",
                    color="white" if light < .48 else INK, fontsize=6.4 if not standalone else 7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    key = frame.axes(left=16, bottom=7.5, right=11, top=frame.h-10.0)
    key.pcolormesh(np.linspace(0, 1, 48)[None, :], cmap=cmap, norm=norm, rasterized=False)
    key.set_axis_off()
    frame.text(frame.w/2, 5.8, "Darker = higher metric or lower Brier", fontsize=6.6, ha="center", va="top")


def fold_distribution(frame, standalone):
    d = data("2B")
    frame.header("B", "AUROC by outer fold", "Five held-out folds per classifier")
    ax = frame.axes(left=9.8, bottom=13.5, right=2, top=12)
    values = [d.loc[d.classifier.eq(name)].sort_values("outer_fold").AUROC for name in ORDER]
    boxes = ax.boxplot(values, positions=np.arange(12), widths=.62, showfliers=False,
                       patch_artist=True, medianprops=dict(color=INK, lw=.8),
                       whiskerprops=dict(color=MUTED, lw=.55), capprops=dict(color=MUTED, lw=.55))
    for i, (box, name, y) in enumerate(zip(boxes["boxes"], ORDER, values)):
        box.set(facecolor=COLORS[name], alpha=.2, edgecolor=COLORS[name], linewidth=.85)
        ax.scatter(i+np.linspace(-.20, .20, 5), y, s=11, marker=MARKERS[name], color=COLORS[name],
                   linewidths=.3, edgecolors="white", zorder=3)
    ax.set(xlim=(-.6, 11.6), xticks=range(12), ylabel="AUROC")
    ax.set_xticklabels([ABBR[n] for n in ORDER], rotation=68, ha="right", rotation_mode="anchor")
    for label, name in zip(ax.get_xticklabels(), ORDER):
        label.set_color(COLORS[name])
    unit_axis(ax, "y")
    ax.axhline(.5, color=MUTED, ls="--", lw=.6, zorder=0)
    ax.grid(axis="y")


def ranking(frame, standalone):
    d = data("2C").sort_values("ranking")
    frame.header("C", "Training-only ranking", "Inner AUROC mean ± SD · three folds")
    ax = frame.axes(left=11.5, bottom=10, right=9, top=11)
    for i, row in enumerate(d.itertuples()):
        name = row.classifier
        ax.errorbar(row.inner_CV_AUROC_mean, i, xerr=row.inner_CV_AUROC_SD, fmt=MARKERS[name],
                    ms=4.8 if row.selected_final else 3.3, color=COLORS[name], lw=.8, capsize=1.6, zorder=3)
        ax.text(1.035, i, f"{row.inner_CV_AUROC_mean:.3f}", ha="left", va="center", fontsize=6.7,
                transform=ax.get_yaxis_transform(), color=COLORS[name],
                fontweight="bold" if row.selected_final else "normal")
    ax.axhspan(-.47, .47, color="#E6F3FA", zorder=-2)
    ax.set(yticks=range(12), yticklabels=[ABBR[n] for n in d.classifier], ylim=(11.65,-.65),
           xlabel="Mean inner AUROC")
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[0].set_color(BLUE)
    unit_axis(ax, "x", step=.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")


def metrics_facets(frame, standalone):
    d = data("2D").set_index(["metric", "classifier"])
    frame.header("D", "Outer-fold metric profiles", "Five held-out folds · mean ± SD · threshold 0.50")
    labels = ["AUROC", "Accuracy", "Precision", "F1", "Sensitivity", "Specificity"]
    left, gap = 12, 2.0
    width = (frame.w-left-2-5*gap)/6
    for j, metric in enumerate(labels):
        start = left + j*(width+gap)
        ax = frame.axes(left=start, bottom=7, right=frame.w-start-width, top=15)
        table = d.loc[metric].loc[ORDER]
        for i, name in enumerate(ORDER):
            value, sd = table.loc[name, ["mean", "std"]]
            ax.errorbar(value, i, xerr=sd, fmt=MARKERS[name], color=COLORS[name],
                        ms=2.8, lw=.65, capsize=1.1)
        unit_axis(ax, "x", step=.5)
        ax.set_xticklabels(["0" if j == 0 else "", "0.5", "1"])
        ax.set(yticks=range(12), yticklabels=[ABBR[n] for n in ORDER] if j == 0 else [],
               ylim=(11.55, -.55), title=metric)
        ax.title.set_fontsize(7)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")


def internal_roc(frame, standalone):
    d = data("2E")
    frame.header("E", "Patient-level outer ROC", "160 surgical patients · 100 LNM+")
    ax = frame.axes(left=11, bottom=10, right=2, top=11)
    roc_axes(ax)
    for name, color, ls, label in [("LR family", BLUE, "-", "LR family"),
                                    ("Full selection procedure", "#565E68", "--", "Full selection")]:
        g = d[d.series.eq(name)]
        ax.step(g.FPR, g.TPR, where="post", color=color, ls=ls, lw=1.25,
                label=f"{label}  {g.AUROC.iloc[0]:.3f}")
    ax.legend(loc="lower right", facecolor="white", frameon=True, framealpha=.94, edgecolor="none")


def oof_scores(frame, standalone):
    d = data("2F")
    frame.header("F", "Ranked surgical OOF scores", "LR family · 160 patients · research cutoff 0.50")
    ax = frame.axes(left=12, bottom=10, right=2, top=11)
    ranked_bars(ax, d)
    ax.legend(handles=[Patch(facecolor=c, label=LNM_LABELS[i]) for i,c in LNM.items()],
              ncol=2, loc="upper left")


def coefficients(frame, standalone):
    d = data("2G").head(10)
    frame.header("G", "Signed RNA coefficients", "Frozen LR · top 10 of 100")
    ax = frame.axes(left=13, bottom=10, right=2, top=11)
    beta = d.calibrated_logit_coefficient
    ax.barh(np.arange(10), beta, height=.61, color=np.where(beta > 0, ORANGE, BLUE), linewidth=0)
    ax.axvline(0, color=INK, lw=.7)
    ax.set(yticks=range(10), yticklabels=d.gene, ylim=(9.6,-.6),
           xlabel="Calibrated log-odds per SD")
    bound = .5
    ax.set(xlim=(-bound,bound), xticks=[-.5, -.25, 0, .25, .5])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")


def waterfall(frame, standalone):
    d = data("2H")
    p0, p1 = expit(d.reference_logit.iloc[0]), d.final_probability.iloc[0]
    frame.header("H", "Prediction decomposition", f"Reference p={p0:.3f} · example p={p1:.3f}")
    ax = frame.axes(left=21 if standalone else 20, bottom=10, right=2, top=11)
    y = np.arange(len(d))
    ax.barh(y, d.contribution, left=d.start_logit, height=.53,
            color=np.where(d.contribution>0, ORANGE, BLUE), linewidth=0)
    for i in range(len(d)-1):
        ax.plot([d.end_logit.iloc[i]]*2, [i+.27, i+1-.27], color=MUTED, lw=.6, ls=":")
    ax.axvline(d.reference_logit.iloc[0], color=MUTED, lw=.65, ls="--")
    ax.scatter(d.final_logit.iloc[0], y[-1], s=17, c=INK, zorder=4)
    ax.set(yticks=y, yticklabels=d.feature, ylim=(5.65,-.65),
           xlabel="Calibrated log-odds", xlim=(0, 2.35), xticks=[0,.5,1,1.5,2])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")


def dependence(frame, standalone):
    d = data("2I")
    frame.header("I", "Feature contribution", f"{d.gene.iloc[0]} · image-inferred RNA")
    ax = frame.axes(left=12, bottom=10, right=2, top=11)
    x = d.standardized_inferred_RNA
    limits = [x.min(), x.max()]
    ax.plot(limits, np.array(limits)*d.coefficient.iloc[0], color=INK, lw=.75, alpha=.65)
    for outcome, group in d.groupby("LNM"):
        ax.scatter(group.standardized_inferred_RNA, group.contribution, color=LNM[outcome],
                   marker="o" if outcome==0 else "^", s=13, lw=.22, edgecolors="white",
                   alpha=.75, label=LNM_LABELS[outcome])
    ax.axhline(0, color=MUTED, ls="--", lw=.65)
    ax.set(xlabel=f"Standardized inferred {d.gene.iloc[0]}", ylabel="Log-odds contribution")
    ax.grid(True)
    ax.legend(loc="upper right")


if __name__ == "__main__":
    draw = dict(zip("ABCDEFGHI", [heatmap, fold_distribution, ranking, metrics_facets,
                                    internal_roc, oof_scores, coefficients, waterfall, dependence]))
    layout = dict(A=(1,116,63,110), B=(65,176,55,50), C=(122,176,57,50), D=(65,116,114,57),
                  E=(1,61,65,53), F=(68,61,111,53), G=(1,1,55,58), H=(58,1,65,58), I=(125,1,54,58))
    sizes = dict(A=(99,116), B=(130,76), C=(99,83), D=(164,85), E=(97,87),
                 F=(143,80), G=(91,79), H=(102,79), I=(91,79))
    build(2, draw, layout, sizes, 227)
