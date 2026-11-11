"""Render both ESD label scenarios using the existing Figure 4 and publication style."""
import sys
import pandas as pd

from recompute_esd import OUT, STAGE23, SCENARIOS, PRIMARY, RETAINED, PANEL_WORKPOINT, read

sys.path.insert(0, str(STAGE23 / "figure4/scripts"))
import draw_figure4 as original_layout
from style import BLUE, INK, MM, MUTED, ORANGE, TEAL, Frame, export, np, plt, roc_axes, unit_axis


def table(key, scenario):
    return read(OUT / "tables" / f"ESD_{key}_{scenario}.csv")


class ScenarioFrame(Frame):
    def __init__(self, fig, bounds, scenario, cutoff):
        super().__init__(fig, bounds)
        self.scenario = scenario
        self.cutoff = cutoff

    def header(self, letter, title, note=None, note_color=MUTED):
        if letter in ("C", "F") and title in ("Confusion matrix", "Ranked ESD scores"):
            title += f" ({self.cutoff:.3f})"
        if self.scenario == "reversed" and title == "Observed LNM by group":
            title = "Hypothetical LNM by group"
        super().header(letter, title, SCENARIOS[self.scenario], ORANGE if self.scenario == "reversed" else MUTED)


def render(scenario, kind, funcs, sizes, layout, height, cutoff, n_positive):
    folder = OUT / ("figure4_all_panels" if kind == "Figure4" else "supplementary")
    for letter, draw in funcs.items():
        w, h = sizes[letter]
        fig = plt.figure(figsize=(w*MM, h*MM))
        draw(ScenarioFrame(fig, (2, 1, w-4, h-2), scenario, cutoff), True)
        export(fig, folder, f"{kind}{letter}_{scenario}")
    fig = plt.figure(figsize=(180*MM, height*MM))
    color = ORANGE if scenario == "reversed" else INK
    fig.text(.012, 1-2/height, SCENARIOS[scenario].upper(), fontsize=9, fontweight="bold", va="top", color=color)
    subtitle = (f"ESD primary · 59 accession groups · {n_positive} LNM+ · frozen cutoff {cutoff:.6f}"
                if kind == "Figure4" else "ESD supplementary · primary n=59 · bridge-retained n=60 · fixed model")
    fig.text(.012, 1-6.5/height, subtitle, fontsize=7.2, va="top", color=MUTED)
    for letter, draw in funcs.items():
        draw(ScenarioFrame(fig, layout[letter], scenario, cutoff), False)
    base = f"{kind}_{scenario}_labels_combined" if kind == "Figure4" else f"{kind}_{scenario}_combined"
    export(fig, folder, base)


def draw_figure4(scenario):
    metrics = table("metrics", scenario)
    m = metrics[metrics.cohort.eq(PRIMARY) & np.isclose(metrics.training_workpoint, float(PANEL_WORKPOINT))].iloc[0]
    cutoff = m.locked_threshold
    ranked = table("scores", scenario).query("cohort == @PRIMARY").assign(threshold=cutoff)
    roc = table("ROC", scenario).query("cohort == @PRIMARY")
    inputs = {"4A": ranked, "4F": ranked,
        "4C": pd.DataFrame([dict(actual=a, predicted=b, cell=cell, n=int(m[cell]), threshold=cutoff)
                for a,b,cell in [(0,0,"TN"),(0,1,"FP"),(1,0,"FN"),(1,1,"TP")]]),
        "4D": pd.DataFrame([dict(group=group, cases=int(m[cases]), LNM_positive=int(m[num]), fraction=m[key],
                CI_low=m[key+"_CI_low"], CI_high=m[key+"_CI_high"], threshold=cutoff)
                for group,cases,num,key in [("Low","low_risk_patients","FN","residual_LNM_in_low_risk"),
                                             ("High","high_risk_patients","TP","PPV")]]),
        "4E": pd.DataFrame([dict(metric=key, value=m[key], CI_low=m[key+"_CI_low"], CI_high=m[key+"_CI_high"])
                for key in ("Sensitivity","Specificity","PPV","NPV")])}
    # The archived panel functions receive only this scenario's read-only plot data.
    original_layout.data = inputs.__getitem__

    def panel_b(frame, standalone):
        frame.header("B", "Frozen-model ROC")
        ax = frame.axes(left=10.5, bottom=10.5, right=2, top=12)
        roc_axes(ax)
        ax.step(roc.FPR, roc.TPR, where="post", color=BLUE, lw=1.5)
        row = roc.iloc[0]
        x, y, ha, va = (.06, .93, "left", "top") if scenario == "original" else (.96, .07, "right", "bottom")
        ax.text(x, y, f"AUROC {row.AUROC:.3f}\n95% CI {row.CI_low:.3f}–{row.CI_high:.3f}",
                transform=ax.transAxes, ha=ha, va=va, color=BLUE, fontsize=7.4)

    def panel_d(frame, standalone):
        original_layout.observed(frame, standalone)
        if scenario == "reversed":
            frame.fig.axes[-1].set_ylabel("Hypothetical LNM fraction")

    f = original_layout
    funcs = dict(zip("ABCDEF", [f.distribution, panel_b, f.matrix, panel_d, f.clinical_metrics, f.ranking]))
    layout = dict(A=(1,58,58,51), B=(61,58,58,51), C=(121,58,58,51),
                  D=(1,1,58,54), E=(61,1,58,54), F=(121,1,58,54))
    sizes = dict(A=(89,78), B=(95,85), C=(86,73), D=(88,76), E=(99,76), F=(116,79))
    render(scenario, "Figure4", funcs, sizes, layout, 121, cutoff, int(m.positive))


def draw_supplement(scenario):
    roc, cal, dca = [table(key, scenario) for key in ("ROC", "calibration", "DCA")]
    series = [(PRIMARY, "Primary", BLUE, "-", "o"), (RETAINED, "Bridge-retained", TEAL, "--", "s")]

    def key(frame, ax, ncol=1):
        ax.legend(loc="lower center", bbox_to_anchor=((frame.x+frame.w/2)/frame.fw, (frame.y+1.0)/frame.fh),
                  bbox_transform=frame.fig.transFigure, fontsize=6.5, ncol=ncol, borderaxespad=0)

    def panel_a(frame, standalone):
        frame.header("A", "Frozen-model ROC")
        ax = frame.axes(left=11, bottom=20, right=2, top=12)
        roc_axes(ax)
        for name, label, color, ls, _ in series:
            d = roc[roc.cohort.eq(name)]
            ax.step(d.FPR, d.TPR, where="post", color=color, lw=1.1, ls=ls, label=f"{label} {d.AUROC.iloc[0]:.3f}")
        key(frame, ax)

    def panel_b(frame, standalone):
        frame.header("B", "Score calibration")
        ax = frame.axes(left=11, bottom=20, right=2, top=12)
        unit_axis(ax)
        ax.plot([0,1], [0,1], color=MUTED, lw=.65, ls=":")
        for name, label, color, ls, marker in series:
            d = cal[cal.cohort.eq(name)]
            ax.plot(d.mean_predicted, d.observed_LNM, color=color, ls=ls, marker=marker, ms=3.4, lw=1.05, label=label)
        ylabel = "Observed LNM fraction" if scenario == "original" else "Hypothetical LNM fraction"
        ax.set(xlabel="Mean frozen probability", ylabel=ylabel)
        ax.grid(True)
        key(frame, ax)

    def panel_c(frame, standalone):
        frame.header("C", "Decision curves")
        ax = frame.axes(left=12, bottom=22, right=2, top=12)
        for name, label, color, ls, _ in series:
            d = dca[dca.cohort.eq(name)]
            ax.plot(d.decision_threshold, d.model_net_benefit, color=color, lw=1.2, ls=ls, label=label)
        for (name, _, _, ls, _), label in zip(series, ("All (primary)", "All (retained)")):
            d = dca[dca.cohort.eq(name)]
            ax.plot(d.decision_threshold, d.all_treat_net_benefit, color="#6E7C87", lw=.9, ls=ls, label=label)
        ax.axhline(0, color=INK, ls=":", lw=.8, label="Treat none")
        ax.set(xlim=(.01,.5), ylim=(-1,1), xticks=[.01,.25,.5], yticks=[-1,-.5,0,.5,1],
               xlabel="Decision threshold", ylabel="Net benefit")
        ax.grid(True)
        key(frame, ax, ncol=2)

    render(scenario, "Supplement_ESD_external", dict(A=panel_a, B=panel_b, C=panel_c),
           dict(A=(100,91), B=(100,91), C=(117,92)),
           dict(A=(1,1,58,64), B=(61,1,58,64), C=(121,1,58,64)), 77, 0, 0)


if __name__ == "__main__":
    for scenario in SCENARIOS:
        draw_figure4(scenario)
        draw_supplement(scenario)
