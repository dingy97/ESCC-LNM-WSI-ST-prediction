"""Editable evaluation figures and H&E attribution panels."""
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm, to_rgb
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path as VectorPath
from sklearn.metrics import roc_curve, precision_recall_curve

from .io import read_json
from .sampling import slide_candidate_mask
from .evaluation import paired_inputs
from .search_space import CLASSIFIERS

BLUE, RED, INK = "#0072B2", "#C44F23", "#182632"
plt.rcParams.update({
    "font.family":"Arial", "font.size":8, "axes.spines.top":False, "axes.spines.right":False,
    "axes.linewidth":.6, "pdf.fonttype":42, "svg.fonttype":"none",
    "figure.facecolor":"white", "savefig.facecolor":"white",
})


def export(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(directory/(name+"."+extension), dpi=600)
    plt.close(fig)


def evaluation_figures(scores, labels, statistics, output):
    joined = paired_inputs(pd.read_csv(scores, dtype={"Patient_ID":str}),
                           pd.read_csv(labels, dtype={"Patient_ID":str}))
    stats = read_json(Path(statistics)/"metrics.json")
    y, probability = joined.LNM.to_numpy(int), joined.probability.to_numpy(float)
    fpr, tpr, _ = roc_curve(y, probability)
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    ax.step(fpr, tpr, where="post", color=BLUE, label=f"AUROC {stats['AUROC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="Sensitivity")
    ax.legend(loc="lower right")
    export(fig, output, "ROC")
    precision, recall, _ = precision_recall_curve(y, probability)
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    ax.step(recall, precision, where="post", color=BLUE)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    export(fig, output, "Precision_recall")
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    matrix = np.asarray([[stats["TN"], stats["FP"]], [stats["FN"], stats["TP"]]])
    for i in range(2):
        for j in range(2):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#DAECEE" if i == j else "#F8DECD", edgecolor="white"))
            ax.text(j+.5, i+.5, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set(xlim=(0, 2), ylim=(2, 0), xticks=[.5, 1.5], xticklabels=["Low", "High"],
           yticks=[.5, 1.5], yticklabels=["LNM−", "LNM+"], xlabel="Predicted group")
    export(fig, output, "Confusion_matrix")
    cal = pd.read_csv(Path(statistics)/"calibration.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    ax.plot(cal.mean_predicted, cal.observed_LNM, "o-", color=BLUE)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=.7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed LNM fraction")
    export(fig, output, "Calibration")
    dca = pd.read_csv(Path(statistics)/"DCA.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.1), layout="constrained")
    for field, color, label in [("model_net_benefit", BLUE, "Model"),
                                ("all_treat_net_benefit", RED, "Treat all"),
                                ("none_net_benefit", "gray", "Treat none")]:
        ax.plot(dca.decision_threshold, dca[field], color=color, label=label)
    ax.set(xlabel="Decision threshold", ylabel="Net benefit")
    ax.legend()
    export(fig, output, "Decision_curve")
    ranked = joined.sort_values("probability", kind="stable")
    fig, ax = plt.subplots(figsize=(4.2, 2.8), layout="constrained")
    ax.bar(np.arange(len(ranked)), ranked.probability, color=np.where(ranked.LNM == 1, RED, BLUE), width=.9)
    ax.axhline(stats["threshold"], color=INK, ls="--", lw=.8)
    ax.set(xlabel="Patient rank", ylabel="Predicted probability", ylim=(0, 1))
    export(fig, output, "Risk_scores")


def classifier_figures(training_output, output):
    from .training import metric_row
    directory = Path(training_output)
    data = pd.read_csv(directory/"classifier_oof.csv")
    if set(data.classifier) != set(CLASSIFIERS):
        raise ValueError("Expected predictions for all twelve classifiers")
    rows, fold_rows = [], []
    fig, ax = plt.subplots(figsize=(4.6, 4), layout="constrained")
    for classifier in CLASSIFIERS:
        d = data[data.classifier.eq(classifier)]
        scores = metric_row(d.LNM.to_numpy(int), d.probability.to_numpy(float))
        rows.append({"classifier":classifier, **scores})
        for fold, f in d.groupby("outer_fold"):
            fold_rows.append({"classifier":classifier, "fold":fold,
                              **metric_row(f.LNM.to_numpy(int), f.probability.to_numpy(float))})
        fpr, tpr, _ = roc_curve(d.LNM, d.probability)
        ax.plot(fpr, tpr, lw=.8, label=f"{classifier} {scores['AUROC']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=.6)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="Sensitivity")
    ax.legend(fontsize=5.5, loc="lower right")
    export(fig, output, "Classifier_ROC")
    summary = pd.DataFrame(rows).set_index("classifier").loc[CLASSIFIERS]
    fields = ["AUROC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "Brier"]
    grid = summary[fields].to_numpy()
    color = grid.copy()
    color[:, -1] = 1-color[:, -1]
    fig, ax = plt.subplots(figsize=(5.2, 4.8), layout="constrained")
    mesh = ax.pcolormesh(color, cmap="Blues", vmin=0, vmax=1, edgecolors="white")
    ax.invert_yaxis()
    ax.set(xticks=np.arange(len(fields))+.5, xticklabels=fields,
           yticks=np.arange(12)+.5, yticklabels=CLASSIFIERS)
    ax.tick_params(axis="x", rotation=45)
    for i in range(12):
        for j in range(len(fields)):
            ax.text(j+.5, i+.5, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if color[i, j] > .65 else INK)
    mesh.set_rasterized(False)
    export(fig, output, "Classifier_metrics")
    folds = pd.DataFrame(fold_rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.5), layout="constrained")
    for i, classifier in enumerate(CLASSIFIERS):
        values = folds.loc[folds.classifier.eq(classifier), "AUROC"].to_numpy()
        ax.scatter(i+np.linspace(-.15, .15, len(values)), values, color=BLUE, s=12)
        ax.plot([i-.2, i+.2], [values.mean()]*2, color=INK, lw=1)
    ax.set(xticks=range(12), xticklabels=CLASSIFIERS, ylabel="Held-out AUROC", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=60)
    export(fig, output, "Classifier_AUROC_distribution")
    ranking = read_json(directory/"classifier_ranking.json")
    fig, ax = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
    ax.errorbar([r["inner_mean_AUROC"] for r in ranking], np.arange(12),
                xerr=[r["inner_SD_AUROC"] for r in ranking], fmt="o", color=BLUE, capsize=2)
    ax.set(yticks=np.arange(12), yticklabels=[r["classifier"] for r in ranking],
           xlabel="Mean inner AUROC ± SD", xlim=(0, 1))
    ax.invert_yaxis()
    export(fig, output, "Classifier_selection")
    winner = ranking[0]["classifier"]
    folder = directory/"full"/winner.replace(" ", "_")
    matched = read_json(folder/"matched_count.json")
    matched = [r for r in matched if "error" not in r]
    fig, ax = plt.subplots(figsize=(4.2, 3.2), layout="constrained")
    ax.errorbar([r["recipe"]["count"] for r in matched], [r["inner_mean_AUROC"] for r in matched],
                yerr=[r["inner_SD_AUROC"] for r in matched], fmt="o-", color=BLUE, capsize=2)
    ax.set(xlabel="Selected RNA features", ylabel="Mean inner AUROC", ylim=(.5, 1))
    export(fig, output, "Feature_count")
    attempts = read_json(folder/"search.json")
    factorial = [r for r in attempts if r["search_pass"] == "factorial" and "error" not in r]
    selectors = list(dict.fromkeys(r["recipe"]["selector"] for r in factorial))
    counts = sorted({r["recipe"]["count"] for r in factorial})
    grid = np.asarray([[max(r["inner_mean_AUROC"] for r in factorial
                            if r["recipe"]["selector"] == selector and r["recipe"]["count"] == count)
                        for count in counts] for selector in selectors])
    fig, ax = plt.subplots(figsize=(4.6, 3.8), layout="constrained")
    ax.pcolormesh(grid, cmap="Blues", vmin=.5, vmax=1, edgecolors="white", rasterized=False)
    ax.invert_yaxis()
    ax.set(xticks=np.arange(len(counts))+.5, xticklabels=counts,
           yticks=np.arange(len(selectors))+.5, yticklabels=selectors,
           xlabel="Selected RNA features", title="Inner AUROC · best of three fixed parameter profiles")
    for i in range(len(selectors)):
        for j in range(len(counts)):
            ax.text(j+.5, i+.5, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=6,
                    color="white" if grid[i,j] > .78 else INK)
    export(fig, output, "Selector_count")
    from .models import load_model
    model = load_model(directory/"classifier.joblib")
    if hasattr(model.estimator, "coef_"):
        weights = model.estimator.coef_[0]*float(model.calibrator.coef_[0,0])
        top = np.argsort(np.abs(weights))[-20:]
        info = read_json(directory/"model.json")
        names = np.asarray(info["selected_genes"])[top]
        fig, ax = plt.subplots(figsize=(4, 4.5), layout="constrained")
        ax.barh(names, weights[top], color=np.where(weights[top] >= 0, RED, BLUE))
        ax.set(xlabel="Calibrated log-odds per feature SD")
        export(fig, output, "RNA_coefficients")
    Path(output).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(output)/"classifier_metrics.csv")
    folds.to_csv(Path(output)/"classifier_folds.csv", index=False)


def tissue_path(mask, scale, box):
    x, y, width, height = box
    left, top = np.floor(np.asarray([x, y])/scale).astype(int)
    right, bottom = np.ceil((np.asarray([x, y])+[width, height])/scale).astype(int)
    contours, _ = cv2.findContours(mask[top:bottom, left:right].astype("uint8"),
                                  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    paths = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, .32, True).reshape(-1, 2)
        if len(contour) >= 3:
            vertices = (contour+[left+.5, top+.5])*scale
            paths.append(VectorPath(np.r_[vertices, vertices[:1]], closed=True))
    return VectorPath.make_compound_path(*paths) if paths else None


def case_figures(attribution_output):
    import openslide
    directory = Path(attribution_output)
    cases = pd.read_csv(directory/"cases.csv")
    regions = pd.read_csv(directory/"fov_attribution.csv", float_precision="round_trip")
    links = pd.read_csv(directory/"private/source_mapping.csv").set_index("anonymous_slide_id")
    bound = read_json(directory/"attribution.json")["global_color_limit"]
    if bound <= 0:
        raise ValueError("Attribution color range is empty")
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    def color(value):
        return to_rgb("#BD3B35" if value > 0 else "#14559F"), .76*np.sqrt(abs(value)/bound)
    palette = [(1-alpha)+alpha*np.asarray(rgb) for rgb, alpha in
               (color(v) for v in np.linspace(-bound, bound, 257))]
    cmap = ListedColormap(palette)
    for case in cases.itertuples():
        values = regions[regions.anonymous_slide_id.eq(case.main_slide)]
        usable = values[values.status.eq("PASS")]
        if usable.empty:
            raise ValueError("The selected WSI has no defined leave-one-FOV-out scores")
        roi = usable.assign(abs_delta=usable.delta_raw_margin.abs()).sort_values(
            ["abs_delta", "FOV_ordinal"], ascending=[False, True]).iloc[0]
        source = links.loc[case.main_slide]
        with np.load(source.cache, allow_pickle=False) as cache:
            mpp, dims = cache["mpp"], cache["dimensions"]
        with closing(openslide.OpenSlide(source.source_path)) as slide:
            image = slide.get_thumbnail((4200, 4200)).convert("RGB")
            small = np.asarray(slide.get_thumbnail((2048, 2048)).convert("RGB"))
            mask, _ = slide_candidate_mask(small, dims, mpp)
            detail = slide.read_region((int(roi.x_level0), int(roi.y_level0)), 0,
                                      (int(roi.width_level0), int(roi.height_level0))).convert("RGB")
        scale = dims/np.asarray(mask.shape[::-1])
        folder = directory/("Main_" if case.category in {"TP", "FN"} else "Supplementary_")
        folder = folder.with_name(folder.name+case.category)
        for panel, title in [
            ("Original_HE", "Original H&E"), ("FOV_locations", "Sampled FOV footprints"),
            ("LNM_attribution_heatmap", "LNM risk attribution"), ("HE_ROI_magnification", "Native-resolution H&E"),
        ]:
            fig = plt.figure(figsize=(114/25.4, 103/25.4))
            fig.text(.035, .968, title, fontsize=9, weight="bold", va="top")
            fig.text(.035, .923, f"{case.anonymous_case_id}  |  {case.category}  |  Probability {case.probability:.3f}",
                     fontsize=7, va="top")
            is_detail = panel == "HE_ROI_magnification"
            width, height = (roi.width_level0, roi.height_level0) if is_detail else dims
            aspect = width*mpp[0]/(height*mpp[1])
            aw = min(.93, .73*103*aspect/114)
            ah = aw*114/(103*aspect)
            ax = fig.add_axes((.5-aw/2, .13+(.73-ah)/2, aw, ah))
            ax.imshow(np.asarray(detail if is_detail else image), extent=(0, width, height, 0),
                      interpolation="none", aspect="auto")
            ax.set(xlim=(0, width), ylim=(height, 0))
            ax.set_axis_off()
            if panel in {"FOV_locations", "LNM_attribution_heatmap"}:
                for row in values.itertuples():
                    box = row.x_level0, row.y_level0, row.width_level0, row.height_level0
                    if panel == "FOV_locations":
                        ax.add_patch(Rectangle(box[:2], box[2], box[3], fill=False, edgecolor="#146A85", lw=.7))
                    elif row.status == "PASS" and row.delta_raw_margin != 0:
                        path = tissue_path(mask, scale, box)
                        if path is not None:
                            rgb, alpha = color(row.delta_raw_margin)
                            patch = PathPatch(path, facecolor=rgb, edgecolor="none", alpha=alpha)
                            ax.add_patch(patch)
                            patch.set_clip_path(Rectangle(box[:2], box[2], box[3], transform=ax.transData))
                ax.add_patch(Rectangle((roi.x_level0, roi.y_level0), roi.width_level0, roi.height_level0,
                                       fill=False, edgecolor="#F4B13D", lw=1.2, ls="--"))
            bar_um = 100 if is_detail else 1000
            length = aw*bar_um/(width*mpp[0])
            fig.add_artist(Line2D([.95-length, .95], [.096, .096], transform=fig.transFigure, color=INK, lw=1.5))
            fig.text(.95, .080, "100 µm" if is_detail else "1 mm", ha="right", va="top", fontsize=6.5)
            if panel == "LNM_attribution_heatmap":
                cax = fig.add_axes((.04, .088, .37, .02))
                bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                                   orientation="horizontal", ticks=[-bound, 0, bound])
                bar.solids.set_rasterized(False)
                bar.solids.set_edgecolor("face")
                fig.text(.04, .027, "Full minus leave-one-FOV-out LR log-odds", fontsize=6.3)
            if is_detail:
                fig.text(.035, .094, f"FOV {int(roi.FOV_ordinal)}  |  Δ log-odds {roi.delta_raw_margin:+.4f}", fontsize=6.5)
            export(fig, folder, panel)
