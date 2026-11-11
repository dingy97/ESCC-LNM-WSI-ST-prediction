"""Extract de-identified figure tables from archived predictions and a frozen model."""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
V21 = ROOT / "21_surgical_RNA_LNM_development/clinical_LNM_v2"
V10 = ROOT / "10_lnm_sampling_corrected"
ESD = V21 / "04_ESD_validation"
INTERNAL = V21 / "03_internal_evaluation"
sys.path.insert(0, str(V21 / "scripts"))
import train_all_12  # Initializes the project's pinned scientific runtime.
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

METRICS = ["AUROC", "AUPRC", "Accuracy", "Precision", "Sensitivity", "Specificity", "F1", "Brier"]
INPUTS = {}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = sha(path)
    return path


def csv(path, **kwargs):
    return pd.read_csv(source(path), float_precision="round_trip", **kwargs)


def read_json(path):
    return json.loads(source(path).read_text(encoding="utf-8"))


def save(panel, frame):
    path = OUT / f"figure{panel[0]}/source_data/Figure{panel}.csv"
    pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8", float_format="%.17g")


def confusion(y, pred):
    return {name: int(np.sum((y == label) & (pred == decision)))
            for name, label, decision in [("TP", 1, 1), ("FN", 1, 0), ("FP", 0, 1), ("TN", 0, 0)]}


def metrics(y, p, cutoff):
    pred = p >= cutoff
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p),
        Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
        Sensitivity=recall_score(y, pred), Specificity=recall_score(y, pred, pos_label=0),
        F1=f1_score(y, pred), Brier=brier_score_loss(y, p))


def roc(y, p, series, interval):
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=False)
    auc, lo, hi = interval
    return pd.DataFrame(dict(series=series, FPR=fpr, TPR=tpr, AUROC=auc, CI_low=lo, CI_high=hi,
                             patients=len(y), positive=int(sum(y))))


def interval(row):
    return tuple(float(row[k]) for k in ("AUROC", "AUROC_CI_low", "AUROC_CI_high"))


def ranked(y, p, cutoff):
    d = pd.DataFrame(dict(LNM=np.asarray(y), probability=np.asarray(p)))
    d = d.sort_values("probability", kind="stable").reset_index(drop=True)
    d.insert(0, "rank", np.arange(1, len(d) + 1))
    return d.assign(threshold=cutoff)


def figure2(truth, lock, model, genes, x):
    archive = pd.read_parquet(source(INTERNAL / "all_12_patient_OOF.parquet"))
    archive = archive[archive.target.eq(.5)]
    pooled = csv(INTERNAL / "all_12_outer_OOF_metrics.csv")
    folds = csv(INTERNAL / "all_12_outer_fold_metrics.csv")
    search = csv(V21 / "07_reports/final_12_model_comparison.csv").sort_values("ranking")
    assert len(archive) == 1920 and len(pooled) == len(search) == 12 and len(folds) == 60
    for name, d in archive.groupby("classifier"):
        y, p = d.LNM.to_numpy(), d.probability.to_numpy()
        assert len(d) == 160 and d.Patient_ID.is_unique and sum(y) == 100
        np.testing.assert_array_equal(y, truth.loc[d.Patient_ID].LNM)
        np.testing.assert_array_equal(d.outer_fold, truth.loc[d.Patient_ID].outer_fold)
        row = pooled.set_index("classifier").loc[name]
        for k, value in metrics(y, p, .5).items():
            assert abs(row[k] - value) < 1e-12, (name, k)
        for fold, fold_d in d.groupby("outer_fold"):
            saved = folds[(folds.classifier == name) & (folds.outer_fold == fold)].iloc[0]
            for k, value in metrics(fold_d.LNM.to_numpy(), fold_d.probability.to_numpy(), .5).items():
                assert abs(saved[k] - value) < 1e-12, (name, fold, k)
    assert search.iloc[0].classifier == lock["winner_classifier"] == "Logistic Regression"
    assert abs(search.iloc[0].inner_CV_AUROC_mean - lock["winner"]["inner_mean_AUROC"]) < 1e-12
    save("2A", pooled[["classifier", "patients", *METRICS]])
    save("2B", folds[["classifier", "outer_fold", "patients", "AUROC"]])
    save("2C", search[["classifier", "ranking", "inner_CV_AUROC_mean", "inner_CV_AUROC_SD", "selected_final"]])
    long = folds.melt(id_vars=["classifier", "outer_fold"], value_vars=["AUROC", "Accuracy",
        "Precision", "F1", "Sensitivity", "Specificity"], var_name="metric", value_name="score")
    summary = long.groupby(["classifier", "metric"]).score.agg(["mean", "std", "count"]).reset_index()
    save("2D", summary)
    lr = archive[archive.classifier.eq("Logistic Regression")]
    process = pd.read_parquet(source(INTERNAL / "complete_selection_process_OOF.parquet"))
    process = process[process.target.eq(.5)]
    process_row = csv(INTERNAL / "complete_model_selection_process_OOF_metrics.csv").iloc[0]
    assert process.Patient_ID.is_unique and len(process) == 160
    np.testing.assert_array_equal(process.LNM, truth.loc[process.Patient_ID].LNM)
    assert abs(roc_auc_score(process.LNM, process.probability) - process_row.AUROC) < 1e-12
    save("2E", pd.concat([roc(lr.LNM, lr.probability, "LR family",
                            interval(pooled[pooled.classifier.eq("Logistic Regression")].iloc[0])),
                           roc(process.LNM, process.probability, "Full selection procedure", interval(process_row))]))
    save("2F", ranked(lr.LNM, lr.probability, .5))

    # Exact additive decomposition on the final calibrated log-odds scale.
    assert np.array_equal(model.estimator.classes_, [0, 1])
    assert np.array_equal(model.calibrator.classes_, [0, 1])
    z = model.features.transform(x.to_numpy(float))
    selected = np.asarray(genes)[model.features.genes]
    np.testing.assert_array_equal(selected, csv(V21 / "07_reports/final_model_RNA_features.csv").gene)
    raw_beta = model.estimator.coef_.ravel()
    slope = model.calibrator.coef_.item()
    beta = slope * raw_beta
    base = slope * model.estimator.intercept_.item() + model.calibrator.intercept_.item()
    contribution = z * beta
    score = model.predict_proba(x.to_numpy(float))[:, 1]
    np.testing.assert_allclose(expit(base + contribution.sum(1)), score, rtol=1e-13, atol=1e-13)
    order = np.argsort(-np.abs(beta), kind="stable")
    save("2G", pd.DataFrame(dict(gene=selected[order], raw_LR_coefficient=raw_beta[order],
                                calibrated_logit_coefficient=beta[order], rank=np.arange(1, 101))))
    # Input is ordered by identifier before this stable median-rank selection.
    case = np.argsort(score, kind="stable")[(len(score) - 1) // 2]
    top = np.argsort(-np.abs(contribution[case]), kind="stable")[:5]
    remaining = np.setdiff1d(np.arange(len(beta)), top)
    step = np.r_[contribution[case, top], contribution[case, remaining].sum()]
    start = base + np.r_[0, np.cumsum(step)[:-1]]
    save("2H", pd.DataFrame(dict(step=np.arange(1, 7), feature=[*selected[top], "Other 95 features"],
        contribution=step, start_logit=start, end_logit=start + step, reference_logit=base,
        final_logit=logit(score[case]), final_probability=score[case],
        score_rank=80, selection_pool=160)))
    lead = order[0]
    save("2I", pd.DataFrame(dict(gene=selected[lead], standardized_inferred_RNA=z[:, lead],
        contribution=contribution[:, lead], LNM=truth.loc[x.index].LNM.to_numpy(), coefficient=beta[lead])))
    return dict(lr_OOF=float(roc_auc_score(lr.LNM, lr.probability)), full_OOF=float(process_row.AUROC),
        inner_AUROC=float(search.iloc[0].inner_CV_AUROC_mean),
        extra_trees_OOF=float(pooled.set_index("classifier").loc["Extra Trees", "AUROC"]),
        top_coefficient_gene=str(selected[lead]), example_probability=float(score[case]),
        example_reference_logit=float(base), example_probability_at_reference=float(expit(base)))


def figure3(truth):
    historical = csv(V10 / "predictions/main_nested_selected_OOF.csv")
    historical = historical[historical.workflow.isin(["A", "B", "D"])]
    aligned = historical.pivot(index="Patient_ID", columns="workflow", values=["LNM", "outer_fold", "probability"])
    assert aligned.shape == (160, 9) and aligned.notna().all().all()
    for key in ("LNM", "outer_fold"):
        assert aligned[key].nunique(axis=1).eq(1).all()
    mapping = csv(V21 / "01_feature_mapping/patient_feature_reconstruction_audit.csv").set_index("legacy_Patient_ID")
    assert mapping.index.is_unique and set(aligned.index) == set(mapping.index)
    current = truth.loc[mapping.loc[aligned.index].Patient_ID]
    old_y = aligned["LNM"]["A"].to_numpy(int)
    disagreements = int(np.sum(old_y != current.LNM.to_numpy()))
    changed_folds = int(np.sum(aligned["outer_fold"]["A"].to_numpy() != current.outer_fold.to_numpy()))
    assert old_y.sum() == 102 and disagreements == 2
    perf = csv(V10 / "results/main_model_performance.csv").set_index("model")
    paired = csv(V10 / "results/main_paired_input_differences.csv").set_index("comparison")
    rocs, ci, calib, met, dca = [], [], [], [], []
    prevalence = old_y.mean()
    # Use the archived threshold grid and rederive all net benefits from the aligned OOF scores.
    grid = np.sort(csv(V10 / "results/decision_curve_data.csv").threshold.unique())
    for name in ("A", "B", "D"):
        p = aligned["probability"][name].to_numpy(float)
        row = perf.loc[f"selected/{name}"]
        values = metrics(old_y, p, .5)
        for key, value in values.items():
            assert abs(value - row[key]) < 1e-12, (name, key)
        rocs.append(roc(old_y, p, name, interval(row)))
        delta = paired.loc[f"{name}-A"] if name != "A" else {}
        ci.append(dict(series=name, AUROC=row.AUROC, CI_low=row.AUROC_CI_low, CI_high=row.AUROC_CI_high,
            delta_AUROC=delta.get("delta_AUROC", 0.), paired_CI_low=delta.get("paired_CI_low", np.nan),
            paired_CI_high=delta.get("paired_CI_high", np.nan)))
        observed, predicted = calibration_curve(old_y, p, n_bins=5, strategy="quantile")
        calib.extend(dict(series=name, mean_score=a, event_fraction=b, Brier=row.Brier)
                     for a, b in zip(predicted, observed))
        met.extend(dict(series=name, metric=key, value=value, cutoff=.5) for key, value in values.items())
        for cutoff in grid:
            counts = confusion(old_y, p >= cutoff)
            dca.append(dict(series=name, threshold=cutoff,
                            net_benefit=(counts["TP"] - counts["FP"] * cutoff / (1 - cutoff)) / len(old_y)))
    for cutoff in grid:
        dca.extend([dict(series="Treat all", threshold=cutoff,
                         net_benefit=prevalence - (1 - prevalence) * cutoff / (1 - cutoff)),
                    dict(series="Treat none", threshold=cutoff, net_benefit=0.)])
    save("3A", pd.concat(rocs))
    save("3B", ci)
    save("3C", calib)
    save("3D", met)
    save("3E", dca)
    return dict(patient_groups=160, old_positive=102, current_positive=100,
                label_disagreements=disagreements, changed_outer_fold_assignments=changed_folds,
                historical_fusion="visual + inferred RNA + four image-derived programs",
                current_matched_comparison_available=False, three_way_ESD_predictions_available=False)


def figure4(lock, model, genes):
    cohort = "ESD_unseen_pipeline_case_groups"
    manifest = csv(ESD / "ESD_primary_analysis_manifest.csv").set_index("Patient_ID")
    pred = csv(ESD / f"{cohort}_frozen_predictions.csv")
    authority = read_json(ESD / "patient_LNM_folder_authority.json")
    ready = read_json(ESD / "ESD_features_ready.json")
    assert ready["authority_sha256"] == INPUTS[(ESD / "patient_LNM_folder_authority.json").relative_to(ROOT).as_posix()]
    for filename, digest in ready["outputs"].items():
        assert sha(source(ESD / filename)) == digest
    features = pd.read_parquet(ESD / "ESD_case_inferred_RNA.parquet")
    audit = csv(ESD / "ESD_case_folder_LNM_audit.csv").set_index("Patient_ID")
    assert authority["status"] == "AUTHOR_CONFIRMED_PATIENT_LNM"
    assert len(pred) == 59 and pred.Patient_ID.is_unique and set(pred.Patient_ID) == set(manifest.index)
    assert not (pred.development_overlap | pred.bridge_overlap).any()
    assert pred.model_SHA256.eq(lock["model_SHA256"]).all()
    assert pred.usable_WSI_count.sum() == 749
    assert list(features.columns) == list(genes)
    y, p = pred.LNM.to_numpy(int), pred.probability.to_numpy(float)
    np.testing.assert_array_equal(y, audit.loc[pred.Patient_ID].positive_directory_slides.gt(0))
    np.testing.assert_array_equal(y, manifest.loc[pred.Patient_ID].LNM)
    np.testing.assert_allclose(model.predict_proba(features.loc[pred.Patient_ID, genes].to_numpy(float))[:, 1],
                               p, atol=1e-13, rtol=1e-13)
    stats = csv(V21 / "07_reports/frozen_external_model_metrics.csv")
    stats = stats[stats.cohort.eq(cohort)]
    for key, cutoff in lock["winner"]["thresholds"].items():
        row = stats[np.isclose(stats.training_workpoint, float(key))].iloc[0]
        assert row.locked_threshold == cutoff
        assert abs(roc_auc_score(y, p) - row.AUROC) < 1e-12
        np.testing.assert_array_equal(p >= cutoff, pred[f"high_risk_{key}"])
        assert all(row[k] == val for k, val in confusion(y, p >= cutoff).items())
    cutoff = lock["winner"]["thresholds"]["0.98"]
    row = stats[np.isclose(stats.training_workpoint, .98)].iloc[0]
    risk = ranked(y, p, cutoff)
    save("4A", risk)
    save("4B", roc(y, p, "Frozen LR", interval(row)))
    save("4C", [dict(actual=a, predicted=b, cell=cell, n=int(row[cell]), threshold=cutoff)
                for a, b, cell in [(0, 0, "TN"), (0, 1, "FP"), (1, 0, "FN"), (1, 1, "TP")]])
    save("4D", [dict(group=group, cases=int(row[cases]), LNM_positive=int(row[num]), fraction=row[metric],
                     CI_low=row[metric + "_CI_low"], CI_high=row[metric + "_CI_high"], threshold=cutoff)
                for group, cases, num, metric in [("Low", "low_risk_patients", "FN", "residual_LNM_in_low_risk"),
                                                 ("High", "high_risk_patients", "TP", "PPV")]])
    save("4E", [dict(metric=key, value=row[key], CI_low=row[key + "_CI_low"], CI_high=row[key + "_CI_high"],
                     numerator=int(row[num]), denominator=int(row[num] + row[other]), threshold=cutoff)
                for key, num, other in [("Sensitivity", "TP", "FN"), ("Specificity", "TN", "FP"),
                                        ("PPV", "TP", "FP"), ("NPV", "TN", "FN")]])
    save("4F", risk)
    stats[["training_workpoint", "locked_threshold", "AUROC", "AUPRC", "TP", "FP", "TN", "FN",
           "Sensitivity", "Specificity", "PPV", "NPV"]].to_csv(OUT / "reports/ESD_locked_workpoints.csv", index=False)
    return dict(groups=len(y), positive=int(y.sum()), WSI=749, AUROC=row.AUROC, AUROC_CI=list(interval(row)[1:]),
        AUPRC=row.AUPRC, threshold=cutoff, counts=confusion(y, p >= cutoff),
        status="diagnostic_unresolved", probability_direction_reproduced=True,
        case_group_labels_and_exclusions_verified=True, archived_feature_hashes_verified=True)


def main():
    qa = read_json(V21 / "logs/final_artifact_QA.json")
    assert qa["all_pass"]
    truth = csv(V21 / "00_inventory/surgical_patient_manifest.csv").set_index("Patient_ID")
    assert len(truth) == 160 and truth.LNM.sum() == 100
    lock = read_json(V21 / "config/final_model_lock.json")
    checkpoint = source(Path(lock["winner_directory"]) / "model.joblib")
    assert INPUTS[checkpoint.relative_to(ROOT).as_posix()] == lock["model_SHA256"]
    model = joblib.load(checkpoint)
    x = pd.read_parquet(source(V21 / "01_feature_mapping/surgical_patient_inferred_RNA.parquet")).sort_index()
    assert set(x.index) == set(truth.index)
    audit = dict(Figure2=figure2(truth, lock, model, x.columns, x), Figure3=figure3(truth),
                 Figure4=figure4(lock, model, x.columns), source_SHA256=INPUTS,
                 model_SHA256=lock["model_SHA256"], model_fitting_performed=False)
    (OUT / "reports/source_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("Figure2", "Figure3", "Figure4")}, indent=2))


if __name__ == "__main__":
    main()
