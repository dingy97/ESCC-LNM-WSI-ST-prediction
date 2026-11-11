"""Compare archived ESD labels with their complement at fixed scores and thresholds."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve

OUT = Path(__file__).resolve().parents[1]
STAGE23 = OUT.parent
ROOT = STAGE23.parent
STAGE21 = ROOT / "21_surgical_RNA_LNM_development/clinical_LNM_v2"
ESD = STAGE21 / "04_ESD_validation"
REPORTS = STAGE21 / "07_reports"
PRIMARY = "ESD_unseen_pipeline_case_groups"
RETAINED = "restricted_ESD_retained_bridge_sensitivity"
COHORTS = {PRIMARY: 59, RETAINED: 60}
SCENARIOS = {"original": "Original labels", "reversed": "Label-reversal diagnostic scenario"}
PANEL_WORKPOINT = "0.98"
sys.path.insert(0, str(ROOT / "12_stage5_high_sensitivity/scripts"))
from clinical_utils import metrics as rates


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / "tables" / f"{name}.csv", index=False, float_format="%.17g")


def scores(y, p):
    return (roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p))


def bootstrap(y, p):
    """Reuse the archived 2,000 stratified draws for both label scenarios."""
    rng = np.random.default_rng(20261107)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = [np.r_[rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)] for _ in range(2000)]
    result = {}
    for scenario, labels in (("original", y), ("reversed", 1-y)):
        lo, hi = np.quantile([scores(labels[ix], p[ix]) for ix in draws], [.025, .975], axis=0)
        result[scenario] = {k: v for name, a, b in zip(("AUROC", "AUPRC", "Brier"), lo, hi)
                            for k, v in ((name+"_CI_low", a), (name+"_CI_high", b))}
    return result


def calibration(y, p):
    n_bins = min(8, len(y)//5)
    observed, predicted = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    bins = np.searchsorted(np.percentile(p, np.linspace(0, 100, n_bins+1))[1:-1], p)
    count = np.bincount(bins, minlength=n_bins)
    count = count[count > 0]
    assert len(count) == len(observed) and count.sum() == len(p)
    return pd.DataFrame(dict(bin=np.arange(1, len(count)+1), cases=count,
        mean_predicted=predicted, observed_LNM=observed, LNM_positive=np.rint(observed*count).astype(int)))


def decision(y, p, grid):
    call = p[:, None] >= grid
    tp = ((y[:, None] == 1) & call).mean(0)
    fp = ((y[:, None] == 0) & call).mean(0)
    weight = grid/(1-grid)
    return pd.DataFrame(dict(decision_threshold=grid, model_net_benefit=tp-fp*weight,
        all_treat_net_benefit=y.mean()-(1-y.mean())*weight, none_net_benefit=0.))


def check_archived(row, archived):
    for key in archived.index.intersection(row.keys()):
        value = row[key]
        if isinstance(value, (int, float, np.number)):
            np.testing.assert_allclose(value, archived[key], rtol=1e-12, atol=1e-12, err_msg=key)


def main():
    lock_path = STAGE21 / "config/final_model_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    cuts = lock["winner"]["thresholds"]
    model = Path(lock["winner_directory"]) / "model.joblib"
    assert sha(model) == lock["model_SHA256"]
    archive_paths = [REPORTS / name for name in ("frozen_external_model_metrics.csv",
        "external_ROC_coordinates.csv", "external_calibration.csv", "external_DCA.csv", "external_cohort_disposition.csv")]
    archived, old_roc, old_cal, old_dca, disposition = [read(path) for path in archive_paths]
    frozen_paths = [lock_path, model, *archive_paths]
    frozen_paths += [p for folder in ("figure2", "figure3", "figure4")
                     for p in (STAGE23 / folder).rglob("*") if p.is_file()]
    frozen_paths += list((STAGE21 / "05_TCGA_evaluation").glob("*_frozen_predictions.csv"))
    frozen_paths += list(ESD.glob("*_frozen_predictions.csv"))
    frozen_paths += [ESD / name for name in ("ESD_primary_analysis_manifest.csv", "ESD_analysis_manifest.csv",
                     "patient_LNM_folder_authority.json")]
    input_hashes = {path.relative_to(ROOT).as_posix(): sha(path) for path in frozen_paths}
    source_predictions = {}
    for cohort, n in COHORTS.items():
        d = read(ESD / f"{cohort}_frozen_predictions.csv")
        assert d.Patient_ID.is_unique and len(d) == n and d.LNM.sum() == 5
        assert set(d.LNM) == {0, 1} and d.probability.between(0, 1).all()
        assert d.model_SHA256.eq(lock["model_SHA256"]).all()
        manifest_name = "ESD_primary_analysis_manifest.csv" if cohort == PRIMARY else "ESD_analysis_manifest.csv"
        manifest = read(ESD / manifest_name).set_index("Patient_ID")
        assert set(manifest.index) == set(d.Patient_ID)
        np.testing.assert_array_equal(d.LNM, manifest.loc[d.Patient_ID, "LNM"])
        np.testing.assert_array_equal(d.LNM, manifest.loc[d.Patient_ID, "positive_directory_slides"].gt(0))
        assert not d.development_overlap.any()
        assert d.bridge_overlap.sum() == (0 if cohort == PRIMARY else 1)
        for key, cut in cuts.items():
            np.testing.assert_array_equal(d.probability.ge(cut), d["high_risk_"+key])
        source_predictions[cohort] = d
    paired = source_predictions[RETAINED].set_index("Patient_ID").loc[source_predictions[PRIMARY].Patient_ID]
    np.testing.assert_array_equal(paired.probability, source_predictions[PRIMARY].probability)
    np.testing.assert_array_equal(paired.LNM, source_predictions[PRIMARY].LNM)
    intervals = {c: bootstrap(d.LNM.to_numpy(int), d.probability.to_numpy(float)) for c, d in source_predictions.items()}

    metrics_both = []
    for scenario, display in SCENARIOS.items():
        tables = {key: [] for key in ("scores", "metrics", "ROC", "PR", "calibration", "DCA", "cohort_summary")}
        for cohort, d in source_predictions.items():
            orig = d.LNM.to_numpy(int)
            y = orig if scenario == "original" else 1-orig
            p = d.probability.to_numpy(float)
            auc, ap, brier = scores(y, p)
            meta = dict(scenario=scenario, scenario_label=display, cohort=cohort)
            ranked = pd.DataFrame(dict(LNM=y, probability=p, original_LNM=orig))
            ranked = ranked.sort_values("probability", kind="stable").reset_index(drop=True)
            ranked.insert(0, "rank", np.arange(1, len(d)+1))
            for key, cut in cuts.items():
                ranked["high_risk_"+key] = ranked.probability.ge(cut)
                m = rates(y, p >= cut)
                tp, tn, fp, fn = (m[k] for k in ("TP", "TN", "FP", "FN"))
                row = {**meta, "training_workpoint": float(key), "locked_threshold": cut,
                    "AUROC": auc, "AUPRC": ap, "Brier": brier, "Accuracy": (tp+tn)/len(y),
                    "Precision": m["PPV"], "F1": 2*tp/(2*tp+fp+fn),
                    **m, **intervals[cohort][scenario]}
                if scenario == "original":
                    ref = archived[(archived.cohort == cohort) & np.isclose(archived.training_workpoint, float(key))].iloc[0]
                    check_archived(row, ref)
                tables["metrics"].append(pd.DataFrame([row]))
                metrics_both.append(row)
            tables["scores"].append(ranked.assign(**meta))
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            cal = calibration(y, p)
            old = old_dca[old_dca.cohort.eq(cohort)]
            dca = decision(y, p, old.decision_threshold.to_numpy())
            if scenario == "original":
                np.testing.assert_allclose(np.c_[fpr, tpr], old_roc[old_roc.cohort.eq(cohort)][["FPR", "TPR"]], atol=1e-14)
                np.testing.assert_allclose(cal[["mean_predicted","observed_LNM"]],
                    old_cal[old_cal.cohort.eq(cohort)][["mean_predicted","observed_LNM"]], atol=1e-14)
                np.testing.assert_allclose(dca, old[dca.columns], atol=1e-14)
            tables["ROC"].append(pd.DataFrame(dict(FPR=fpr, TPR=tpr, AUROC=auc,
                CI_low=intervals[cohort][scenario]["AUROC_CI_low"], CI_high=intervals[cohort][scenario]["AUROC_CI_high"])).assign(**meta))
            tables["PR"].append(pd.DataFrame(dict(recall=recall, precision=precision, AUPRC=ap, prevalence=y.mean())).assign(**meta))
            tables["calibration"].append(cal.assign(**meta))
            tables["DCA"].append(dca.assign(**meta))
            old_n = disposition[disposition.cohort.eq(cohort)].iloc[0]
            assert old_n.case_or_patient_groups == len(d) and old_n.WSI_count == d.usable_WSI_count.sum()
            tables["cohort_summary"].append(pd.DataFrame([dict(**meta, groups=len(d), H_and_E_slides=int(d.usable_WSI_count.sum()),
                positive=int(y.sum()), negative=int((1-y).sum()), prevalence=y.mean(),
                mean_prediction=p.mean(), mean_prediction_minus_observed=p.mean()-y.mean(), Brier=brier,
                weighted_absolute_bin_error=np.average(abs(cal.mean_predicted-cal.observed_LNM), weights=cal.cases))]))
        for key, frames in tables.items():
            save(f"ESD_{key}_{scenario}", pd.concat(frames, ignore_index=True))

    both = pd.DataFrame(metrics_both)
    for _, group in both.groupby(["cohort", "training_workpoint"]):
        orig, rev = group.iloc[0], group.iloc[1]
        np.testing.assert_allclose([orig.AUROC+rev.AUROC, orig.Accuracy+rev.Accuracy], [1,1], atol=1e-14)
        assert (orig.TP, orig.TN, orig.FP, orig.FN) == (rev.FP, rev.FN, rev.TP, rev.TN)
        assert orig.high_risk_patients == rev.high_risk_patients and orig.locked_threshold == rev.locked_threshold
    save("ESD_workpoints_both_scenarios", both)
    (OUT / "reports/source_audit.json").write_text(json.dumps(dict(model_SHA256=lock["model_SHA256"],
        label_operation="Case-level complement 1-y after the archived ESD cohort and folder-derived outcomes were fixed",
        locked_thresholds=cuts, figure4_workpoint=PANEL_WORKPOINT,
        original_archived_metrics_ROC_calibration_and_DCA_reproduced=True,
        original_label_source="Archived folder-derived labels; clinical direction under review",
        reversal_clinically_confirmed=False, source_SHA256=input_hashes), indent=2), encoding="utf-8")
    print(both.loc[both.cohort.eq(PRIMARY), ["scenario", "locked_threshold", "AUROC", "Accuracy",
        "Sensitivity", "Specificity", "TP", "FN", "FP", "TN"]].to_string(index=False))


if __name__ == "__main__":
    main()
