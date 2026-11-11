"""Evaluate fixed patient predictions and explicitly supplied binary outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def paired_inputs(scores, labels):
    for name, frame in (("scores", scores), ("labels", labels)):
        if "Patient_ID" not in frame or frame.Patient_ID.isna().any() or not frame.Patient_ID.is_unique:
            raise ValueError(f"{name} must contain unique nonmissing Patient_ID values")
    if set(scores.Patient_ID) != set(labels.Patient_ID):
        raise ValueError("Score and outcome cohorts differ")
    joined = scores.drop(columns=["LNM"], errors="ignore").merge(
        labels[["Patient_ID", "LNM"]], on="Patient_ID", validate="one_to_one")
    if set(joined.LNM) != {0, 1} or not joined.probability.between(0, 1).all():
        raise ValueError("Require both binary outcome classes and probabilities in [0, 1]")
    return joined


def interval(successes, total):
    if total == 0:
        return [None, None]
    lo = 0. if successes == 0 else float(beta.ppf(.025, successes, total-successes+1))
    hi = 1. if successes == total else float(beta.ppf(.975, successes+1, total-successes))
    return [lo, hi]


def metrics(y, probability, cutoff):
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    high = probability >= cutoff
    tp = int(np.sum(high & (y == 1)))
    fp = int(np.sum(high & (y == 0)))
    tn = int(np.sum(~high & (y == 0)))
    fn = int(np.sum(~high & (y == 1)))
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp) if tp+fp else None
    npv = tn/(tn+fn) if tn+fn else None
    return {
        "patients": len(y), "positive": int(y.sum()), "threshold": cutoff,
        "AUROC": float(roc_auc_score(y, probability)), "AUPRC": float(average_precision_score(y, probability)),
        "Brier": float(brier_score_loss(y, probability)), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Accuracy": (tp+tn)/len(y), "Sensitivity": sensitivity, "Specificity": specificity,
        "Sensitivity_95CI": interval(tp, tp+fn), "Specificity_95CI": interval(tn, tn+fp),
        "FNR": 1-sensitivity, "PPV": precision, "NPV": npv,
        "PPV_95CI": interval(tp, tp+fp), "NPV_95CI": interval(tn, tn+fn),
        "F1": 2*tp/(2*tp+fp+fn), "low_risk_patients": tn+fn,
        "residual_LNM_in_low_risk": fn/(tn+fn) if tn+fn else None,
        "residual_LNM_95CI": interval(fn, tn+fn),
    }


def evaluate(scores_path, labels_path, output, cutoff, draws=2000, bootstrap_reference=None):
    data = paired_inputs(pd.read_csv(scores_path, dtype={"Patient_ID": str}),
                         pd.read_csv(labels_path, dtype={"Patient_ID": str}))
    y, p = data.LNM.to_numpy(int), data.probability.to_numpy(float)
    result = metrics(y, p, cutoff)
    rng = np.random.default_rng(20261107)
    strata = y
    if bootstrap_reference is not None:
        reference = paired_inputs(data[["Patient_ID", "probability"]],
                                  pd.read_csv(bootstrap_reference, dtype={"Patient_ID": str}))
        strata = reference.LNM.to_numpy(int)
        if not (np.array_equal(strata, y) or np.array_equal(strata, 1-y)):
            raise ValueError("Bootstrap reference must define the same unordered binary strata")
    positive, negative = np.flatnonzero(strata == 1), np.flatnonzero(strata == 0)
    resampled = []
    for _ in range(draws):
        index = np.r_[rng.choice(positive, len(positive), True), rng.choice(negative, len(negative), True)]
        resampled.append([roc_auc_score(y[index], p[index]), average_precision_score(y[index], p[index]),
                          brier_score_loss(y[index], p[index])])
    ci = np.quantile(resampled, [.025, .975], axis=0)
    result.update({name+"_95CI": ci[:, i].tolist() for i, name in enumerate(("AUROC", "AUPRC", "Brier"))})
    output.mkdir(parents=True, exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    observed, predicted = calibration_curve(y, p, n_bins=max(1, min(8, len(y)//5)), strategy="quantile")
    pd.DataFrame({"mean_predicted": predicted, "observed_LNM": observed}).to_csv(output/"calibration.csv", index=False)
    rows = []
    for threshold in np.linspace(.01, .5, 100):
        high = p >= threshold
        weight = threshold/(1-threshold)
        rows.append({
            "decision_threshold": threshold,
            "model_net_benefit": np.mean(high & (y == 1))-np.mean(high & (y == 0))*weight,
            "all_treat_net_benefit": y.mean()-(1-y.mean())*weight,
            "none_net_benefit": 0.,
        })
    pd.DataFrame(rows).to_csv(output/"DCA.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="Patient_ID,LNM using the confirmed outcome direction")
    parser.add_argument("--threshold", type=float, required=True, help="Previously locked development threshold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reference-labels", type=Path,
                        help="Optional original outcome table to preserve archived bootstrap draw order after label correction")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be in [0, 1]")
    result = evaluate(args.scores, args.labels, args.output, args.threshold,
                      bootstrap_reference=args.bootstrap_reference_labels)
    print(f"Evaluated {result['patients']} fixed patient predictions")


if __name__ == "__main__":
    main()
