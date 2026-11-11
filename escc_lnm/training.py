"""Nested patient-level feature selection and classifier optimization."""
from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from .io import read_features, read_json, recipe_key, seed_for, sha256, write_json
from .models import RNAFeatures, CalibratedRNAModel, decision_score, save_model
from .search_space import CLASSIFIERS, SELECTORS, COUNTS, build, coarse, sample
from .selection import near_constant_columns, nine_rankings


def split(labels, folds, seed):
    if min(np.bincount(np.asarray(labels, dtype=int), minlength=2)) < folds:
        raise ValueError("Each outcome class needs enough patients for the requested folds")
    return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels))


def select_threshold(labels, probability, target):
    if not 0 < target <= 1:
        raise ValueError("Sensitivity target must be in (0, 1]")
    positive = np.sort(np.asarray(probability)[np.asarray(labels) == 1])
    if not len(positive):
        raise ValueError("Threshold selection requires positive training patients")
    allowed = int(np.floor((1-target)*len(positive)+1e-12))
    return float(positive[min(allowed, len(positive)-1)])


def metric_row(labels, probability):
    high = probability >= .5
    tp = int(np.sum(high & (labels == 1)))
    tn = int(np.sum(~high & (labels == 0)))
    fp = int(np.sum(high & (labels == 0)))
    fn = int(np.sum(~high & (labels == 1)))
    return {
        "AUROC":float(roc_auc_score(labels, probability)),
        "AUPRC":float(average_precision_score(labels, probability)),
        "Brier":float(brier_score_loss(labels, probability)),
        "Accuracy":(tp+tn)/len(labels), "Sensitivity":tp/(tp+fn),
        "Specificity":tn/(tn+fp), "Precision":tp/(tp+fp) if tp+fp else 0.,
        "F1":2*tp/(2*tp+fp+fn), "TP":tp, "TN":tn, "FP":fp, "FN":fn,
    }


def selection_order(row):
    return (-row["inner_mean_AUROC"], -row["inner_mean_AUPRC"], row["Brier"],
            row["inner_SD_AUROC"], row["recipe"]["count"],
            recipe_key({"classifier":row["classifier"], "recipe":row["recipe"]}))


class PatientData:
    def __init__(self, features, manifest):
        frame = read_features(features)
        patients = pd.read_csv(manifest, dtype={"Patient_ID":str})
        if patients.Patient_ID.isna().any() or patients.Patient_ID.duplicated().any():
            raise ValueError("The manifest requires unique patient identifiers")
        patients = patients.sort_values("Patient_ID").reset_index(drop=True)
        if set(patients.Patient_ID) != set(frame.index):
            raise ValueError("Patient manifest and feature rows differ")
        if set(patients.LNM) != {0, 1}:
            raise ValueError("Require both binary LNM classes")
        self.patients = patients
        self.ids = patients.Patient_ID.to_numpy(str)
        self.genes = frame.columns.to_numpy(str)
        self.x = frame.loc[self.ids].to_numpy(float)
        self.y = patients.LNM.to_numpy(int)
        self.rank_cache = {}
        self.feature_cache = {}

    def rankings(self, indices):
        key = tuple(indices)
        if key not in self.rank_cache:
            valid = ~near_constant_columns(self.x[indices])
            if valid.sum() < max(COUNTS):
                raise ValueError("Training subset has fewer than 100 variable RNA features")
            import hashlib
            training_key = hashlib.sha256(self.ids[indices].tobytes()).hexdigest()[:20]
            self.rank_cache[key], _ = nine_rankings(self.x[indices], self.y[indices], valid, seed_for(training_key))
        return self.rank_cache[key]

    def transform(self, indices, recipe):
        key = (tuple(indices), recipe["selector"], recipe["count"])
        if key not in self.feature_cache:
            rankings = self.rankings(indices)
            genes = rankings[recipe["selector"]+"__order"][:recipe["count"]]
            fitted = RNAFeatures(genes).fit(self.x[indices])
            self.feature_cache[key] = (fitted, fitted.transform(self.x))
        return self.feature_cache[key]


def fit_raw(data, indices, recipe, classifier, seed):
    features, values = data.transform(indices, recipe)
    estimator = build(classifier, recipe["config"], seed)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(values[indices], data.y[indices])
    needs_repair = any(issubclass(w.category, ConvergenceWarning) for w in messages)
    if needs_repair and classifier in {"MLP", "Logistic Regression"}:
        estimator = build(classifier, recipe["config"], seed)
        estimator.set_params(max_iter=24000 if classifier == "MLP" else 40000)
        if classifier == "MLP":
            estimator.set_params(max_fun=240000)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(values[indices], data.y[indices])
        if any(issubclass(w.category, ConvergenceWarning) for w in messages):
            raise RuntimeError("Convergence failed after the uniform numerical retry")
    return features, estimator, values


def calibrated_fit(data, train, recipe, classifier, seed, partitions):
    margin = np.full(len(train), np.nan)
    for a, b in partitions:
        features, estimator, values = fit_raw(data, train[a], recipe, classifier, seed)
        margin[b] = decision_score(estimator, values[train[b]])
    if not np.isfinite(margin).all():
        raise ValueError("Calibration partition coverage is incomplete")
    calibrator = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=20261106)
    calibrator.fit(margin.reshape(-1, 1), data.y[train])
    probability = calibrator.predict_proba(margin.reshape(-1, 1))[:, 1]
    features, estimator, _ = fit_raw(data, train, recipe, classifier, seed)
    model = CalibratedRNAModel(features, estimator, calibrator, recipe, data.genes)
    return model, probability


def evaluate_trial(data, train, classifier, recipe, context):
    inner = split(data.y[train], 3, context["inner_split_seed"])
    probability = np.full(len(train), np.nan)
    rows = []
    for j, (a, b) in enumerate(inner):
        fit, holdout = train[a], train[b]
        subfolds = split(data.y[fit], 2, context["calibration_split_seeds"][j])
        model, _ = calibrated_fit(data, fit, recipe, classifier, context["estimator_seeds"][j], subfolds)
        probability[b] = model.predict_proba(data.x[holdout])[:, 1]
        rows.append(metric_row(data.y[holdout], probability[b]))
    result = metric_row(data.y[train], probability)
    return {
        "classifier":classifier, "recipe":recipe, "candidate_id":recipe_key(recipe), **result,
        "inner_mean_AUROC":float(np.mean([r["AUROC"] for r in rows])),
        "inner_SD_AUROC":float(np.std([r["AUROC"] for r in rows], ddof=1)),
        "inner_mean_AUPRC":float(np.mean([r["AUPRC"] for r in rows])),
        "fold_metrics":rows,
    }


def search_classifier(data, train, classifier, context, settings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows, attempts = [], []

    def attempt(recipe, search_pass):
        key = recipe_key(recipe)
        destination = directory/"trials"/(key+".json")
        if destination.exists():
            result = read_json(destination)
        else:
            try:
                result = evaluate_trial(data, train, classifier, recipe, context)
            except RuntimeError as error:
                result = {"classifier":classifier, "recipe":recipe, "error":str(error)}
            write_json(destination, result)
        attempts.append({"attempt":len(attempts)+1, "search_pass":search_pass, **result})
        if "error" not in result:
            rows.append(result)
        write_json(directory/"progress.json", {"completed":len(attempts), "total":192})
        return result

    for recipe in coarse(classifier):
        attempt(recipe, "factorial")
    by_selector = []
    for selector in SELECTORS:
        eligible = [r for r in rows if r["recipe"]["selector"] == selector]
        if eligible:
            by_selector.append(min(eligible, key=selection_order))
    if len(by_selector) < 2:
        raise RuntimeError("Fewer than two selectors have valid search results")
    top = sorted(by_selector, key=selection_order)[:2]
    draws = settings["refinement_candidates"][context["name"]][classifier]
    position = 0
    for candidate in top:
        for count in COUNTS:
            for _ in range(2):
                attempt({"selector":candidate["recipe"]["selector"], "count":count,
                         "config":draws[position]}, "refinement")
                position += 1
    best = min(rows, key=selection_order)
    matched = [attempt({**best["recipe"], "count":count}, "matched_count") for count in COUNTS]
    chosen = min(rows, key=selection_order)
    partitions = split(data.y[train], 3, context["inner_split_seed"])
    model, probability = calibrated_fit(data, train, chosen["recipe"], classifier,
                                         context["final_estimator_seed"], partitions)
    thresholds = {"0.5":.5, **{str(t):select_threshold(data.y[train], probability, t) for t in (.95, .98)}}
    chosen = {**chosen, "thresholds":thresholds, "genes":data.genes[model.features.genes].tolist(),
              "attempts":len(attempts), "unique_trials":len({r["candidate_id"] for r in rows})}
    save_model(model, directory/"model.joblib")
    write_json(directory/"search.json", attempts)
    write_json(directory/"matched_count.json", matched)
    write_json(directory/"selected.json", chosen)
    return chosen, model


def train(features, manifest, settings_path, output):
    from filelock import FileLock
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output/"training.lock"), timeout=0):
        return _train(features, manifest, settings_path, output)


def _train(features, manifest, settings_path, output):
    settings = read_json(settings_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"features":sha256(features), "manifest":sha256(manifest), "settings":sha256(settings_path)}
    identity["implementation"] = recipe_key({
        p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")
    })
    identity_path = output/"run_identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError("Output directory belongs to a different run")
    write_json(identity_path, identity)
    data = PatientData(features, manifest)
    if "outer_fold" not in data.patients:
        data.patients["outer_fold"] = -1
        for fold, (_, test) in enumerate(split(data.y, 5, settings["outer_seed"])):
            data.patients.loc[test, "outer_fold"] = fold
    folds = data.patients.outer_fold.to_numpy()
    if set(folds) != set(range(5)):
        raise ValueError("outer_fold must contain the five values 0 through 4")
    data.patients[["Patient_ID", "outer_fold"]].to_csv(output/"patient_folds.csv", index=False)
    predictions, process, final = [], [], []
    with threadpool_limits(limits=1):
        for context in settings["contexts"]:
            fold = context["fold"]
            train_indices = np.arange(len(data.y)) if fold == -1 else np.flatnonzero(folds != fold)
            test = np.array([], dtype=int) if fold == -1 else np.flatnonzero(folds == fold)
            context_models = []
            for classifier in CLASSIFIERS:
                directory = output/context["name"]/classifier.replace(" ", "_")
                if (directory/"selected.json").exists():
                    from .models import load_model
                    chosen = read_json(directory/"selected.json")
                    model = load_model(directory/"model.joblib")
                else:
                    chosen, model = search_classifier(data, train_indices, classifier, context, settings, directory)
                context_models.append((chosen, model, directory))
                if len(test):
                    probability = model.predict_proba(data.x[test])[:, 1]
                    predictions.extend({
                        "Patient_ID":str(data.ids[i]), "LNM":int(data.y[i]), "outer_fold":int(fold),
                        "classifier":classifier, "probability":float(p),
                    } for i, p in zip(test, probability))
                print(context["name"], classifier, f"inner AUROC {chosen['inner_mean_AUROC']:.4f}", flush=True)
            winner = min(context_models, key=lambda x:selection_order(x[0]))
            if len(test):
                probability = winner[1].predict_proba(data.x[test])[:, 1]
                process.extend({"Patient_ID":str(data.ids[i]), "LNM":int(data.y[i]), "outer_fold":int(fold),
                                "classifier":winner[0]["classifier"], "probability":float(p)}
                               for i,p in zip(test,probability))
            else:
                final = sorted([x[0] for x in context_models], key=selection_order)
                save_model(winner[1], output/"classifier.joblib")
                write_json(output/"model.json", {
                    "classifier":winner[0]["classifier"], "recipe":winner[0]["recipe"],
                    "gene_names":data.genes.tolist(), "selected_genes":winner[0]["genes"],
                    "thresholds":winner[0]["thresholds"], "model_sha256":sha256(output/"classifier.joblib"),
                })
            data.rank_cache.clear()
            data.feature_cache.clear()
            gc.collect()
    pd.DataFrame(predictions).to_csv(output/"classifier_oof.csv", index=False, float_format="%.17g")
    pd.DataFrame(process).to_csv(output/"selection_oof.csv", index=False, float_format="%.17g")
    write_json(output/"classifier_ranking.json", final)
