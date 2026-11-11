"""Patient-weighted Ridge regression from image embeddings to RNA."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import KFold

from .io import write_json
from .models import save_model


def patient_weights(groups):
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    return weights/weights.mean()


def moments(values, weights):
    mean = np.average(values, axis=0, weights=weights)
    scale = np.sqrt(np.average((values-mean)**2, axis=0, weights=weights))
    return mean, np.maximum(scale, 1e-6)


def patient_mse(actual, predicted, groups):
    errors = np.mean((actual-predicted)**2, axis=1)
    return float(np.mean([errors[groups == group].mean() for group in np.unique(groups)]))


class TargetSpace:
    def __init__(self, values, counts, negative, groups, quantile=.95, fraction=.1):
        self.weights = patient_weights(groups)
        self.mean, self.scale = moments(values, self.weights)
        detected = counts > np.quantile(negative, quantile, axis=1)[:, None]
        self.support = np.stack([
            detected[groups == group].mean(0) >= .5 for group in np.unique(groups)
        ]).mean(0)
        self.variable = np.average((values-self.mean)**2, axis=0, weights=self.weights) > 1e-12
        self.mask = (self.support >= fraction) & self.variable
        if not self.mask.any():
            raise ValueError("No RNA target meets the training support and variance rules")

    def transform(self, values):
        return (values[:, self.mask]-self.mean[self.mask])/self.scale[self.mask]

    def decode(self, values):
        result = np.tile(self.mean, (len(values), 1))
        result[:, self.mask] += values*self.scale[self.mask]
        return result


def dual_fit(features, targets, weights, penalty):
    root_weight = np.sqrt(weights)
    design = features*root_weight[:, None]
    coefficients = cho_solve(
        cho_factor(design@design.T + penalty*np.eye(len(design)), lower=True),
        targets*root_weight[:, None],
    )
    return design, coefficients


def dual_predict(features, regressor):
    return (features@regressor[0].T)@regressor[1]


class RidgeBridge:
    def fit(self, features, targets, counts, negative, groups, alpha, primary=None):
        weights = patient_weights(groups)
        self.mu, self.sd = moments(features, weights)
        self.target = TargetSpace(targets, counts, negative, groups)
        normalized = (features-self.mu)/self.sd
        self.regressor = dual_fit(normalized, self.target.transform(targets), weights, alpha*len(features))
        self.correction = None
        self.baseline = self.target.mean.copy()
        if primary is not None:
            primary = np.asarray(primary, dtype=bool)
            xp = normalized[primary]
            wp = patient_weights(groups[primary])
            residual = self.target.transform(targets)[primary]-dual_predict(xp, self.regressor)
            self.cm, _ = moments(residual, wp)
            self.xm, _ = moments(xp, wp)
            self.correction = dual_fit(xp-self.xm, residual-self.cm, wp, len(xp))
            self.baseline = np.average(targets[primary], axis=0, weights=wp)
        return self

    def predict(self, features):
        normalized = (np.asarray(features, dtype=float)-self.mu)/self.sd
        latent = dual_predict(normalized, self.regressor)
        if self.correction is not None:
            latent += .5*(self.cm+dual_predict(normalized-self.xm, self.correction))
        result = self.target.decode(latent)
        result[:, ~self.target.mask] = self.baseline[~self.target.mask]
        return np.clip(result, 0, np.log1p(1e6)).astype("float32")


def train_bridge(input_path, output, alphas=(.1, 10., 1000.), seed=20261104):
    from pathlib import Path
    with np.load(input_path, allow_pickle=False) as bundle:
        features = bundle["embeddings"]
        targets = bundle["log_expression"]
        counts = bundle["counts"]
        negative = bundle["negative_counts"]
        groups = bundle["Patient_ID"].astype(str)
        genes = bundle["gene_names"].astype(str).tolist()
        folds = bundle["outer_fold"]
        primary = bundle["primary"].astype(bool) if "primary" in bundle else np.ones(len(groups), dtype=bool)
    if len(features) != len(targets) or len(groups) != len(features):
        raise ValueError("FOV arrays must be row-aligned")
    for group in np.unique(groups):
        if len(np.unique(folds[groups == group])) != 1:
            raise ValueError("A patient's FOVs cross outer folds")
    predictions = np.empty_like(targets, dtype="float32")
    records = []

    def choose(indices, fold_seed):
        patients = np.unique(groups[indices])
        scores = []
        for alpha in alphas:
            values = []
            splitter = KFold(3, shuffle=True, random_state=fold_seed)
            for a, b in splitter.split(patients):
                train = indices[np.isin(groups[indices], patients[a])]
                validation = indices[np.isin(groups[indices], patients[b]) & primary[indices]]
                if not len(validation):
                    raise ValueError("An inner validation fold has no primary FOVs")
                model = RidgeBridge().fit(features[train], targets[train], counts[train],
                                           negative[train], groups[train], alpha)
                values.append(patient_mse(targets[validation], model.predict(features[validation]), groups[validation]))
            scores.append(float(np.mean(values)))
        return float(alphas[int(np.argmin(scores))]), scores

    for fold in np.unique(folds):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        alpha, scores = choose(train, seed+int(fold))
        model = RidgeBridge().fit(features[train], targets[train], counts[train], negative[train], groups[train], alpha)
        predictions[test] = model.predict(features[test])
        evaluated = test[primary[test]]
        if not len(evaluated):
            raise ValueError("An outer validation fold has no primary FOVs")
        records.append({"fold":int(fold), "alpha":alpha, "inner_MSE":scores,
                        "patient_MSE":patient_mse(targets[evaluated], predictions[evaluated], groups[evaluated])})
    alpha, scores = choose(np.arange(len(groups)), seed)
    model = RidgeBridge().fit(features, targets, counts, negative, groups, alpha)
    model.gene_names = genes
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_model(model, output/"bridge.joblib")
    np.savez_compressed(output/"bridge_oof.npz", prediction=predictions, Patient_ID=groups, gene_names=genes)
    write_json(output/"bridge_cv.json", {"folds":records, "alpha":alpha, "inner_MSE":scores})
