"""Feature scaling, calibrated classifiers and checkpoint loading."""
from pathlib import Path

import joblib
import numpy as np
from joblib.numpy_pickle import NumpyUnpickler, _read_fileobject
from scipy.special import logit
from sklearn.preprocessing import StandardScaler


def decision_score(estimator, features):
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(features), dtype=float)
    return logit(np.clip(estimator.predict_proba(features)[:, 1], 1e-6, 1-1e-6))


class RNAFeatures:
    def __init__(self, genes):
        self.genes = np.asarray(genes, dtype=int)

    def fit(self, values):
        self.scaler = StandardScaler().fit(values[:, self.genes])
        return self

    def transform(self, values):
        return self.scaler.transform(np.asarray(values)[:, self.genes])


class CalibratedRNAModel:
    def __init__(self, features, estimator, calibrator, recipe, gene_names=None):
        self.features = features
        self.estimator = estimator
        self.calibrator = calibrator
        self.recipe = recipe
        self.gene_names = list(gene_names) if gene_names is not None else None

    def raw_margin(self, values):
        return decision_score(self.estimator, self.features.transform(values))

    def predict_proba(self, values):
        margin = self.raw_margin(values)
        probability = self.calibrator.predict_proba(margin.reshape(-1, 1))[:, 1]
        return np.c_[1-probability, probability]


class ModelReader(NumpyUnpickler):
    def find_class(self, module, name):
        if name in {"RNAFeatures", "CalibratedRNAModel"}:
            return globals()[name]
        if name in {"FullRidge", "TargetSpace"}:
            from .bridge import RidgeBridge, TargetSpace
            return {"FullRidge": RidgeBridge, "TargetSpace": TargetSpace}[name]
        return super().find_class(module, name)


def load_model(path):
    """Read a trusted joblib checkpoint into the package's model classes."""
    path = Path(path)
    with path.open("rb") as stream:
        with _read_fileobject(stream, str(path), mmap_mode=None) as decoded:
            return ModelReader(str(path), decoded, mmap_mode=None).load()


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(model, temporary, compress=3)
    temporary.replace(path)
