"""Training-only RNA rankings."""
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression
from .search_space import SELECTORS
from .io import seed_for


def near_constant_columns(x):
    return np.ptp(x, axis=0) <= 8 * np.finfo('float32').eps * np.maximum(1, np.abs(x).mean(0))

def nine_rankings(x, y, valid, seed):
    from boruta import BorutaPy
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    z = StandardScaler().fit_transform(x[:, valid])
    raw = {}
    extra = {}
    models = {'Decision Tree': DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=seed), 'GBDT': GradientBoostingClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, learning_rate=0.03, random_state=seed), 'LightGBM': LGBMClassifier(n_estimators=100, num_leaves=7, min_child_samples=10, learning_rate=0.03, reg_lambda=10, n_jobs=2, verbosity=-1, random_state=seed, deterministic=True, force_col_wise=True), 'Logistic Regression': LogisticRegression(C=0.1, solver='liblinear', dual=False, max_iter=10000, random_state=seed), 'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=4, n_jobs=2, random_state=seed), 'XGBoost': XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.03, reg_lambda=10, tree_method='hist', n_jobs=2, random_state=seed)}
    for name, m in models.items():
        m.fit(z, y)
        raw[name] = np.abs(m.coef_[0]) if hasattr(m, 'coef_') else m.feature_importances_
    boruta = BorutaPy(RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=4, n_jobs=2, random_state=seed), n_estimators=120, max_iter=30, perc=100, random_state=seed, verbose=0)
    boruta.fit(z, y)
    hist = np.nanmedian(boruta.importance_history_[1:], axis=0)
    raw['Boruta'] = 2 * boruta.support_.astype(float) + boruta.support_weak_.astype(float) + rankdata(np.nan_to_num(hist), method='average') / len(hist) / 2
    extra = {'boruta_confirmed_count': int(boruta.support_.sum()), 'boruta_tentative_count': int(boruta.support_weak_.sum()), 'boruta_note': 'Fixed top-k Boruta-informed ranking; top-k membership does not imply confirmed Boruta acceptance'}
    nb = GaussianNB(var_smoothing=0.001).fit(z, y)
    ll = -0.5 * ((z - nb.theta_[1]) ** 2 / nb.var_[1] - (z - nb.theta_[0]) ** 2 / nb.var_[0])
    ll0 = -0.5 * (nb.theta_[1] ** 2 / nb.var_[1] - nb.theta_[0] ** 2 / nb.var_[0])
    raw['Gaussian Naive Bayes'] = np.abs(ll - ll0).mean(0)
    pls = PLSRegression(n_components=2, scale=False).fit(z, y.astype(float))
    energy = (pls.x_scores_ ** 2).sum(0) * (pls.y_loadings_ ** 2).sum(0)
    w = pls.x_weights_
    raw['PLS-DA'] = np.sqrt(z.shape[1] * (w * w / (w * w).sum(0)[None, :] * energy[None, :]).sum(1) / max(energy.sum(), 1e-20))
    out = {}
    indices = np.flatnonzero(valid)
    for name in SELECTORS:
        imp = np.asarray(raw[name], float)
        assert np.isfinite(imp).all()
        rng = np.random.default_rng(seed_for((seed, name)))
        order = np.lexsort((rng.permutation(len(imp)), -imp))
        allimp = np.full(x.shape[1], np.nan)
        allimp[valid] = imp
        out[name + '__importance'] = allimp
        out[name + '__order'] = indices[order]
    return (out, extra)
