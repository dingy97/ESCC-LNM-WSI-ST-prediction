import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from escc_lnm.attribution import choose_cases, leave_one_out
from escc_lnm.bridge import RidgeBridge, patient_weights, train_bridge
from escc_lnm.evaluation import paired_inputs, metrics
from escc_lnm.models import RNAFeatures, CalibratedRNAModel, load_model, save_model
from escc_lnm.sampling import spatial_sample, candidate_fovs
from escc_lnm.search_space import CLASSIFIERS, SELECTORS, COUNTS, coarse, build
from escc_lnm.training import calibrated_fit, search_classifier, selection_order, select_threshold


def classifier():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(40, 5))
    y = (x[:, 0]+x[:, 1] > 0).astype(int)
    features = RNAFeatures([0, 1, 2]).fit(x)
    estimator = LogisticRegression(C=1).fit(features.transform(x), y)
    margin = estimator.decision_function(features.transform(x))
    calibrator = LogisticRegression(C=1).fit(margin.reshape(-1, 1), y)
    return CalibratedRNAModel(features, estimator, calibrator, {}, ["g"+str(i) for i in range(5)])


class WorkflowTests(unittest.TestCase):
    def test_classifier_checkpoint_roundtrip(self):
        model = classifier()
        x = np.arange(15).reshape(3, 5)/10
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"classifier.joblib"
            save_model(model, p)
            np.testing.assert_array_equal(load_model(p).predict_proba(x), model.predict_proba(x))

    def test_fov_removal_recalculates_slide_and_patient_means(self):
        model = classifier()
        rng = np.random.default_rng(6)
        slides = [rng.normal(size=(4, 5)).astype("float32"), rng.normal(size=(3, 5)).astype("float32")]
        full, _, results = leave_one_out(model, slides)
        expected = np.asarray([s.mean(0) for s in slides]).mean(0)
        np.testing.assert_array_equal(full, expected)
        for j, slide in enumerate(slides):
            for i in range(len(slide)):
                means = np.asarray([s.mean(0) for s in slides])
                means[j] = np.delete(slide, i, 0).mean(0)
                actual = model.raw_margin(full.astype(float).reshape(1, -1))[0]-model.raw_margin(means.mean(0).astype(float).reshape(1, -1))[0]
                self.assertAlmostEqual(results[j].delta_raw_margin.iloc[i], actual, places=12)

    def test_single_fov_has_undefined_removal(self):
        _, _, results = leave_one_out(classifier(), [np.ones((1, 5), dtype="float32")])
        self.assertTrue(np.isnan(results[0].delta_raw_margin.iloc[0]))
        self.assertEqual(results[0].status.iloc[0], "UNDEFINED_SINGLE_FOV")

    def test_case_medians_and_ties(self):
        data = pd.DataFrame({
            "Patient_ID":list("abcdefghi"), "LNM":[1,1,1,1,0,0,0,0,0],
            "probability":[.7,.8,.9,.1,.1,.2,.7,.8,.9],
            "anonymous_case_id":["p3","p2","p1","fn","tn2","tn1","fp1","fp2","fp3"],
        })
        selected = choose_cases(data, .5).set_index("category")
        self.assertEqual(selected.loc["TP", "Patient_ID"], "b")
        self.assertEqual(selected.loc["FN", "Patient_ID"], "d")
        self.assertEqual(selected.loc["TN", "Patient_ID"], "f")
        self.assertEqual(selected.loc["FP", "Patient_ID"], "h")

    def test_sampler_large_coordinate_distances(self):
        points = np.asarray([[0,0],[100000,0],[0,100000],[100000,100000]], dtype="int32")
        selected = spatial_sample(points, 3)
        self.assertEqual(selected.dtype, np.int64)
        self.assertEqual(len(np.unique(selected, axis=0)), 3)

    def test_anisotropic_physical_fov_geometry(self):
        points, info = candidate_fovs((20000, 12000), (.25, .5), np.ones((120, 200), bool), (500, 500), 32)
        self.assertEqual(len(points), 32)
        self.assertEqual(info["FOV_native_width"], 2000)
        self.assertEqual(info["FOV_native_height"], 1000)

    def test_weights_give_each_patient_equal_total_weight(self):
        ids = np.asarray(["a","a","b"])
        weights = patient_weights(ids)
        self.assertEqual(weights.mean(), 1)
        self.assertEqual(weights[ids == "a"].sum(), weights[ids == "b"].sum())

    def test_bridge_shape_and_output_scale(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(18, 8))
        raw = rng.poisson(15, size=(18, 6))
        y = np.log1p(raw/raw.sum(1)[:, None]*1e6)
        negative = np.zeros((18, 3), dtype=int)
        ids = np.repeat(np.arange(9), 2)
        model = RidgeBridge().fit(x, y, raw, negative, ids, .1)
        predicted = model.predict(x)
        self.assertEqual(predicted.shape, y.shape)
        self.assertEqual(predicted.dtype, np.float32)
        self.assertTrue(np.isfinite(predicted).all())

    def test_bridge_training_entry_point(self):
        rng = np.random.default_rng(3)
        raw = rng.poisson(20, size=(36, 6))
        ids = np.repeat(np.arange(18).astype(str), 2)
        folds = np.repeat(np.arange(18)%3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"fov.npz"
            np.savez(path, embeddings=rng.normal(size=(36, 5)), log_expression=np.log1p(raw/raw.sum(1)[:,None]*1e6),
                     counts=raw, negative_counts=np.zeros((36,3)), Patient_ID=ids, outer_fold=folds,
                     gene_names=np.asarray(["g"+str(i) for i in range(6)]))
            train_bridge(path, Path(tmp)/"bridge", alphas=(.1,))
            self.assertTrue((Path(tmp)/"bridge/bridge.joblib").exists())
            self.assertEqual(len(json.loads((Path(tmp)/"bridge/bridge_cv.json").read_text())["folds"]), 3)

    def test_patient_join_is_keyed_and_rejects_duplicates(self):
        scores = pd.DataFrame({"Patient_ID":["a","b"], "probability":[.9,.1]})
        labels = pd.DataFrame({"Patient_ID":["b","a"], "LNM":[0,1]})
        self.assertEqual(paired_inputs(scores, labels).LNM.tolist(), [1,0])
        with self.assertRaises(ValueError):
            paired_inputs(pd.concat([scores, scores.iloc[:1]]), labels)

    def test_confusion_and_empty_low_risk_group(self):
        result = metrics([1,1,0,0], [.9,.2,.8,.1], .5)
        self.assertEqual([result[k] for k in ("TP","FN","TN","FP")], [1,1,1,1])
        self.assertIsNone(metrics([1,0], [.9,.8], 0)["NPV"])

    def test_cutoff_and_tie_prefer_lower_feature_count(self):
        self.assertEqual(select_threshold([1,1,1,0], [.2,.3,.8,.7], .95), .2)
        a = {"classifier":"Logistic Regression", "inner_mean_AUROC":.8, "inner_mean_AUPRC":.8,
             "Brier":.2, "inner_SD_AUROC":.05, "recipe":{"selector":"PLS-DA","count":5,"config":{}}}
        b = {**a, "recipe":{**a["recipe"],"count":100}}
        self.assertLess(selection_order(a), selection_order(b))

    def test_all_classifier_factorial_budgets(self):
        self.assertEqual(len(CLASSIFIERS), 12)
        self.assertEqual(len(SELECTORS), 9)
        for name in CLASSIFIERS:
            recipes = coarse(name)
            self.assertEqual(len(recipes), 162)
            self.assertEqual({r["count"] for r in recipes}, set(COUNTS))
            self.assertTrue(hasattr(build(name, recipes[0]["config"], 7), "fit"))

    def test_search_completes_all_passes_and_selects_by_inner_scores(self):
        from types import SimpleNamespace
        data = SimpleNamespace(y=np.tile([0, 1], 18), genes=np.asarray([f"g{i}" for i in range(100)]))
        context = {"name":"full", "inner_split_seed":7, "final_estimator_seed":9}
        settings = {"refinement_candidates":{"full":{"Logistic Regression":[{"C":.5,"penalty":"l2"}]*24}}}
        def trial(data, train, name, recipe, context):
            return {"classifier":name,"recipe":recipe,"candidate_id":str(recipe),
                    "inner_mean_AUROC":.6+recipe["count"]/1000,"inner_mean_AUPRC":.7,
                    "Brier":.2,"inner_SD_AUROC":.05}
        with tempfile.TemporaryDirectory() as tmp, patch("escc_lnm.training.evaluate_trial", side_effect=trial), \
                patch("escc_lnm.training.calibrated_fit", return_value=(classifier(), np.linspace(.1,.9,36))):
            selected, _ = search_classifier(data, np.arange(36), "Logistic Regression", context, settings, tmp)
            records = json.loads((Path(tmp)/"search.json").read_text())
            self.assertEqual(selected["attempts"], 192)
            self.assertEqual(selected["recipe"]["count"], 100)
            self.assertEqual([sum(r["search_pass"] == p for r in records) for p in
                              ("factorial","refinement","matched_count")], [162,24,6])

    def test_calibration_fit_excludes_outer_holdout(self):
        from escc_lnm.training import split
        class Data:
            def __init__(self):
                self.x = np.random.default_rng(5).normal(size=(32, 5))
                self.y = np.tile([0,1], 16)
                self.genes = np.asarray(["g"+str(i) for i in range(5)])
                self.fits = []
            def transform(self, indices, recipe):
                self.fits.append(set(indices))
                f = RNAFeatures([0,1]).fit(self.x[indices])
                return f, f.transform(self.x)
        data = Data()
        train = np.arange(24)
        model, p = calibrated_fit(data, train, {"config":{"C":1.,"penalty":"l2"}}, "Logistic Regression",
                                  7, split(data.y[train], 3, 9))
        self.assertTrue(all(indices <= set(train) for indices in data.fits))
        self.assertEqual(len(p), len(train))
        self.assertEqual(model.predict_proba(data.x[24:]).shape, (8,2))


if __name__ == "__main__":
    unittest.main()
