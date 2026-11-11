"""Source and synthetic numerical checks. No clinical data fixtures."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class ReleaseTests(unittest.TestCase):
    def test_code_only(self):
        self.assertGreater(len(module("scripts/check_code_only.py").check(ROOT)), 40)

    def test_windows_path_rendering(self):
        installer = module("scripts/install_sources.py")
        text = "ROOT=Path(r'__ESCC_PROJECT_ROOT__')"
        rendered = installer.render_source(text, {"__ESCC_PROJECT_ROOT__": "D:/private/project"})
        self.assertEqual(rendered, "ROOT=Path(r'D:/private/project')")

    def test_install_rejects_conflicting_code_before_writes(self):
        installer = module("scripts/install_sources.py")
        paths = {key: "/private/location" for key in [
            "__ESCC_PROJECT_ROOT__", "__ESCC_LEGACY_ROOT__", "__ESCC_SURGICAL_WSI_ROOT__",
            "__ESCC_ESD_WSI_ROOT__", "__ESCC_TCGA_WSI_ROOT__", "__ESCC_USER_HOME__",
            "__ESCC_COSMX_HE_WSI__", "__ESCC_RAW_COSMX_ROOT__",
        ]}
        first = sorted((ROOT/"analysis").rglob("*.py"))[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root/first.relative_to(ROOT/"analysis")
            target.parent.mkdir(parents=True)
            target.write_text("# preexisting code", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                installer.install(root, paths)
            self.assertEqual(target.read_text(encoding="utf-8"), "# preexisting code")

    def test_exact_patient_key_join(self):
        ev = module("scripts/evaluate_predictions.py")
        scores = pd.DataFrame({"Patient_ID": ["a", "b"], "probability": [.9, .1]})
        labels = pd.DataFrame({"Patient_ID": ["b", "a"], "LNM": [0, 1]})
        result = ev.paired_inputs(scores, labels)
        self.assertEqual(result.LNM.tolist(), [1, 0])
        with self.assertRaises(ValueError):
            ev.paired_inputs(pd.concat([scores, scores.iloc[:1]]), labels)
        with self.assertRaises(ValueError):
            ev.paired_inputs(scores.iloc[:1], labels)

    def test_confusion_and_undefined_low_risk(self):
        ev = module("scripts/evaluate_predictions.py")
        result = ev.metrics([1, 1, 0, 0], [.9, .2, .8, .1], .5)
        self.assertEqual([result[x] for x in ["TP", "FN", "TN", "FP"]], [1, 1, 1, 1])
        self.assertEqual(result["Accuracy"], .5)
        self.assertEqual(result["residual_LNM_in_low_risk"], .5)
        result = ev.metrics([1, 0], [.9, .8], 0)
        self.assertIsNone(result["NPV"])

    def test_sampler_uses_64bit_distances(self):
        sampler = module("analysis/10_lnm_sampling_corrected/scripts/spatial_sampling.py")
        points = np.asarray([[0, 0], [100_000, 0], [0, 100_000], [100_000, 100_000]], dtype=np.int32)
        chosen = sampler.spatial_sample(points, 3)
        self.assertEqual(chosen.dtype, np.int64)
        self.assertEqual(len(np.unique(chosen, axis=0)), 3)
        np.testing.assert_array_equal(chosen, sampler.spatial_sample(points, 3))

    def test_feature_count_and_search_design(self):
        import json
        design = json.loads((ROOT/"config/search_design.json").read_text(encoding="utf-8"))
        self.assertEqual(len(design["classifiers"]), 12)
        self.assertEqual(len(design["selectors"]), 9)
        self.assertEqual(design["counts"], [5, 10, 20, 30, 50, 100])
        self.assertEqual(design["selection_rule"][0], "maximum mean inner-fold AUROC")


if __name__ == "__main__":
    unittest.main()
