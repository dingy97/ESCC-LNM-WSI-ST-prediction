"""Representative cases and leave-one-FOV-out risk contributions."""
from contextlib import closing
from decimal import Decimal
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .io import read_json, sha256, write_json
from .models import load_model


def anonymous_id(value, prefix):
    return prefix + hashlib.sha256(str(value).encode()).hexdigest()[:10].upper()


def choose_cases(frame, threshold):
    if frame.Patient_ID.duplicated().any() or set(frame.LNM) != {0, 1}:
        raise ValueError("Require unique patients and binary LNM labels")
    selected = []
    rows = frame.copy()
    if "anonymous_case_id" not in rows:
        rows["anonymous_case_id"] = rows.Patient_ID.map(lambda x:anonymous_id(x, "ESD-"))
    high = rows.probability >= threshold
    rows["category"] = np.where(rows.LNM == 1, np.where(high, "TP", "FN"), np.where(high, "FP", "TN"))
    for category in ("TP", "FN", "TN", "FP"):
        subset = rows[rows.category.eq(category)].copy()
        if subset.empty:
            raise ValueError(f"No {category} patient is available")
        values = sorted(Decimal(str(x)) for x in subset.probability)
        median = (values[(len(values)-1)//2]+values[len(values)//2])/2
        subset["median_distance"] = subset.probability.map(lambda x:abs(Decimal(str(x))-median))
        selected.append(subset.sort_values(["median_distance", "anonymous_case_id"]).iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def leave_one_out(model, slide_rna):
    means = np.asarray([rna.mean(0) for rna in slide_rna])
    full = means.mean(0)
    margin = float(model.raw_margin(full.astype(float).reshape(1, -1))[0])
    probability = float(model.predict_proba(full.astype(float).reshape(1, -1))[0, 1])
    results = []
    for j, rna in enumerate(slide_rna):
        if len(rna) == 1:
            results.append(pd.DataFrame({
                "FOV_ordinal":[1], "full_raw_margin":[margin],
                "without_raw_margin":[np.nan], "delta_raw_margin":[np.nan],
                "full_probability":[probability], "without_probability":[np.nan],
                "delta_probability":[np.nan], "status":["UNDEFINED_SINGLE_FOV"],
            }))
            continue
        perturbed = []
        for k in range(len(rna)):
            current = means.copy()
            current[j] = np.delete(rna, k, axis=0).mean(0)
            perturbed.append(current.mean(0))
        perturbed = np.asarray(perturbed)
        removed_margin = model.raw_margin(perturbed.astype(float))
        removed_probability = model.predict_proba(perturbed.astype(float))[:, 1]
        delta = margin-removed_margin
        coefficient = model.estimator.coef_[0]/model.features.scaler.scale_
        actual_difference = full[model.features.genes].astype(float)-perturbed[:, model.features.genes].astype(float)
        np.testing.assert_allclose(delta, actual_difference@coefficient, rtol=0, atol=1e-10)
        r64 = rna[:, model.features.genes].astype(float)
        ideal_difference = (r64-r64.mean(0))/((len(rna)-1)*len(slide_rna))
        ideal_delta = ideal_difference@coefficient
        results.append(pd.DataFrame({
            "FOV_ordinal":np.arange(1, len(rna)+1), "full_raw_margin":margin,
            "without_raw_margin":removed_margin, "delta_raw_margin":delta,
            "full_probability":probability, "without_probability":removed_probability,
            "delta_probability":probability-removed_probability,
            "ideal_float64_delta":ideal_delta, "aggregation_roundoff":delta-ideal_delta, "status":"PASS",
        }))
    return full, probability, results


def explain(manifest_path, labels_path, classifier_path, model_info, threshold, output):
    import openslide
    model = load_model(classifier_path)
    info = read_json(model_info)
    if sha256(classifier_path) != info["model_sha256"]:
        raise ValueError("Classifier checksum mismatch")
    if not hasattr(model.estimator, "coef_"):
        raise ValueError("Log-odds attribution requires the frozen linear classifier")
    manifest = pd.read_csv(manifest_path, dtype={"Patient_ID":str, "slide_id":str})
    if "status" in manifest:
        manifest = manifest[manifest.status.eq("PASS")].copy()
    labels = pd.read_csv(labels_path, dtype={"Patient_ID":str}).set_index("Patient_ID")
    if not labels.index.is_unique or manifest.slide_id.duplicated().any():
        raise ValueError("Patient labels and slide identifiers must be unique")
    if set(manifest.Patient_ID) != set(labels.index):
        raise ValueError("WSI manifest and labels have different patient cohorts")
    patients, caches = [], {}
    for patient, subset in manifest.groupby("Patient_ID", sort=True):
        slide_rows = []
        for row in subset.itertuples():
            identity = read_json(Path(row.cache).with_suffix(".json"))["identity"]
            stat = Path(row.path).stat()
            if (identity["path"] != str(Path(row.path).resolve()) or identity["source_bytes"] != stat.st_size
                    or identity["source_mtime_ns"] != str(stat.st_mtime_ns)):
                raise ValueError("WSI identity differs from its FOV cache")
            with np.load(row.cache, allow_pickle=False) as saved:
                content = {key:saved[key] for key in saved.files}
            if list(content["gene_names"].astype(str)) != info["gene_names"]:
                raise ValueError("FOV gene schema differs from the classifier")
            if not np.isfinite(content["RNA"]).all() or not len(content["RNA"]):
                raise ValueError("FOV feature cache is empty or nonfinite")
            content["source_path"] = row.path
            content["slide_id"] = row.slide_id
            content["anonymous_slide_id"] = anonymous_id(row.slide_id, "WSI-")
            slide_rows.append(content)
        values = np.asarray([s["RNA"].mean(0) for s in slide_rows]).mean(0)
        probability = float(model.predict_proba(values.astype(float).reshape(1, -1))[0, 1])
        case = {"Patient_ID":patient, "LNM":int(labels.loc[patient, "LNM"]), "probability":probability}
        if "anonymous_case_id" in labels:
            case["anonymous_case_id"] = labels.loc[patient, "anonymous_case_id"]
        patients.append(case)
        caches[patient] = slide_rows
    cases = choose_cases(pd.DataFrame(patients), threshold)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Select the main WSI before any perturbation results are inspected.
    cases["main_slide"] = cases.Patient_ID.map(lambda key:min(
        caches[key], key=lambda s:(-float(s["tissue_area_mm2"]), s["anonymous_slide_id"])
    )["anonymous_slide_id"])
    cases["WSI_count"] = cases.Patient_ID.map(lambda key:len(caches[key]))
    cases["FOV_count"] = cases.Patient_ID.map(lambda key:sum(len(s["RNA"]) for s in caches[key]))
    cases.drop(columns=["Patient_ID", "median_distance"]).to_csv(output/"cases.csv", index=False, float_format="%.17g")
    rows, slide_summary, links = [], [], []
    private = output/"private"
    private.mkdir(exist_ok=True)
    for case in cases.itertuples():
        slides = caches[case.Patient_ID]
        _, probability, contributions = leave_one_out(model, [s["RNA"] for s in slides])
        if abs(probability-case.probability) > 1e-12:
            raise ValueError("Full patient probability replay failed")
        for content, values in zip(slides, contributions):
            coords = content["coords_level0"]
            mpp = content["mpp"]
            dims = content["dimensions"]
            width, height = np.rint(content["FOV_um"]/mpp).astype(int)
            if len(coords) != len(values) or (coords < 0).any() or (coords+[width, height] > dims).any():
                raise ValueError("Invalid FOV-to-WSI geometry")
            with closing(openslide.OpenSlide(content["source_path"])) as slide:
                if not np.array_equal(slide.dimensions, dims):
                    raise ValueError("WSI dimensions differ from cached geometry")
                actual_mpp = [float(slide.properties[key]) for key in
                              (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y)]
                np.testing.assert_allclose(actual_mpp, mpp, rtol=0, atol=1e-12)
            values = values.assign(
                category=case.category, anonymous_case_id=case.anonymous_case_id,
                anonymous_slide_id=content["anonymous_slide_id"],
                main_slide=content["anonymous_slide_id"] == case.main_slide,
                x_level0=coords[:, 0], y_level0=coords[:, 1], width_level0=width, height_level0=height,
                retained_tiles=(content["patch_tissue_fractions"] >= .05).sum(1),
            )
            rows.append(values)
            links.append({"anonymous_slide_id":content["anonymous_slide_id"], "Patient_ID":case.Patient_ID,
                          "source_path":content["source_path"],
                          "cache":manifest.loc[manifest.slide_id.eq(content["slide_id"]), "cache"].iloc[0]})
            slide_summary.append({"category":case.category, "anonymous_case_id":case.anonymous_case_id,
                "anonymous_slide_id":content["anonymous_slide_id"], "FOV_count":len(values),
                "tissue_area_mm2":float(content["tissue_area_mm2"]), "mpp_x":float(mpp[0]), "mpp_y":float(mpp[1]),
                "main_slide":content["anonymous_slide_id"] == case.main_slide})
    all_values = pd.concat(rows, ignore_index=True)
    all_values.to_csv(output/"fov_attribution.csv", index=False, float_format="%.17g")
    pd.DataFrame(slide_summary).to_csv(output/"slides.csv", index=False, float_format="%.17g")
    pd.DataFrame(links).to_csv(private/"source_mapping.csv", index=False)
    write_json(output/"attribution.json", {
        "threshold":threshold, "model_sha256":sha256(classifier_path),
        "unit":"Full minus leave-one-FOV-out uncalibrated LR log-odds",
        "global_color_limit":float(np.ceil(all_values.delta_raw_margin.abs().max()*1000)/1000),
    })
    return all_values
