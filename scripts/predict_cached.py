"""Apply the frozen private classifier to full-panel patient RNA features."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def configure_imports(project):
    for relative in [
        "03_bridge_training/scripts", "10_lnm_sampling_corrected/scripts",
        "12_stage5_high_sensitivity/scripts", "14_RNA_LNM_classifier_optimization/scripts",
        "08_lnm_modeling/runtime_deps",
    ]:
        sys.path.insert(0, str(project / relative))


def predict(project, features_path, output_path):
    configure_imports(project)
    import joblib
    import numpy as np
    import pandas as pd
    stage = project / "21_surgical_RNA_LNM_development/clinical_LNM_v2"
    lock = json.loads((stage / "config/final_model_lock.json").read_text(encoding="utf-8"))
    checkpoint = Path(lock["winner_directory"]) / "model.joblib"
    if not checkpoint.exists():
        checkpoint = stage / "02_training/full_development/outer_-1/Logistic_Regression/model.joblib"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != lock["model_SHA256"]:
        raise ValueError("The private classifier does not match its frozen lock")
    if lock["winner_classifier"] != "Logistic Regression":
        raise ValueError("Expected the Stage21 LR candidate")
    model = joblib.load(checkpoint)
    genes = json.loads((project / "03_bridge_training/bridge_model_checkpoints/broad_transcriptome/bundle_card.json").read_text(encoding="utf-8"))["gene_names"]
    frame = pd.read_parquet(features_path)
    if "Patient_ID" in frame.columns:
        frame = frame.set_index("Patient_ID")
    if not frame.index.is_unique or set(frame.columns) != set(genes):
        raise ValueError("Require one row per patient and the exact 6,175-gene schema")
    matrix = frame.loc[:, genes].to_numpy(float)
    if len(frame) == 0 or not np.isfinite(matrix).all():
        raise ValueError("Require finite patient features")
    result = pd.DataFrame({
        "Patient_ID": frame.index.astype(str),
        "probability": model.predict_proba(matrix)[:, 1],
        "model_SHA256": digest,
    })
    for name, cutoff in lock["winner"]["thresholds"].items():
        result[f"high_risk_{name}"] = result.probability >= cutoff
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.17g")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args.project_root.resolve(), args.features, args.output)
    print(f"Scored {len(result)} patient rows with the frozen classifier")


if __name__ == "__main__":
    main()
