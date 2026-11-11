# ESCC LNM prediction from whole-slide histology

This package predicts lymph node metastasis in esophageal squamous cell carcinoma from H&E whole-slide images. A Phikon encoder and CosMx-supervised Ridge bridge generate image-inferred RNA features. Patient-level feature selection and machine learning produce the LNM score.

The package includes image sampling, RNA bridge training, nine feature selectors, twelve classifiers, nested cross-validation, model evaluation and spatial risk attribution.

## Installation

Use Python 3.12 and a CUDA-enabled PyTorch installation for image encoding.

    pip install -e .
    escc-lnm --help

## Workflow

### 1. Train the RNA bridge

Prepare an NPZ file containing aligned FOV arrays.

| Array | Contents |
|---|---|
| embeddings | Phikon embeddings, FOV × 768 |
| log_expression | Measured log1p(CPM), FOV × gene |
| counts, negative_counts | RNA and negative-probe counts |
| Patient_ID, outer_fold | Patient identifiers and fixed fold assignments |
| gene_names | Ordered gene names |
| primary | Optional boolean indicator for primary-tissue evaluation |

    escc-lnm train-bridge --input inputs/cosmx_fovs.npz --output models/bridge

### 2. Extract patient RNA features

The slide manifest is a CSV with Patient_ID, slide_id and path. Supply the local Phikon directory and fitted Ridge bridge. The ordered gene list can be supplied as a JSON array or an object containing gene_names.

    escc-lnm extract --slides inputs/slides.csv --encoder models/phikon --bridge models/bridge/bridge.joblib --output output/features

Sampling uses physical FOVs of approximately 512 µm and tissue-positive 224-pixel tiles. FOV RNA is averaged within each slide, followed by an equal-slide patient mean. The command writes patient_rna.parquet and a slides.csv file linking each slide to its FOV features and coordinates.

### 3. Train and compare classifiers

The patient manifest contains Patient_ID and binary LNM. Supply outer_fold values 0–4 to use fixed patient partitions. The feature table has one patient per row and RNA gene names as columns.

    escc-lnm train --features output/features/patient_rna.parquet --patients inputs/patients.csv --output output/training

Nine selectors and feature counts 5, 10, 20, 30, 50 and 100 are compared for all twelve classifiers. Each classifier receives 162 factorial, 24 refinement and six matched-count attempts within each training context. Mean inner AUROC determines selection, followed by AUPRC, Brier score, AUROC SD, feature count and deterministic ties. Feature ranking, scaling and probability calibration are fitted inside the corresponding training partitions.

Outputs include classifier.joblib, model.json, patient-level OOF scores, classifier rankings and feature-count comparisons.

### 4. Predict and evaluate

    escc-lnm predict --features output/features/patient_rna.parquet --classifier output/training/classifier.joblib --model-info output/training/model.json --output output/predictions.csv
    escc-lnm evaluate --scores output/predictions.csv --labels inputs/outcomes.csv --threshold 0.36731575640740904 --output output/evaluation

The outcome CSV contains Patient_ID and LNM. Evaluation returns AUROC, AUPRC, classification metrics, confidence intervals, calibration and decision-curve data. Use the working threshold saved with the selected model.

To prepare an existing fitted classifier for the prediction command, supply its checkpoint, ordered gene names and a JSON mapping of working-point names to thresholds.

    escc-lnm prepare-model --classifier models/classifier.joblib --genes inputs/genes.json --thresholds inputs/thresholds.json --output models/predictor

### 5. Generate H&E risk attribution

    escc-lnm explain --slides output/features/slides.csv --labels inputs/outcomes.csv --classifier output/training/classifier.joblib --model-info output/training/model.json --threshold 0.36731575640740904 --output output/attribution
    escc-lnm figures --kind attribution --input output/attribution

Representative TP, FN, TN and FP cases are selected by proximity to their group median score. Ties use an anonymous identifier. The main slide has the largest tissue-support area. Each FOV is removed in turn and the patient score is recomputed. Positive contributions appear red and negative contributions blue, using one shared scale. Each case receives independent H&E, FOV-location, contribution and native-resolution detail panels.

### 6. Export analysis figures

    escc-lnm figures --kind classifiers --input output/training --output output/classifier_figures
    escc-lnm figures --kind evaluation --input output/evaluation --scores output/predictions.csv --labels inputs/outcomes.csv --output output/evaluation_figures

Figures are exported as editable PDF and SVG, with 600 dpi PNG previews.

## Package structure

| Module | Function |
|---|---|
| sampling, inference | Physical FOV sampling, Phikon encoding and patient prediction |
| bridge | Patient-weighted image-to-RNA Ridge regression |
| selection, search_space, training | RNA selection and nested classifier optimization |
| models | Feature scaling, calibrated classifiers and checkpoint loading |
| evaluation | Patient-level metrics, intervals, calibration and DCA |
| attribution, plots | FOV contributions and publication figures |
| config | Sampling settings and reproducible search parameters |

## Tests

    python -m unittest discover -s tests -v
