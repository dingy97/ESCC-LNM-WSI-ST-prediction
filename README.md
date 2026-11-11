# ESCC LNM prediction from H&E with a CosMx-supervised RNA bridge

Manuscript analysis source for the frozen Phikon → Ridge → image-inferred RNA → PLS-DA Top100 → logistic regression workflow.

The source revision follows the audited Stage21 clinical-LNM development branch and Stage24 local attribution analysis. This commit replaces the earlier ResNet50/Image2Gene implementation. The parent Git commit retains that implementation.

## Repository scope

Source manifests map each analysis file to its original and release SHA256. Local filesystem constants serve as installation tokens. Stage14 retains the rankers and calibrated model definitions consumed by Stage21, with legacy experimental entry points removed. Historical discrepancy sensitivity is controlled via private manifest flags instead of hard-coded accession identifiers.

## Analysis workflow

The pipeline extracts patch embeddings from H&E WSI tiles using a frozen Phikon encoder, aggregates features within sampled FOVs, and maps visual embeddings to inferred RNA profiles via a CosMx-supervised Ridge bridge. Aggregated patient-level inferred RNA vectors are subsetted to the PLS-DA top 100 genes, standardized, and fed into a L2-regularized logistic regression classifier with cross-fitted sigmoid calibration for patient-level LNM prediction. Model performance is evaluated against clinical outcome labels.

## Source layout

- `analysis/03_bridge_training`: patient-grouped bridge regression, control experiments, evaluation and plotting routines
- `analysis/10_lnm_sampling_corrected/scripts`: FOV sampler and shared utilities
- `analysis/14_RNA_LNM_classifier_optimization/scripts/rna_experiment.py`: RNA ranking, scaling and calibration definitions used in Stage21
- `analysis/21_surgical_RNA_LNM_development/clinical_LNM_v2`: 12-classifier search, model selection ordering and internal analyses
- `analysis/23_manuscript_figures`: manuscript panel functions and ESD outcome-direction validation
- `analysis/SCI论文最终组图素材`: Stage24 attribution, figure generation and feature-count audit scripts
- `scripts`: installation, frozen inference, fixed prediction evaluation and corrected-ESD rendering entrypoints

## Environment

Archived analyses were run under Python 3.12. Dependencies are listed in `requirements.txt`. PyTorch and torchvision must match the CUDA build used for frozen feature extraction (CUDA 12.4). The optional Illustrator workflow requires `pywin32` and a local Adobe Illustrator installation.

Frozen Phikon weights and private project checkpoints are required at runtime. Third-party weights and proprietary runtime directories are not distributed here.

## Install analysis source

Prepare a private JSON path config following `config/paths.example.json` with absolute paths to workspace, image assets and encoder resources.

```
python scripts/install_sources.py --paths /private/paths.json --dry-run
python scripts/install_sources.py --paths /private/paths.json
```

The installer validates all target files before writing. Use `--overwrite-code` to replace existing source files when needed. Retain matching input and configuration revisions for reproducibility. Input schemas are documented in `docs/private_inputs.md`. Private manifests, model locks and fitted preprocessing artifacts are supplied separately.

## Frozen prediction from cached full-panel RNA

```
python scripts/predict_cached.py --project-root /private/CosMx_HE_Bridge_v3 --features /private/patient_inferred_RNA.parquet --output /private/output/frozen_predictions.csv
```

The entrypoint validates checkpoint SHA256 and gene panel schema before running the fixed classifier. Load joblib checkpoints only from trusted sources.

## Fixed-prediction evaluation

```
python scripts/evaluate_predictions.py --scores /private/output/frozen_predictions.csv --labels /private/confirmed_outcomes.csv --threshold 0.36731575640740904 --output /private/output/evaluation
```

The label file contains case-level LNM status. For ESD label correction, case labels are updated after grouping while predicted probabilities and thresholds remain unchanged. Evaluation performs an exact patient-key join using the supplied labels.

To reproduce archived bootstrap intervals under corrected outcomes:

```
--bootstrap-reference-labels /private/original_outcomes.csv
```

This preserves the original stratified sampling order; all metrics are computed using the updated labels provided via `--labels`.

## Twelve classifiers and nine selectors

Each classifier is evaluated across a factorial design of nine selectors, six feature counts and three classifier parameter profiles. Each configuration receives 24 refinement runs and six matched-count runs, yielding 192 trials per context. Duplicated configurations reuse prior trial records.

Primary model ordering: mean inner-fold AUROC → mean inner AUPRC → pooled inner Brier score → inner AUROC SD → feature count → deterministic hash. Training-only preprocessing, ranking, calibration and threshold selection are refitted within partitions. The full search is repeated on every outer training split. Separate OOF estimates are generated for the selection pipeline and the final classifier family.

```
python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage train12
python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage compare12
```

Search configuration records algorithm names, budgets, parameter profiles, precomputed draws, seeds and ranking rules. Patient partitions and input hashes must be reconstructed locally. Sample overlap between bridge supervision and training cohorts should be considered when interpreting internal and external results.

## ESD figures and attribution

```
python scripts/render_corrected_esd.py --project-root /private/CosMx_HE_Bridge_v3 --output /private/output/corrected_ESD --confirm-corrected-esd-labels
python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage attribution
python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage attribution-figures
```

Stage24 selects TP, TN and FP samples by distance to group median frozen probability with a fixed tie-breaking rule for anonymous IDs, and includes the single FN. Main-slide selection is based on tissue-support area. Attribution is implemented by sequential leave-one-FOV removal and recomputation of slide-to-patient aggregation.

Attribution values are defined as the difference between full-model and leave-one-FOV uncalibrated LR log-odds. The output CSV also tracks calibrated probability shifts and float32 aggregation rounding error. All visualizations use a symmetric colour scale; only sampled footprints overlapping the tissue mask are rendered. H&E crops are selected by maximum absolute attribution within the slide.
