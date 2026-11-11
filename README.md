# ESCC LNM prediction from H&E with a CosMx-supervised RNA bridge

Manuscript analysis source for the frozen Phikon → Ridge → image-inferred RNA → PLS-DA Top100 → logistic regression workflow.

The source revision follows the audited Stage21 clinical-LNM development branch and Stage24 local attribution analysis. This commit replaces the earlier ResNet50/Image2Gene implementation. The parent Git commit retains that implementation.

## Repository scope

This is a code-only repository. Clinical spreadsheets, accession mappings, patient outcomes, fold membership, WSI images, caches, fitted models, prediction tables, search results and manuscript figures belong in a separate private workspace.

The source manifest maps every analysis source file to its original and release SHA256. Local filesystem constants are installation tokens. The Stage14 dependency retains the rankers and calibrated model definitions used by Stage21. Its earlier experiment entry points were omitted. The historical discrepancy sensitivity uses a private manifest flag instead of embedded accession numbers.

## Analysis workflow

1. Sample virtual FOVs on the original Level0 WSI using MPP and the frozen tissue-support mask.
2. Read each physical FOV on the inherited 1024-pixel grid. Encode eligible 224-pixel tiles using frozen Phikon CLS features.
3. Average tile embeddings within each FOV and apply the frozen CosMx-supervised Ridge bridge.
4. Average FOV RNA vectors within each WSI, then average WSI vectors within each patient or pathology accession group. Preserve the original float32 aggregation order.
5. Select the frozen PLS-DA-ranked 100 columns from the complete 6,175-gene panel. Apply the stored StandardScaler, L2 logistic regression and training-crossfitted sigmoid calibrator.
6. Evaluate frozen patient predictions against an explicitly supplied outcome table.

The image-inferred RNA representation has heterogeneous gene-specific accuracy. The bridge and LNM evaluations use different outcomes and statistical units.

## Source layout

- analysis/03_bridge_training contains patient-grouped bridge regression, controls, evaluation and figure code.
- analysis/10_lnm_sampling_corrected/scripts supplies the physical sampler and shared dependencies.
- analysis/14_RNA_LNM_classifier_optimization/scripts/rna_experiment.py supplies the original RNA ranking, scaling and calibration definitions used by Stage21.
- analysis/21_surgical_RNA_LNM_development/clinical_LNM_v2 contains the complete twelve-classifier search, model-selection ordering and internal analyses.
- analysis/23_manuscript_figures contains manuscript panel functions and both explicitly labelled ESD outcome-direction checks.
- analysis/SCI论文最终组图素材 contains Stage24 attribution/figure/QA source and the matched feature-count audit.
- scripts provides installation, frozen cached inference, fixed-prediction evaluation and corrected-ESD rendering entry points.

## Environment

Python 3.12 was used for the archived analyses. Install the dependencies in requirements.txt. PyTorch and torchvision should use the same CUDA build as the frozen image extraction environment. The recorded build was CUDA 12.4. The optional Windows Illustrator check requires pywin32 and an installed Adobe Illustrator.

The analysis expects user-supplied local Phikon weights and trusted private project checkpoints. Third-party weights and the original project runtime dependency directories are not distributed here.

## Install analysis source

Create a private paths JSON following config/paths.example.json, using absolute paths for your workspace, images and encoder resources.

    python scripts/install_sources.py --paths /private/paths.json --dry-run
    python scripts/install_sources.py --paths /private/paths.json

The installer validates the complete write set before writing. A differing existing source file requires the explicit --overwrite-code option. Retain the corresponding input/configuration revision when reproducing an archived run.

The archived protocol consumes an audited private project tree. The required input schemas are described in docs/private_inputs.md. Private manifests, model locks and fitted preprocessing must be supplied by the data custodian.

## Frozen prediction from cached full-panel RNA

    python scripts/predict_cached.py --project-root /private/CosMx_HE_Bridge_v3 --features /private/patient_inferred_RNA.parquet --output /private/output/frozen_predictions.csv

The entry point verifies the checkpoint SHA256 and full gene schema, then applies the existing classifier. Load joblib checkpoints only from a trusted source.

## Fixed-prediction evaluation

    python scripts/evaluate_predictions.py --scores /private/output/frozen_predictions.csv --labels /private/confirmed_outcomes.csv --threshold 0.36731575640740904 --output /private/output/evaluation

The outcome file must contain the confirmed case-level LNM direction. In the study ESD correction, the previously fixed case labels were complemented after case grouping, while probabilities and thresholds were preserved. The evaluation entry point performs an exact patient-key join and uses the supplied labels.

To replay the archived ESD bootstrap intervals exactly after outcome correction, additionally pass --bootstrap-reference-labels /private/original_outcomes.csv. This preserves the archived draw order within the same two strata. All metrics still use the confirmed outcomes supplied with --labels.

## Twelve classifiers and nine selectors

Every classifier receives the same complete Stage-A factorial budget of nine selectors, six feature counts and three classifier-specific parameter profiles. Every classifier then receives 24 refinement attempts and six matched-count attempts, totaling 192 attempts per context. Duplicate recipes reuse their trial records.

The primary ordering is mean inner-fold AUROC, followed by mean inner AUPRC, pooled inner Brier, inner AUROC SD, feature count and a deterministic hash. Training-only preprocessing, ranking, calibration and threshold selection are refitted within the relevant partitions. The complete search is repeated in each outer training portion. The full selection process and the selected classifier family have separate OOF estimates.

    python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage train12
    python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage compare12

The search design configuration records the algorithm names, counts, parameter profiles, precomputed refinement draws, seeds and selection ordering. Private patient partitions and input hashes must be reconstructed locally. Historical exposures and bridge-supervision overlap must be retained in the interpretation of internal and external evaluation.

## ESD figures and attribution

    python scripts/render_corrected_esd.py --project-root /private/CosMx_HE_Bridge_v3 --output /private/output/corrected_ESD --confirm-corrected-esd-labels
    python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage attribution
    python scripts/run_stage.py --project-root /private/CosMx_HE_Bridge_v3 --stage attribution-figures

Stage24 selects TP, TN and FP by distance to their group median frozen probability, with a fixed anonymous-ID tie rule, and includes the sole FN. Main-slide selection uses frozen tissue-support area. Attribution removes one actual sampled FOV at a time and recomputes the original slide-to-patient aggregation.

Attribution is the full minus leave-one-FOV-out uncalibrated LR log-odds. The CSV also records calibrated probability changes and float32 aggregation roundoff. All examples share a symmetric color range. Only sampled footprints intersecting the tissue-support mask receive color. Native H&E detail is selected by the largest absolute attribution on the fixed main slide.

Figure3 source represents the historical matched visual/RNA/fusion comparison under the earlier surgical labels. Its captions identify that historical status. The original Figure4 module supplies reusable panel functions. Use render_corrected_esd.py for the corrected-label independent panels.

## Checks

    python -m unittest discover -s tests -v
    python scripts/check_code_only.py

Checks use generated synthetic arrays and source inspection. Validation against the private frozen cohort is documented locally by the data custodian and is excluded from this repository.
