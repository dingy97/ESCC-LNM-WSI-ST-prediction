# Private input contracts

Keep this workspace outside the Git checkout.

## Whole-slide images

Use a private manifest with unique slide identifiers, original file paths, patient/accession keys, dimensions, MPP, file size and modification time. All slides belonging to a patient share the same development partition. Folder names alone do not establish a patient-level endpoint.

The inherited physical FOV is approximately 512 µm on each axis. The precise floating-point dimensions and all tissue admission rules are read from the original private 10_lnm_sampling_corrected/config/frozen_analysis.json. Anisotropic X and Y MPP are preserved.

## Full-panel patient features

predict_cached.py accepts a Parquet table with one unique Patient_ID index and exactly the gene-name set declared in the private bridge bundle card. Columns are explicitly reordered by gene name. Each row must already use the frozen equal-FOV-then-equal-WSI aggregation.

The bridge bundle and classifier model locks reside under the original analysis paths. The final classifier object includes feature indices, the fitted scaler, LR and the sigmoid calibrator. These fitted artifacts remain private.

## Outcomes and folds

evaluate_predictions.py reads two files keyed by Patient_ID. The score file contains probability. The outcome file contains binary LNM. Keys must match exactly and remain unique. The full cohort is joined before evaluation.

Stage21 development also requires the surgical patient manifest with verified clinical LNM, inherited outer_fold, case mapping, bridge-overlap flags and the full-panel feature matrix. The source Data class checks identity and the original study dimensions. The sensitivity manifest must explicitly provide historical_label_discrepancy as a boolean field. Its positive records are determined by the data custodian.

The private frozen search plan contains per-context inputs and checksums. Generate its input_hashes and execution_contract for the installed source and actual local data. A source revision produces a new execution identity. Retain existing runs separately.

## Stage24 inputs

Use the corrected-label audit mapping, original frozen patient prediction file, original WSI manifest and extraction QC, frozen FOV caches, and model locks. Stage24 reproduces all case vectors before perturbation. Original cached coordinates are checked against deterministic resampling and patch admission. Missing coordinates require a successful replay of ordered RNA rows before use.

Do not send QA/Private, model checkpoints, examples from clinical H&E, patient prediction files or local configuration files to this repository.
