# ESCC Spatial-Guided WSI LNM Prediction

[Spatial Transcriptomics-Guided Deep Learning for Occult Lymph Node Metastasis Detection in ESCC]

## Overview

This repository contains the trained models and inference code for predicting lymph node metastasis (LNM) from H&E-stained whole-slide images of esophageal squamous cell carcinoma (ESCC). The framework uses spatial transcriptomics as a molecular bridge to infer gene expression from H&E morphology, enabling molecularly-informed AI diagnostics.

## Model Performance

| Cohort | Method | AUC | Sensitivity | Specificity |
|--------|--------|-----|-------------|-------------|
| Internal validation (651 WSIs) | Bridge genes + LR | 0.845 | - | - |
| External ESD validation (600 WSIs) | Direct ResNet + LR | 0.838 | 95.5% | 75.0% |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from scripts.bridge_model import load_bridge, load_classifier
from scripts.predict import predict_lnm

# Load models
bridge = load_bridge("models/image2gene_bridge.pth")
classifier = load_classifier("models/lnm_classifier_full.pkl")

# Predict LNM from WSI
result = predict_lnm("path/to/slide.ndpi", bridge, classifier)
print(result)  # {"prediction": "LNM+", "probability": 0.87, "genes": {...}}
```

## Model Architecture

The pipeline consists of three components:

1. **ResNet50 Encoder** - Extracts 2048-dim visual features from H&E patches
2. **Image2Gene Bridge** - Translates visual features to 241 LNM-associated gene expressions
3. **LNM Classifier** - Logistic regression on inferred gene features

## Directory Structure

```
├── models/
│   ├── image2gene_bridge.pth    # Trained Image2Gene translator
│   ├── lnm_classifier_full.pkl  # LNM classifier (trained on 651 WSIs)
│   └── lnm_classifier.pkl       # LNM classifier (alternative)
├── scripts/
│   ├── predict.py               # Main inference script
│   └── bridge_model.py          # Model architecture definitions
├── example/
│   └── example_inference.py     # Usage example
├── requirements.txt
└── README.md
```

## Citation

If you use this model in your research, please cite:
```
[Paper citation - to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
