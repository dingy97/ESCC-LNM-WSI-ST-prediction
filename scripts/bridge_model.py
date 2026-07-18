"""
Bridge model architecture: ResNet50 encoder + GenePredictor MLP.
Trained on TMA H&E patches paired with spatial transcriptomics gene expression.
"""
import torch
import torch.nn as nn
import torchvision
import joblib
import numpy as np
from torchvision import transforms
from PIL import Image


class GenePredictor(nn.Module):
    """MLP that maps ResNet50 features (2048-dim) to gene expression (241-dim)."""
    def __init__(self, in_dim=2048, out_dim=241):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_bridge(model_path, device=None):
    """Load the Image2Gene bridge model.

    Args:
        model_path: Path to image2gene_bridge.pth
        device: torch device (default: cuda if available, else cpu)

    Returns:
        dict with keys: encoder, predictor, target_genes, gene_mean, gene_std
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Build encoder
    resnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.to(device).eval()

    # Build predictor
    predictor = GenePredictor(2048, len(checkpoint["target_genes"]))
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.to(device).eval()

    return {
        "encoder": encoder,
        "predictor": predictor,
        "target_genes": checkpoint["target_genes"],
        "gene_mean": checkpoint["gene_mean"],
        "gene_std": checkpoint["gene_std"],
    }


def load_classifier(model_path):
    """Load the LNM classifier.

    Args:
        model_path: Path to lnm_classifier_full.pkl

    Returns:
        dict with keys: lr (LogisticRegression), scaler (StandardScaler), target_genes
    """
    return joblib.load(model_path)


def get_transform():
    """Get the standard image transform for ResNet50."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
