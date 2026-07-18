"""
Main inference script for LNM prediction from H&E WSI.
"""
import sys, os
import numpy as np
import random
import openslide
import torch
from PIL import Image
from sklearn.preprocessing import StandardScaler

from bridge_model import load_bridge, load_classifier, get_transform


def predict_lnm(
    slide_path,
    bridge,
    classifier,
    n_patches=50,
    patch_size=224,
    seed=42,
):
    """Predict LNM status from a WSI.

    Args:
        slide_path: Path to .ndpi or .svs file
        bridge: Loaded bridge model dict
        classifier: Loaded classifier dict
        n_patches: Number of patches to sample (default: 50)
        patch_size: Patch size in pixels (default: 224)
        seed: Random seed for reproducibility

    Returns:
        dict with keys: prediction, probability, gene_expression
    """
    random.seed(seed)
    np.random.seed(seed)

    device = next(bridge["encoder"].parameters()).device
    encoder = bridge["encoder"]
    predictor = bridge["predictor"]
    target_genes = bridge["target_genes"]
    gene_mean = bridge["gene_mean"]
    gene_std = bridge["gene_std"]

    lr = classifier["lr"]
    scaler = classifier["scaler"]
    transform = get_transform()

    # Open slide
    slide = openslide.OpenSlide(slide_path)

    # Read level-4 thumbnail
    lv = min(4, slide.level_count - 1)
    lw, lh = slide.level_dimensions[lv]
    img = slide.get_thumbnail((lw, lh))
    arr = np.array(img)
    slide.close()

    # Sample patches and extract features
    feats = []
    PS = patch_size
    for _ in range(n_patches * 3):
        if len(feats) >= n_patches:
            break
        px = random.randint(0, max(0, lw - PS))
        py = random.randint(0, max(0, lh - PS))
        sub = arr[py:py + PS, px:px + PS]
        if sub.shape[0] != PS or sub.shape[1] != PS:
            continue
        t = transform(Image.fromarray(sub)).unsqueeze(0).to(device)
        with torch.no_grad():
            f = encoder(t).squeeze()
        feats.append(f)

    if len(feats) < 10:
        raise RuntimeError(f"Only {len(feats)} valid patches sampled")

    # Mean pool and predict genes
    mean_feat = torch.stack(feats).mean(dim=0).unsqueeze(0)
    with torch.no_grad():
        gene_pred = predictor(mean_feat).detach().cpu().numpy().flatten()
    gene_pred = gene_pred * gene_std + gene_mean

    # Classify
    X = gene_pred.reshape(1, -1).astype(np.float64)
    X_s = scaler.transform(X)
    prob = lr.predict_proba(X_s)[0, 1]
    pred = "LNM+" if prob >= 0.5 else "LNM-"

    return {
        "prediction": pred,
        "probability": float(prob),
        "gene_expression": {
            gene: float(val) for gene, val in zip(target_genes, gene_pred)
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict LNM from H&E WSI")
    parser.add_argument("slide_path", help="Path to WSI file (.ndpi, .svs)")
    parser.add_argument("--bridge", default="models/image2gene_bridge.pth", help="Bridge model path")
    parser.add_argument("--classifier", default="models/lnm_classifier_full.pkl", help="Classifier path")
    parser.add_argument("--n_patches", type=int, default=50, help="Number of patches")
    args = parser.parse_args()

    bridge = load_bridge(args.bridge)
    classifier = load_classifier(args.classifier)
    result = predict_lnm(args.slide_path, bridge, classifier, n_patches=args.n_patches)

    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Top expressed genes:")
    genes = sorted(result["gene_expression"].items(), key=lambda x: x[1], reverse=True)
    for gene, val in genes[:10]:
        print(f"  {gene}: {val:.4f}")
