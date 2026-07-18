"""
Example: Run LNM prediction on a sample WSI.

Usage:
    python example/example_inference.py /path/to/slide.ndpi
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.bridge_model import load_bridge, load_classifier
from scripts.predict import predict_lnm

# Paths
BRIDGE_PATH = "models/image2gene_bridge.pth"
CLASSIFIER_PATH = "models/lnm_classifier_full.pkl"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example/example_inference.py <slide_path>")
        sys.exit(1)

    slide_path = sys.argv[1]
    print(f"Loading models...")
    bridge = load_bridge(BRIDGE_PATH)
    classifier = load_classifier(CLASSIFIER_PATH)

    print(f"Analyzing: {slide_path}")
    result = predict_lnm(slide_path, bridge, classifier)

    print(f"\nResults:")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Probability: {result['probability']:.4f}")
    print(f"\nTop 10 inferred genes:")
    genes = sorted(result["gene_expression"].items(), key=lambda x: x[1], reverse=True)
    for gene, val in genes[:10]:
        print(f"  {gene}: {val:.4f}")
