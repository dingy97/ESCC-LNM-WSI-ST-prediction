"""Command-line analysis workflow."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="escc-lnm", description="WSI-to-RNA lymph node metastasis prediction")
    commands = parser.add_subparsers(dest="command", required=True)
    bridge = commands.add_parser("train-bridge", help="Train and cross-validate the image-to-RNA Ridge bridge")
    bridge.add_argument("--input", type=Path, required=True)
    bridge.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract", help="Extract FOV embeddings and patient RNA features")
    extract.add_argument("--slides", type=Path, required=True)
    extract.add_argument("--encoder", type=Path, required=True)
    extract.add_argument("--bridge", type=Path, required=True)
    extract.add_argument("--genes", type=Path)
    extract.add_argument("--config", type=Path, default=Path("config/settings.json"))
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--skip-failed", action="store_true")
    training = commands.add_parser("train", help="Optimize and compare twelve patient-level classifiers")
    training.add_argument("--features", type=Path, required=True)
    training.add_argument("--patients", type=Path, required=True)
    training.add_argument("--config", type=Path, default=Path("config/search.json"))
    training.add_argument("--output", type=Path, required=True)
    prediction = commands.add_parser("predict", help="Apply a frozen classifier to patient RNA features")
    prediction.add_argument("--features", type=Path, required=True)
    prediction.add_argument("--classifier", type=Path, required=True)
    prediction.add_argument("--model-info", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Calculate patient-level performance")
    evaluation.add_argument("--scores", type=Path, required=True)
    evaluation.add_argument("--labels", type=Path, required=True)
    evaluation.add_argument("--threshold", type=float, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--bootstrap-reference-labels", type=Path)
    explanation = commands.add_parser("explain", help="Select representative cases and compute FOV contributions")
    explanation.add_argument("--slides", type=Path, required=True)
    explanation.add_argument("--labels", type=Path, required=True)
    explanation.add_argument("--classifier", type=Path, required=True)
    explanation.add_argument("--model-info", type=Path, required=True)
    explanation.add_argument("--threshold", type=float, required=True)
    explanation.add_argument("--output", type=Path, required=True)
    figures = commands.add_parser("figures", help="Export PDF, SVG and PNG figures")
    figures.add_argument("--kind", choices=["evaluation", "classifiers", "attribution"], required=True)
    figures.add_argument("--input", type=Path, required=True)
    figures.add_argument("--scores", type=Path)
    figures.add_argument("--labels", type=Path)
    figures.add_argument("--output", type=Path)
    prepare = commands.add_parser("prepare-model", help="Write the model and gene metadata for inference")
    prepare.add_argument("--classifier", type=Path, required=True)
    prepare.add_argument("--genes", type=Path, required=True)
    prepare.add_argument("--thresholds", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train-bridge":
        from .bridge import train_bridge
        train_bridge(args.input, args.output)
    elif args.command == "extract":
        from .inference import extract
        extract(args.slides, args.encoder, args.bridge, args.config, args.output, args.genes, args.skip_failed)
    elif args.command == "train":
        from .training import train
        train(args.features, args.patients, args.config, args.output)
    elif args.command == "predict":
        from .inference import predict
        predict(args.features, args.classifier, args.model_info, args.output)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        if not 0 <= args.threshold <= 1:
            parser.error("threshold must be in [0, 1]")
        evaluate(args.scores, args.labels, args.output, args.threshold, bootstrap_reference=args.bootstrap_reference_labels)
    elif args.command == "explain":
        from .attribution import explain
        explain(args.slides, args.labels, args.classifier, args.model_info, args.threshold, args.output)
    elif args.command == "prepare-model":
        from .io import read_json, write_json, sha256
        from .models import load_model, save_model
        genes = read_json(args.genes)
        genes = genes["gene_names"] if isinstance(genes, dict) else genes
        thresholds = read_json(args.thresholds)
        if not isinstance(thresholds, dict) or not all(0 <= float(v) <= 1 for v in thresholds.values()):
            parser.error("thresholds must map working-point names to values in [0, 1]")
        model = load_model(args.classifier)
        model.gene_names = genes
        if hasattr(model, "fit_patients"):
            del model.fit_patients
        args.output.mkdir(parents=True, exist_ok=True)
        save_model(model, args.output/"classifier.joblib")
        write_json(args.output/"model.json", {
            "gene_names":genes, "selected_genes":[genes[i] for i in model.features.genes],
            "thresholds":thresholds, "recipe":model.recipe, "model_sha256":sha256(args.output/"classifier.joblib"),
        })
    elif args.command == "figures":
        from .plots import evaluation_figures, classifier_figures, case_figures
        if args.kind == "attribution":
            case_figures(args.input)
        elif args.kind == "classifiers":
            if args.output is None:
                parser.error("--output is required for classifier figures")
            classifier_figures(args.input, args.output)
        else:
            if args.output is None or args.scores is None or args.labels is None:
                parser.error("evaluation figures require --scores, --labels and --output")
            evaluation_figures(args.scores, args.labels, args.input, args.output)


if __name__ == "__main__":
    main()
