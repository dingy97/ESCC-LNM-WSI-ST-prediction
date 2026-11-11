"""Run one installed manuscript stage in a private project."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

STAGES = {
    "bridge": "03_bridge_training/scripts/train_bridge.py",
    "extract-wsi": "21_surgical_RNA_LNM_development/scripts/extract_ESD_features.py",
    "train12": "21_surgical_RNA_LNM_development/clinical_LNM_v2/scripts/train_all_12.py",
    "compare12": "21_surgical_RNA_LNM_development/clinical_LNM_v2/scripts/stage21_analyze.py",
    "overlap-sensitivity": "21_surgical_RNA_LNM_development/clinical_LNM_v2/scripts/fixed_candidate_sensitivity.py",
    "feature-count": "SCI论文最终组图素材/原始数据索引/RNA_feature_optimization/draw_rna_feature_optimization.py",
    "attribution": "SCI论文最终组图素材/24_典型病例风险热图/Scripts/stage24_attribution.py",
    "attribution-figures": "SCI论文最终组图素材/24_典型病例风险热图/Scripts/stage24_draw.py",
    "attribution-qa": "SCI论文最终组图素材/24_典型病例风险热图/Scripts/stage24_verify.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("stage_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.project_root.resolve() / STAGES[args.stage]
    if not script.is_file():
        raise FileNotFoundError(f"Install the analysis source first: {script}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    forwarded = args.stage_arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    subprocess.run([sys.executable, "-X", "utf8", str(script), *forwarded], env=env, check=True)


if __name__ == "__main__":
    main()
