"""Phikon extraction, RNA prediction and patient-level scoring."""
from contextlib import closing
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .io import read_features, read_json, sha256, write_json
from .models import load_model
from .sampling import slide_candidate_mask, candidate_fovs, read_patches


def gene_names(bridge, path=None):
    genes = getattr(bridge, "gene_names", None)
    if genes is None and path is not None:
        value = read_json(path)
        genes = value["gene_names"] if isinstance(value, dict) else value
    if genes is None or len(set(genes)) != len(genes):
        raise ValueError("Supply the ordered gene names for the bridge")
    return list(genes)


class PhikonEncoder:
    def __init__(self, directory, batch=24):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        from transformers import ViTModel
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the configured FP32 image encoder")
        torch.set_num_threads(1)
        torch.manual_seed(20261104)
        torch.cuda.manual_seed_all(20261104)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        self.model = ViTModel.from_pretrained(
            directory, add_pooling_layer=False, local_files_only=True
        ).eval().to("cuda").requires_grad_(False)
        self.batch = batch

    def encode(self, images, batch=None):
        import torch
        batch = self.batch if batch is None else batch
        result = []
        for start in range(0, len(images), batch):
            current = images[start:start+batch]
            try:
                values = torch.from_numpy(np.stack(current)).permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)/255
                mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
                scale = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
                with torch.inference_mode():
                    features = self.model(pixel_values=(values-mean)/scale).last_hidden_state[:, 0].cpu().numpy()
                result.append(features)
                del values
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                result.append(self.encode(current, max(1, batch//2)))
        return np.concatenate(result)


def prepare_slide(path, settings):
    import openslide
    with closing(openslide.OpenSlide(str(path))) as slide:
        mpp = np.asarray([float(slide.properties[key]) for key in
                          (openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y)])
        if not np.isfinite(mpp).all() or (mpp <= 0).any():
            raise ValueError("The WSI requires valid X and Y MPP")
        dimensions = slide.dimensions
        thumbnail = np.asarray(slide.get_thumbnail((2048, 2048)).convert("RGB"))
        mask, mask_qc = slide_candidate_mask(thumbnail, dimensions, mpp)
        coords, sampling = candidate_fovs(dimensions, mpp, mask, settings["FOV_um"], settings["max_FOVs"])
        images, groups, fractions = read_patches(slide, coords, mpp, settings["FOV_um"])
    if not len(images):
        raise ValueError("No tissue-positive image tiles")
    valid = np.unique(groups)
    tissue_area = mask.sum()*np.prod(np.asarray(dimensions)*mpp/np.asarray(mask.shape[::-1]))/1e6
    geometry = {
        "coords_level0":coords[valid], "patch_tissue_fractions":fractions[valid], "mpp":mpp,
        "dimensions":np.asarray(dimensions), "FOV_um":np.asarray(settings["FOV_um"]),
        "tissue_area_mm2":np.asarray(tissue_area),
    }
    return images, groups, valid, geometry, {"sampling":sampling, "mask":mask_qc}


def extract(manifest_path, encoder_directory, bridge_path, settings_path, output, genes_path=None, skip_failed=False):
    manifest = pd.read_csv(manifest_path, dtype={"Patient_ID":str, "slide_id":str})
    required = {"Patient_ID", "slide_id", "path"}
    if not required <= set(manifest) or manifest[list(required)].isna().any().any():
        raise ValueError("WSI manifest requires Patient_ID, slide_id and path")
    if manifest.slide_id.duplicated().any() or manifest.path.duplicated().any():
        raise ValueError("WSI identifiers and paths must be unique")
    settings = read_json(settings_path)["sampling"]
    bridge = load_model(bridge_path)
    genes = gene_names(bridge, genes_path)
    output = Path(output)
    (output/"fov_features").mkdir(parents=True, exist_ok=True)
    encoder = None
    encoder_identity = {
        p.name:sha256(p) for p in Path(encoder_directory).iterdir()
        if p.is_file() and p.suffix in {".json", ".bin", ".safetensors"}
    }
    if not encoder_identity:
        raise ValueError("No local encoder configuration or weights found")
    implementation = {
        name:sha256(Path(__file__).with_name(name)) for name in ("inference.py", "sampling.py")
    }
    bridge_checksum = sha256(bridge_path)
    rows = []
    for row in manifest.itertuples():
        source = Path(row.path)
        stat = source.stat()
        identifier = hashlib.sha256(row.slide_id.encode()).hexdigest()[:24]
        destination = output/"fov_features"/(identifier+".npz")
        metadata_path = destination.with_suffix(".json")
        identity = {
            "source_bytes":stat.st_size, "source_mtime_ns":str(stat.st_mtime_ns),
            "path":str(source.resolve()), "bridge_sha256":bridge_checksum,
            "settings":settings, "encoder":str(Path(encoder_directory).resolve()),
            "encoder_files":encoder_identity, "implementation":implementation,
        }
        if destination.exists() and metadata_path.exists() and read_json(metadata_path)["identity"] == identity:
            status = "PASS"
        else:
            try:
                images, groups, valid, geometry, qc = prepare_slide(source, settings)
                if encoder is None:
                    encoder = PhikonEncoder(encoder_directory, settings["batch"])
                features = encoder.encode(images)
                pooled = np.asarray([features[groups == k].mean(0) for k in valid], dtype="float32")
                with threadpool_limits(limits=2):
                    rna = bridge.predict(np.asarray(pooled, dtype=float))
                if rna.shape != (len(valid), len(genes)) or not np.isfinite(rna).all():
                    raise ValueError("Invalid bridge prediction")
                temporary = destination.with_suffix(".partial.npz")
                np.savez_compressed(temporary, Phikon=pooled, RNA=rna, gene_names=np.asarray(genes), **geometry)
                temporary.replace(destination)
                write_json(metadata_path, {"identity":identity, "QC":qc})
                status = "PASS"
            except Exception as error:
                if not skip_failed:
                    raise
                rows.append({"Patient_ID":row.Patient_ID, "slide_id":row.slide_id, "path":str(source),
                             "status":"FAILED", "reason":str(error), "cache":""})
                continue
        rows.append({"Patient_ID":row.Patient_ID, "slide_id":row.slide_id, "path":str(source),
                     "status":status, "reason":"", "cache":str(destination.resolve())})
        print(row.slide_id, status, flush=True)
    cached = pd.DataFrame(rows)
    cached.to_csv(output/"slides.csv", index=False)
    aggregate(cached[cached.status.eq("PASS")], output/"patient_rna.parquet", genes)


def aggregate(manifest, output, genes):
    if manifest.empty:
        raise ValueError("No successfully extracted slides are available")
    values = {}
    for row in manifest.itertuples():
        with np.load(row.cache, allow_pickle=False) as f:
            rna = f["RNA"]
            if list(f["gene_names"].astype(str)) != genes:
                raise ValueError("Inconsistent gene order among WSI caches")
            if rna.ndim != 2 or not len(rna) or not np.isfinite(rna).all():
                raise ValueError("Invalid FOV RNA cache")
            values.setdefault(row.Patient_ID, []).append(rna.mean(0))
    frame = pd.DataFrame(
        [np.asarray(values[k]).mean(0) for k in sorted(values)],
        index=pd.Index(sorted(values), name="Patient_ID"), columns=genes,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    return frame


def predict(features_path, classifier_path, model_info, output):
    features = read_features(features_path)
    model = load_model(classifier_path)
    info = read_json(model_info)
    if sha256(classifier_path) != info["model_sha256"]:
        raise ValueError("Classifier checksum does not match model.json")
    genes = info["gene_names"]
    if set(features.columns) != set(genes):
        raise ValueError("Input genes differ from the model gene schema")
    scores = model.predict_proba(features.loc[:, genes].to_numpy(float))[:, 1]
    result = pd.DataFrame({"Patient_ID":features.index, "probability":scores})
    for name, value in info["thresholds"].items():
        result["high_risk_"+name] = result.probability >= value
    result["model_sha256"] = info["model_sha256"]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.17g")
    return result
