"""Input tables and model output files."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, float_precision="round_trip")


def read_features(path):
    frame = read_table(path)
    if "Patient_ID" in frame:
        frame = frame.set_index("Patient_ID")
    frame.index = frame.index.astype(str)
    if frame.empty or frame.index.has_duplicates or frame.index.isna().any():
        raise ValueError("Feature rows require unique patient identifiers")
    if frame.columns.has_duplicates or not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError("Feature columns must be unique and finite")
    return frame


def seed_for(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], "little")


def recipe_key(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]
