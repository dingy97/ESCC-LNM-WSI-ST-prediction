"""Stage 3 paths and deterministic, patient-level operations."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '10_lnm_sampling_corrected'
sys.path.insert(0, str(ROOT / '08_lnm_modeling/runtime_deps'))
sys.path.insert(0, str(ROOT / '03_bridge_training/scripts'))
SEED = 20261104

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_status(path, value):
    """A transient Windows reader lock must not abort a scientific computation."""
    import time
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    write_json(temporary,value)
    for attempt in range(20):
        try:
            temporary.replace(path)
            return True
        except PermissionError:
            time.sleep(.05)
    print(f'Status publishing delayed by reader lock; payload retained: {temporary}',flush=True)
    return False

def stable_seed(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], 'little')

def patient_mean(features, patient_ids):
    import numpy as np
    patient_ids = np.asarray(patient_ids, dtype=str)
    ids = np.unique(patient_ids)
    return ids, np.asarray([features[patient_ids == p].mean(0) for p in ids])

def make_splits(ids, y, n=5, seed=SEED):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Split input must have exactly one row per patient')
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(ids, y))

def bootstrap_indices(y, repeats=2000, seed=SEED):
    import numpy as np
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    a, b = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    return np.asarray([np.r_[rng.choice(a, len(a)), rng.choice(b, len(b))] for _ in range(repeats)])
