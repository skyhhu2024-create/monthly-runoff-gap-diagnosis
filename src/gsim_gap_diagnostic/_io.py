"""Portable, strict JSON and table I/O."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [clean(v) for v in value]
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def read_frame(path):
    keys = ['case_id', 'station_id', 'block', 'split', 'algorithm', 'source_population']
    return pd.read_csv(path, dtype={k: str for k in keys}, float_precision='round_trip',low_memory=False)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError('Output directory must be empty; preserve previous runs.')
    path.mkdir(parents=True, exist_ok=True)
    return path

