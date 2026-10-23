"""Frozen five-degree-block hash assignment and TRAIN/CAL/TEST rotations."""
import hashlib
import numpy as np

SALT = 'R4-internal-spatial-cv-20261014-v1'



def assign_blocks(blocks):
    def block_hash(block):
        return hashlib.sha256((SALT + '|' + block).encode('utf-8')).hexdigest()
    ordered = sorted(set(blocks), key=lambda b: (block_hash(b), b))
    return {b: rank % 5 for rank, b in enumerate(ordered)}, [
        {'block': b, 'block_hash': block_hash(b), 'hash_rank': i, 'spatial_group': i % 5}
        for i, b in enumerate(ordered)]



def roles_for_fold(test_folds, outer_fold):
    values = np.asarray(test_folds, int)
    return np.where(values == outer_fold, 'test', np.where(values == (outer_fold + 1) % 5, 'calibration', 'train'))



def block_id(latitude, longitude):
    if not np.isfinite(latitude) or not np.isfinite(longitude) or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError('Finite geographic coordinates are required.')
    return f'{int(np.floor(latitude/5))}_{int(np.floor(longitude/5))}'
