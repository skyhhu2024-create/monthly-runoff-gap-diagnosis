"""Original full 12-month footprint replay, with package-relative imports."""
import numpy as np
from .inputs import prepare_visible
from . import windows as _LEGACY

VERSION = 'context-replay-v2-20261014'



def prepare_history(y_visible, donors_visible, query_indices, history_start,
                    months, distances_km, position, case_id):
    y = np.asarray(y_visible, float)
    donors = np.asarray(donors_visible, float)
    query = np.asarray(query_indices, int)
    if not np.isnan(y[query]).all():
        raise ValueError('Current target query must already be all NaN.')
    start = int(history_start)
    inner, replay, hq = _LEGACY.make_replay(y, donors, start, query, position)
    prefix_before = np.isfinite(replay).sum()
    for offset in range(-11, 0):
        history_time = start + offset
        outer_time = int(query[0]) + offset
        if history_time < 0:
            continue
        allowed = np.isfinite(donors[outer_time]) if outer_time >= 0 else np.zeros(donors.shape[1], bool)
        replay[history_time] = np.where(allowed, replay[history_time], np.nan)
    removed = int(prefix_before - np.isfinite(replay).sum())
    assert np.all(~np.isfinite(replay) | np.isfinite(donors))
    assert np.isnan(inner[query]).all() and np.isnan(inner[hq]).all()
    prepared = prepare_visible(inner, replay, hq, months, distances_km, case_id)
    # A missing observation inside the calendar normally uses seasonal raw
    # imputation. A date outside the outer calendar instead requires padding.
    outer_context_time = query[:, None] + np.arange(-11, 1)[None, :]
    structural_padding = outer_context_time < 0
    for relative_row, lag in np.argwhere(structural_padding):
        row = int(hq[relative_row])
        prepared['X_sequence'][row, lag, 4:] = 0.
        prepared['X_sequence'][row, lag, 6::3] = 1.
    prepared['X_flat'] = prepared['X_sequence'].reshape(len(y), -1)
    # Shared harmonics retain the historical calendar; no outer target values.
    prepared['provenance']['history_context_replay'] = VERSION
    prepared['provenance']['structural_padding_query_override'] = True
    details = {'version':VERSION, 'prefix_candidate_cells_removed':removed,
               'outer_structural_padding_months':max(0, 11-int(query[0])),
               'structural_padding_query_timesteps':int(structural_padding.sum()),
               'current_target_hidden':True, 'historical_target_hidden':True,
               'donor_selection_after_full_context_mask':True,
               'historical_harmonics_preserved':True}
    return prepared, details



def input_differences(first, second):
    """Conservative actual-input equivalence, excluding descriptive provenance."""
    array_fields = ['X_sequence', 'X_monthly', 'y_standardized', 'observed_mask',
                    'query_indices', 'y_visible', 'months', 'selected_indices',
                    'active_indices', 'donor_values', 'fallback_prediction_all']
    differences = []
    for field in array_fields:
        a, b = np.asarray(first[field]), np.asarray(second[field])
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            differences.append(field)
    for field in ['y_mean', 'y_scale', 'fallback', 'donor_calibrations']:
        if first[field] != second[field]:
            differences.append(field)
    return differences

