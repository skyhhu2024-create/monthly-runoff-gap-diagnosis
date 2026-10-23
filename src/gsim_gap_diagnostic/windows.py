"""Outcome-blind original historical-window selection and replay."""
import hashlib
import numpy as np

SALT = 'information-transfer-window-matching-20261013-v1'



def _hash(*parts):
    return hashlib.sha256((SALT + '|' + '|'.join(map(str, parts))).encode()).hexdigest()



def _inputs(y, d, query):
    y = np.asarray(y, dtype=float).copy()
    d = np.asarray(d, dtype=float).copy()
    query = np.asarray(query, dtype=int).reshape(-1)
    if y.ndim != 1 or d.ndim != 2 or d.shape[0] != len(y):
        raise ValueError('Expected target T and donors T x K.')
    if not len(query) or np.any(np.diff(query) != 1) or query[0] < 0 or query[-1] >= len(y):
        raise ValueError('Query must be a nonempty contiguous index vector inside T.')
    y[query] = np.nan
    return y, d, query



def make_replay(y, d, h, query, position):
    """Return (inner target, inner donors, historical query indices).

    h is a candidate dictionary or an integer start. The caller may save the
    genuinely visible historical target values separately for scoring. Leading
    replay uses only later target support; trailing replay only earlier support.
    Donor availability is intersected with the outer query's availability.
    """
    y, d, query = _inputs(y, d, query)
    if position not in {'interior', 'leading', 'trailing'}:
        raise ValueError('position must be interior, leading or trailing')
    start = int(h['start'] if isinstance(h, dict) else h)
    stop = start + len(query)
    if start < 0 or stop > len(y):
        raise ValueError('Historical query is outside T.')
    hq = np.arange(start, stop, dtype=int)
    if np.intersect1d(hq, query).size:
        raise ValueError('Historical query overlaps the hidden outer query.')
    query_availability = np.isfinite(d[query]).copy()
    y[hq] = np.nan
    if position == 'leading':
        y[:start] = np.nan
    elif position == 'trailing':
        y[stop:] = np.nan
    d[hq] = np.where(query_availability, d[hq], np.nan)
    assert np.isnan(y[query]).all()
    return y, d, hq



def _mean_paired(y, d):
    if d.shape[1] == 0:
        return 0.0
    return float(np.mean(np.sum(np.isfinite(y[:, None]) & np.isfinite(d), axis=0)))



def _coverage(d, indices):
    return float(np.isfinite(d[indices]).mean()) if d.shape[1] else 0.0



def _relative_difference(a, b):
    return float(abs(a - b) / max(a, b, 1.0))



def _greedy(candidates, max_windows):
    chosen = []
    for candidate in candidates:
        if all(candidate['end_exclusive'] <= c['start'] or candidate['start'] >= c['end_exclusive'] for c in chosen):
            chosen.append(candidate)
            if len(chosen) == max_windows:
                break
    return chosen



def select_windows(y_visible, d_visible, query, months, position, case_id, max_windows=4):
    """Select equally numerous matched and SHA-random replay windows.

    Distance is the mean of four dimensionless terms in [0, 1]: circular
    start-month distance / 6; relative retained-target-count difference;
    relative mean paired-month difference across donors; absolute donor-query
    coverage difference. Target and paired differences use max(a,b,1).
    """
    y, d, query = _inputs(y_visible, d_visible, query)
    months = np.asarray(months, dtype=int)
    if months.shape != y.shape:
        raise ValueError('months must have length T.')
    if position not in {'interior', 'leading', 'trailing'}:
        raise ValueError('Unknown boundary position.')
    if not isinstance(max_windows, (int, np.integer)) or max_windows < 1:
        raise ValueError('max_windows must be a positive integer.')
    length = len(query)
    current_n = int(np.isfinite(y).sum())
    current_pair = _mean_paired(y, d)
    current_coverage = _coverage(d, query)
    complete = np.convolve(np.isfinite(y).astype(int), np.ones(length, dtype=int), mode='valid') == length
    candidates = []
    for start in np.flatnonzero(complete):
        start = int(start)
        inner_y, inner_d, hq = make_replay(y, d, start, query, position)
        inner_n = int(np.isfinite(inner_y).sum())
        if inner_n < 12:
            continue
        # Interior replay needs target support on each side of its own gap.
        if position == 'interior' and not (np.isfinite(inner_y[:start]).any() and np.isfinite(inner_y[start + length:]).any()):
            continue
        inner_pair = _mean_paired(inner_y, inner_d)
        replay_coverage = _coverage(inner_d, hq)
        month_distance = abs(int(months[start]) - int(months[query[0]])) % 12
        components = {
            'season': float(min(month_distance, 12 - month_distance) / 6.0),
            'target_support': _relative_difference(inner_n, current_n),
            'paired_support': _relative_difference(inner_pair, current_pair),
            'query_donor_coverage': abs(replay_coverage - current_coverage),
        }
        candidates.append({
            'start': start, 'end_exclusive': start + length,
            'distance': float(np.mean(list(components.values()))),
            'distance_components': components,
            'inner_n_target': inner_n, 'inner_mean_paired_months': inner_pair,
            'replay_gap_donor_coverage': replay_coverage,
        })
    matched = _greedy(sorted(candidates, key=lambda c: (c['distance'], _hash(case_id, 'matched-tie', c['start']))), max_windows)
    unmatched = _greedy(sorted(candidates, key=lambda c: _hash(case_id, 'unmatched', c['start'])), max_windows)
    same_n = min(len(matched), len(unmatched))
    matched, unmatched = matched[:same_n], unmatched[:same_n]
    return {
        'matched': matched, 'unmatched': unmatched,
        'eligible_count': len(candidates), 'equal_window_count': same_n,
        'current_n_target': current_n, 'current_mean_paired_months': current_pair,
        'current_gap_donor_coverage': current_coverage,
        'matched_mean_distance': float(np.mean([c['distance'] for c in matched])) if same_n else None,
        'unmatched_mean_distance': float(np.mean([c['distance'] for c in unmatched])) if same_n else None,
        'distance_definition': 'Equal mean: month circular distance/6, relative target count, relative mean paired count, absolute gap donor coverage. Relative difference = abs(a-b)/max(a,b,1).',
        'selection_inputs': ['target finite mask', 'donor finite mask', 'query indices', 'calendar months', 'boundary position', 'case_id SHA'],
    }

