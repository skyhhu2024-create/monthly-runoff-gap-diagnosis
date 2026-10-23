"""Visible arrays -> current reconstruction and within-station historical evidence."""
import numpy as np
from .inputs import prepare_visible
from .replay import prepare_history
from .reconstruction import reconstruct
from .scores import history_summary, event_score, climatology_prediction
from .windows import select_windows


def run_case(y_visible, donors, query_indices, months, distances_km, *, case_id,
             station_id='target', block='unassigned', algorithm='rf', seed=20261014,
             position='interior', event='original_event', input_preflight=False):
    """No hidden current-gap values or labels are accepted by this interface.

    Arrays preserve calendar rows. Candidate donors are at most 100 nearest,
    archive-screened donor stations, sorted by geographic distance.
    """
    y = np.asarray(y_visible, float).copy(); d = np.asarray(donors, float).copy()
    q = np.asarray(query_indices, int); months = np.asarray(months, int)
    if q.ndim != 1 or not len(q) or np.any(np.diff(q) != 1) or q[0] < 0 or q[-1] >= len(y):
        raise ValueError('A contiguous in-calendar query is required.')
    if not np.isnan(y[q]).all(): raise ValueError('Mask current target observations before calling diagnosis.')
    if position == 'leading' and np.isfinite(y[:q[0]]).any(): raise ValueError('Leading gaps may use target support only after the gap.')
    if position == 'trailing' and np.isfinite(y[q[-1]+1:]).any(): raise ValueError('Trailing gaps may use target support only before the gap.')
    if position == 'interior' and not (np.isfinite(y[:q[0]]).any() and np.isfinite(y[q[-1]+1:]).any()):
        raise ValueError('Internal gaps require visible target observations on both sides.')
    windows = select_windows(y,d,q,months,position,case_id,max_windows=4)
    current = prepare_visible(y,d,q,months,distances_km,case_id)
    prediction = reconstruct(current,algorithm,seed)
    selected = windows['matched']; prepared_history = {}
    preflight = {'enabled': bool(input_preflight),'budget':len(selected),'pool_rejected':0,'active_rejected':0}
    if input_preflight:
        if algorithm != 'idw': raise ValueError('Published historical input preflight applies to IDW only.')
        from .controls import input_preflight_windows
        selected, prepared_history, preflight = input_preflight_windows(y,d,q,months,distances_km,position,case_id)
    histories = []
    for window in selected:
        start = window['start']
        if start in prepared_history: prepared = prepared_history[start]
        else: prepared,_ = prepare_history(y,d,q,start,months,distances_km,position,case_id+f':history:{start}')
        hq = prepared['query_indices']; truth = y[hq].copy()
        result = reconstruct(prepared,algorithm,seed)
        baseline = climatology_prediction(prepared['y_visible'], months, hq)
        histories.append({'start':int(start),'distance':window['distance'],
            **event_score(truth,result['prediction_query'],baseline,event)})
    stats = history_summary(histories)
    features = {'case_id':case_id,'station_id':station_id,'block':block,
        'algorithm':algorithm,'seed':int(seed),'position':position,'gap_length':len(q),
        'arm':'native','scheme':'matched', **current['summary'], **stats}
    return {'features':features,'history_records':histories,
        'query_indices':q,'prediction_query':prediction['prediction_query'],
        'model_summary':prediction['summary'],'history_windows':windows,
        'input_preflight':preflight,'current_truth_read':False}

