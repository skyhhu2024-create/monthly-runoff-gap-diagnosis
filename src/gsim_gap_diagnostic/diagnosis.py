"""TRAIN probability fitting, independent CAL policy selection, visible prediction.

M retains the original increasing isotonic fit and P2 CAL85/80 selection rule.
H delegates its scientific fit to the original spline-logistic/Platt functions.
TEST outcomes are refused by the prediction interface.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ._io import clean, write_json
from ._history_model import as_bool, valid_labels

BASE = ['hist_valid_count', 'hist_R_median', 'hist_NSE_median',
        'hist_joint_rate', 'hist_R_iqr', 'hist_NSE_iqr']
KEYS = ['case_id', 'algorithm', 'seed']
FORBIDDEN = {'R', 'NSE', 'pass_R', 'pass_joint', 'evaluable', 'model_available',
    'target', 'label', 'current_truth', 'y_true', 'probability', 'adopted', 'rank_score'}


def eligible(frame, probability):
    return np.asarray(as_bool(frame.diagnostic_available) &
        pd.to_numeric(frame.hist_valid_count, errors='raise').ge(1).to_numpy() &
        np.isfinite(probability), bool)


def validate_visible(frame, method='M'):
    forbidden = set(frame) & FORBIDDEN
    if forbidden: raise ValueError('Current outcomes or previous decisions are not prediction inputs: '+str(sorted(forbidden)))
    required = set(KEYS + ['station_id', 'diagnostic_available', 'hist_window_count', 'hist_valid_count', 'hist_NSE_median'])
    if method == 'H': required.update(BASE)
    if required - set(frame): raise ValueError('Missing visible columns: '+str(sorted(required-set(frame))))
    if frame[KEYS+['station_id']].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Null or duplicate case/algorithm/seed identities.')
    as_bool(frame.diagnostic_available)
    count = pd.to_numeric(frame.hist_valid_count, errors='raise').to_numpy(float)
    selected = pd.to_numeric(frame.hist_window_count, errors='raise').to_numpy(float)
    if not (np.isfinite(count).all() and np.isfinite(selected).all() and
            np.all((0 <= count) & (count <= selected) & (selected <= 4)) and
            np.all(count == np.floor(count)) and np.all(selected == np.floor(selected))):
        raise ValueError('Historical evidence counts must be integers 0<=valid<=selected<=4.')
    score = pd.to_numeric(frame.hist_NSE_median, errors='raise').to_numpy(float)
    if np.isinf(score).any() or np.isfinite(score[count == 0]).any():
        raise ValueError('Missing historical evidence must remain undefined, not filled with zero.')
    if method == 'H':
        for c in BASE[1:]:
            z = pd.to_numeric(frame[c], errors='raise').to_numpy(float)
            if np.isinf(z).any() or np.isfinite(z[count == 0]).any():
                raise ValueError('Invalid historical feature: '+c)


def choose_policy(frame, probability):
    """Exact original P2 rule, complete score ties, all registered CAL denominators."""
    score = np.asarray(probability, float)
    use = eligible(frame, score)
    success = valid_labels(frame) & frame.pass_joint.eq(1).to_numpy()
    candidates = np.unique(score[np.isfinite(score)])
    if not len(candidates):
        return {'threshold': None, 'calibration_target_unmet': True,
            'selection_status': 'calibration_targets_not_estimable', 'n_candidates': 0}
    order = np.argsort(score[use]); s = score[use][order]; y = success[use][order]
    i = np.searchsorted(s, candidates, side='left'); n = len(s)-i
    cs = np.r_[0, np.cumsum(y)]; ns = cs[-1]-cs[i]
    precision = np.divide(ns, n, out=np.full(len(n), np.nan), where=n > 0)
    recall = ns/success.sum() if success.sum() else np.full(len(n), np.nan)
    coverage = n/len(frame); balanced = np.minimum(precision/.85, recall/.8)
    table = pd.DataFrame({'threshold': candidates, 'precision': precision,
        'retention': recall, 'coverage': coverage, 'balanced_score': balanced})
    feasible = table.loc[(precision >= .85) & (recall >= .8)]
    if len(feasible):
        chosen = feasible.sort_values(['coverage','precision','threshold'], ascending=[False,False,True]).iloc[0]
        unmet = False
    else:
        finite = table.loc[np.isfinite(balanced)]
        if not len(finite):
            return {'threshold': None, 'calibration_target_unmet': True,
                'selection_status': 'calibration_targets_not_estimable', 'n_candidates': len(candidates)}
        chosen = finite.sort_values(['balanced_score','precision','coverage','threshold'], ascending=[False,False,False,True]).iloc[0]
        unmet = True
    return clean({'threshold': float(chosen.threshold), 'calibration_target_unmet': unmet,
        'selection_status': 'calibration_target_unmet_max_min_fallback' if unmet else 'both_calibration_targets_met',
        'n_candidates': len(candidates), 'selection_metrics': chosen.to_dict()})


def load_model(path, *, trusted_pickle=False):
    path = Path(path)
    if path.suffix == '.json': return json.loads(path.read_text('utf-8-sig'))
    if not trusted_pickle:
        raise ValueError('H pickle models execute Python during loading; use only your own file and explicitly enable trusted_pickle.')
    import pickle
    with path.open('rb') as stream: return pickle.load(stream)


def probability(frame, model):
    if model['method'] == 'M':
        score = pd.to_numeric(frame.hist_NSE_median, errors='raise').to_numpy(float)
        use = eligible(frame, score); p = np.full(len(frame), np.nan)
        iso = model['isotonic']; x = np.asarray(iso['x_thresholds'], float); y = np.asarray(iso['y_thresholds'], float)
        if not len(x) or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or np.any((y < 0) | (y > 1)):
            raise ValueError('Malformed bounded increasing isotonic model.')
        p[use] = np.interp(score[use], x, y)
        return p
    from ._history_model import predict
    return predict(model['history_model'], frame)[1]


def diagnose(frame, model):
    if isinstance(model, (str, Path)): model = load_model(model)
    validate_visible(frame, model['method'])
    if not frame.algorithm.eq(model['algorithm']).all() or not frame.seed.eq(int(model['seed'])).all():
        raise ValueError('Reconstruction algorithm/seed differs from the frozen diagnostic model.')
    p = probability(frame, model); use = eligible(frame, p)
    threshold = model['policy']['threshold']
    adopted = np.zeros(len(frame), bool) if threshold is None else use & (p >= threshold)
    keep = [c for c in ['case_id','station_id','block','algorithm','seed','hist_window_count','hist_valid_count','hist_NSE_median','diagnostic_available','cv_test_fold'] if c in frame]
    result = frame[keep].copy(); result['method'] = model['method']; result['event'] = model.get('event','original_event')
    result['probability'] = p; result['eligible'] = use; result['adopted'] = adopted
    result['threshold'] = threshold
    result['decision'] = np.where(~use, 'evidence_insufficient', np.where(adopted, 'accept', 'low_probability_reject'))
    result['calibration_target_unmet'] = model['policy'].get('calibration_target_unmet', False)
    return result


def fit_diagnostic(features, labels, method='M', event='original_event'):
    """Fit exactly one algorithm and seed, from spatially separate TRAIN and CAL."""
    if method not in ('M','H'): raise ValueError('method must be M or H.')
    validate_visible(features, method)
    for c in ['block','split']:
        if c not in features: raise ValueError('Missing development column: '+c)
    if set(features.split) != {'train','calibration'}:
        raise ValueError('Fit accepts only explicit train and calibration rows; no TEST rows.')
    if features.algorithm.nunique() != 1 or features.seed.nunique() != 1:
        raise ValueError('Fit one reconstruction algorithm and seed at a time.')
    for c in KEYS+['R','NSE','pass_R','pass_joint','evaluable','model_available']:
        if c not in labels: raise ValueError('Missing development label: '+c)
    if labels.duplicated(KEYS).any(): raise ValueError('Duplicate labels.')
    if set(map(tuple, labels[KEYS].to_numpy())) != set(map(tuple, features[KEYS].to_numpy())):
        raise ValueError('Development labels must exactly match development feature identities.')
    tr = features.loc[features.split.eq('train')]; ca = features.loc[features.split.eq('calibration')]
    if set(tr.block) & set(ca.block) or set(tr.station_id) & set(ca.station_id):
        raise ValueError('TRAIN and CAL must have disjoint stations and spatial blocks.')
    wanted = KEYS+['R','NSE','pass_R','pass_joint','evaluable','model_available']
    tr = tr.merge(labels[wanted], on=KEYS, validate='one_to_one'); ca = ca.merge(labels[wanted], on=KEYS, validate='one_to_one')
    model = {'format_version':'1.0','method':method,'algorithm':features.algorithm.iloc[0],
        'seed':int(features.seed.iloc[0]),'event':event,'score_column':'hist_NSE_median',
        'fit_audit':{'train_registered':len(tr),'cal_registered':len(ca),
                     'train_blocks':tr.block.nunique(),'cal_blocks':ca.block.nunique(),'blocks_disjoint':True}}
    if method == 'M':
        from sklearn.isotonic import IsotonicRegression
        use = valid_labels(tr) & eligible(tr, tr.hist_NSE_median.to_numpy(float))
        if not use.any(): raise ValueError('No eligible, known TRAIN evidence.')
        estimator = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
        estimator.fit(tr.loc[use,'hist_NSE_median'].to_numpy(float), tr.loc[use,'pass_joint'].to_numpy(float))
        model['isotonic'] = {'x_thresholds':estimator.X_thresholds_.tolist(), 'y_thresholds':estimator.y_thresholds_.tolist()}
    else:
        from ._history_model import fit_model
        # Original H weights station x arm equally; a native deployment uses one arm.
        tr = tr.copy(); ca = ca.copy()
        if 'arm' not in tr: tr['arm'] = 'native'
        if 'arm' not in ca: ca['arm'] = 'native'
        model['history_model'] = fit_model(tr, ca, BASE, int(model['seed']))
    model['policy'] = choose_policy(ca, probability(ca, model))
    return model


def save_model(path, model):
    path = Path(path)
    if model['method'] == 'M': write_json(path, model)
    else:
        import pickle
        with path.open('wb') as stream: pickle.dump(model, stream, pickle.HIGHEST_PROTOCOL)

